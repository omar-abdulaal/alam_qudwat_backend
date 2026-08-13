"""Text-to-speech via the org's own SILMA model on SageMaker (not OpenAI).

The deployed inference.py (see the SageMaker deployment scripts) always
returns a JSON body — {"audio_base64": <base64 of a COMPLETE WAV file>,
"format": "wav", "sample_rate": <int>} — regardless of what Accept header
is sent; it does not implement the older raw-PCM/"response_format":"pcm"
contract this module used to assume. This module invokes the endpoint,
decodes that WAV file, and strips its header so callers still get plain
headerless PCM samples — matching what the Flutter client expects and
what POST /api/v1/tts/speak has always documented (see
api_documentation.txt): raw PCM, not a WAV/MP3 container. Getting this
wrong (streaming the JSON bytes, or the whole WAV file including its
header, straight through as if it were already raw PCM) is what produces
loud static/noise on playback instead of speech.

sample_rate/channels/sample_format are per-request values discovered from
that WAV file's own header (authoritative for how to actually play the
samples back), not fixed constants — SILMA's native output rate isn't
guaranteed to be any particular value.

SageMaker itself is not internally streaming: one call always processes
its whole input text before returning any audio at all. `speak()` takes
an *async iterator* of text (not a single string) specifically to work
around that: incoming text is buffered and split into sentence-sized
segments as it arrives, each segment is synthesized as its own SageMaker
call, and that segment's audio starts streaming to the caller as soon as
it's ready — while later segments are still being read from the input
stream and/or synthesized. For a caller that already has the complete
text up front (app/api/routes/tts.py today), wrapping it as a one-shot
iterator still gets the benefit: the first sentence's audio can reach the
client well before the last sentence has even been sent to SageMaker,
instead of one giant call that must finish the entire answer first. The
same interface is also ready for a genuinely live source (e.g. piping
tokens straight from chat generation) with no further changes needed
here.

The module-level lock + boto3 client are shared across requests on
purpose, since the SageMaker endpoint may not tolerate high call
concurrency well — segments are synthesized one at a time even when
multiple requests overlap.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import re
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import get_app_settings

logger = logging.getLogger("app.tts")

# The only format app/api/routes/tts.py's X-Audio-Sample-Format contract
# and the Flutter raw-PCM player currently know how to handle.
TTS_SAMPLE_FORMAT = "pcm_s16le"
_BYTES_PER_SAMPLE = 2

# 1 MB ~= 16.38s of raw PCM at 32000 Hz int16 mono (varies with the
# actual sample rate SILMA returns for a given call) — HTTP transport
# chunking of one already-synthesized segment's audio, unrelated to
# _MAX_SEGMENT_CHARS below (that's about the *text* sent to SageMaker).
_MAX_CHUNK_BYTES = 1024 * 1024

# Every sentence/line is synthesized as its own segment (a separate
# SageMaker call) regardless of how short it is — this is a safety cap
# only, for the rare run-on chunk of text with no sentence terminator at
# all: past this many characters with no boundary found yet, force-split
# at the nearest word boundary so one segment/call never grows unbounded.
_MAX_SEGMENT_CHARS = 300

# Strips inline citation markers like "[1]" / "[١٢]" (ASCII or Arabic-Indic
# digits, optionally preceded by whitespace) — reads badly aloud otherwise.
_CITATION_MARKER = re.compile(r"\s*[\[［][0-9٠-٩]+[\]］]")

# A segment boundary is a sentence-ending punctuation mark or a newline
# (paragraph break) — matches Arabic and Latin sentence terminators.
_SEGMENT_BOUNDARY_RE = re.compile(r"[.!؟\n]")

_tts_lock = asyncio.Lock()
_sagemaker_runtime = None


class TTSUpstreamAudioError(RuntimeError):
    """SageMaker was reachable and responded, but its audio can't be
    served as-is (missing/malformed payload, empty audio, or a format
    this module doesn't know how to label/stream correctly). Maps to 502
    in app/api/routes/tts.py, distinct from plain RuntimeError (502 vs
    503) — the endpoint itself is up and configured; what it returned
    isn't usable."""


@dataclass(frozen=True)
class TTSAudio:
    sample_rate: int
    channels: int
    sample_format: str
    chunks: AsyncIterator[bytes]


def strip_citation_markers(text: str) -> str:
    return _CITATION_MARKER.sub("", text).strip()


def _get_sagemaker_runtime():
    global _sagemaker_runtime
    if _sagemaker_runtime is None:
        settings = get_app_settings()
        _sagemaker_runtime = boto3.client(
            "sagemaker-runtime",
            region_name=settings.aws_region,
            config=Config(
                connect_timeout=5,
                read_timeout=70,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )
    return _sagemaker_runtime


def _extract_pcm_from_wav(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    """Parses a complete WAV file and returns
    (pcm_data, sample_rate, channels, sample_width_bytes) — the WAV
    header's own fmt chunk, not any value we merely asked for, since
    that's what's actually needed to play the samples back correctly."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            pcm_data = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        raise TTSUpstreamAudioError(f"SageMaker returned an unparsable WAV file: {exc}") from exc
    return pcm_data, frame_rate, channels, sample_width


def _invoke_silma_sagemaker(text: str, voice_id: str, endpoint_name: str) -> tuple[bytes, int, int]:
    """Calls the SageMaker endpoint for a single text segment, decodes its
    JSON {audio_base64, format, sample_rate} response, and returns
    (pcm_data, sample_rate, channels) extracted from the WAV file's own
    header."""
    payload = {"text": text, "voice_id": voice_id}
    try:
        response = _get_sagemaker_runtime().invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        raw_body = response["Body"].read()
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        raise RuntimeError(f"SageMaker Silma TTS error: {error_code}: {error_message}") from e

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise TTSUpstreamAudioError(f"SageMaker response was not valid JSON: {exc}") from exc

    audio_b64 = body.get("audio_base64")
    if not audio_b64:
        raise TTSUpstreamAudioError("SageMaker response had no audio_base64 field.")

    try:
        wav_bytes = base64.b64decode(audio_b64, validate=True)
    except binascii.Error as exc:
        raise TTSUpstreamAudioError(f"SageMaker returned invalid base64 audio: {exc}") from exc

    pcm_data, sample_rate, channels, sample_width = _extract_pcm_from_wav(wav_bytes)

    if not pcm_data:
        raise TTSUpstreamAudioError("SageMaker returned empty audio.")
    if sample_width != _BYTES_PER_SAMPLE:
        raise TTSUpstreamAudioError(
            f"SILMA returned {sample_width * 8}-bit audio; only 16-bit PCM ({TTS_SAMPLE_FORMAT}) is supported."
        )
    if channels != 1:
        raise TTSUpstreamAudioError(f"SILMA returned {channels}-channel audio; only mono is supported.")

    return pcm_data, sample_rate, channels


def _iter_pcm_chunks_minimal(pcm_bytes: bytes, max_chunk_bytes: Optional[int] = None):
    """Yields one already-synthesized segment's audio in the fewest
    practical chunks, aligned to int16 sample boundaries."""
    if max_chunk_bytes is None:
        max_chunk_bytes = _MAX_CHUNK_BYTES
    max_chunk_bytes = max(_BYTES_PER_SAMPLE, max_chunk_bytes)
    max_chunk_bytes -= max_chunk_bytes % _BYTES_PER_SAMPLE

    if len(pcm_bytes) <= max_chunk_bytes:
        yield pcm_bytes
        return

    for i in range(0, len(pcm_bytes), max_chunk_bytes):
        chunk = pcm_bytes[i : i + max_chunk_bytes]
        if chunk:
            yield chunk


def _pop_ready_segment(buffer: str, max_chars: int = _MAX_SEGMENT_CHARS) -> tuple[Optional[str], str]:
    """If `buffer` has a complete sentence/paragraph boundary (or has
    simply grown past `max_chars` with no boundary yet), pop and return
    (segment, remaining_buffer). Otherwise (None, buffer) — wait for more
    text to arrive before deciding."""
    match = _SEGMENT_BOUNDARY_RE.search(buffer)
    if match:
        segment = buffer[: match.end()].strip()
        remainder = buffer[match.end() :]
        if segment:
            return segment, remainder
        # A boundary character with nothing meaningful before it (e.g. a
        # leading blank line) — consume it and keep looking.
        return _pop_ready_segment(remainder, max_chars)
    if len(buffer) > max_chars:
        split_at = buffer.rfind(" ", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        return buffer[:split_at].strip(), buffer[split_at:]
    return None, buffer


def split_into_speech_segments(text: str, max_chars: int = _MAX_SEGMENT_CHARS) -> list[str]:
    """Split a complete, already-known text into the same sentence-sized
    segments the streaming path (SilmaSageMakerTTS.speak) would produce
    incrementally. Exposed mainly for direct testing of the segmenting
    logic against full inputs."""
    segments: list[str] = []
    remaining = text
    while True:
        segment, remaining = _pop_ready_segment(remaining, max_chars)
        if segment is None:
            break
        segments.append(segment)
    tail = remaining.strip()
    if tail:
        segments.append(tail)
    return segments


class TextToSpeech(Protocol):
    async def speak(self, text_stream: AsyncIterator[str], *, voice_id: str | None = None) -> TTSAudio:
        """Synthesize the text carried by `text_stream` (an async
        iterator of text pieces — a single complete string wrapped as a
        one-shot iterator is fine) and return its audio metadata plus a
        chunked stream of raw PCM samples matching that metadata."""
        ...


class SilmaSageMakerTTS:
    async def speak(self, text_stream: AsyncIterator[str], *, voice_id: str | None = None) -> TTSAudio:
        settings = get_app_settings()
        endpoint_name = settings.silma_sagemaker_endpoint_name
        if not endpoint_name:
            raise RuntimeError(
                "SILMA_SAGEMAKER_ENDPOINT_NAME is not set. Add it to your .env file to enable TTS."
            )
        resolved_voice = voice_id or settings.tts_default_voice_id

        # A background task keeps reading `text_stream` and pushing
        # complete segments onto this queue as they become ready, fully
        # decoupled from how fast the caller below consumes synthesized
        # audio — this is what lets segment 2's text keep arriving (or
        # even keep being generated upstream, for a live source) while
        # segment 1's audio is already streaming out.
        segment_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _produce_segments() -> None:
            buffer = ""
            try:
                async for piece in text_stream:
                    buffer += piece
                    while True:
                        segment, buffer = _pop_ready_segment(buffer)
                        if segment is None:
                            break
                        # Stripped per-segment (not once on the whole text
                        # up front) so this also works for a live,
                        # incrementally-arriving stream (see
                        # app/api/routes/tts.py's /speak/live) where no
                        # complete text ever exists to pre-strip.
                        segment = strip_citation_markers(segment)
                        if segment:
                            await segment_queue.put(segment)
                tail = strip_citation_markers(buffer.strip())
                if tail:
                    await segment_queue.put(tail)
            finally:
                await segment_queue.put(None)  # sentinel: no more segments

        producer_task = asyncio.create_task(_produce_segments())

        async def _synthesize(segment: str) -> tuple[bytes, int, int]:
            async with _tts_lock:
                return await asyncio.to_thread(_invoke_silma_sagemaker, segment, resolved_voice, endpoint_name)

        # The first segment is synthesized eagerly (not lazily inside
        # _chunks below) because sample_rate/channels are needed up front
        # for the HTTP response headers, before any audio can be sent.
        try:
            first_segment = await segment_queue.get()
            if first_segment is None:
                await producer_task  # surface a producer exception, if any caused the empty stream
                raise TTSUpstreamAudioError("No text to synthesize.")
            pcm_data, sample_rate, channels = await _synthesize(first_segment)
        except Exception:
            producer_task.cancel()
            logger.exception("Silma SageMaker TTS error")
            raise

        async def _chunks() -> AsyncIterator[bytes]:
            segment_count = 1
            for chunk in _iter_pcm_chunks_minimal(pcm_data):
                yield chunk
                await asyncio.sleep(0)

            try:
                while True:
                    segment = await segment_queue.get()
                    if segment is None:
                        break
                    segment_count += 1
                    seg_pcm, _sr, _ch = await _synthesize(segment)
                    for chunk in _iter_pcm_chunks_minimal(seg_pcm):
                        yield chunk
                        await asyncio.sleep(0)
                await producer_task  # propagate any late producer-side exception
            except Exception:
                logger.exception("Silma SageMaker TTS error mid-stream")
                raise
            logger.info("Silma SageMaker TTS: %d segment(s), %d Hz", segment_count, sample_rate)

        return TTSAudio(
            sample_rate=sample_rate,
            channels=channels,
            sample_format=TTS_SAMPLE_FORMAT,
            chunks=_chunks(),
        )
