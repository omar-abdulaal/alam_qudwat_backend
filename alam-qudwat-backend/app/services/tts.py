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

The SageMaker call itself is synchronous and returns the *entire* audio
in one response (there's no true incremental synthesis happening on the
SILMA side); `_iter_pcm_chunks_minimal` only chunks it for HTTP transport
efficiency afterwards. The module-level lock + boto3 client are shared
across requests on purpose, since the SageMaker endpoint may not tolerate
high call concurrency well.
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
# actual sample rate SILMA returns for a given call).
_MAX_CHUNK_BYTES = 1024 * 1024

# Strips inline citation markers like "[1]" / "[١٢]" (ASCII or Arabic-Indic
# digits, optionally preceded by whitespace) — reads badly aloud otherwise.
_CITATION_MARKER = re.compile(r"\s*[\[［][0-9٠-٩]+[\]］]")

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
    """Calls the SageMaker endpoint, decodes its JSON {audio_base64,
    format, sample_rate} response, and returns (pcm_data, sample_rate,
    channels) extracted from the WAV file's own header."""
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
    """Yields audio in the fewest practical chunks, aligned to int16
    sample boundaries."""
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


class TextToSpeech(Protocol):
    async def speak(self, text: str, *, voice_id: str | None = None) -> TTSAudio:
        """Synthesize `text` and return its audio metadata plus a chunked
        stream of raw PCM samples matching that metadata."""
        ...


class SilmaSageMakerTTS:
    async def speak(self, text: str, *, voice_id: str | None = None) -> TTSAudio:
        settings = get_app_settings()
        endpoint_name = settings.silma_sagemaker_endpoint_name
        if not endpoint_name:
            raise RuntimeError(
                "SILMA_SAGEMAKER_ENDPOINT_NAME is not set. Add it to your .env file to enable TTS."
            )
        resolved_voice = voice_id or settings.tts_default_voice_id

        try:
            async with _tts_lock:
                pcm_data, sample_rate, channels = await asyncio.to_thread(
                    _invoke_silma_sagemaker, text, resolved_voice, endpoint_name
                )
        except Exception:
            logger.exception("Silma SageMaker TTS error")
            raise

        async def _chunks() -> AsyncIterator[bytes]:
            chunk_count = 0
            for chunk in _iter_pcm_chunks_minimal(pcm_data):
                chunk_count += 1
                yield chunk
                await asyncio.sleep(0)
            logger.info("Silma SageMaker TTS: %d chunk(s), %d Hz", chunk_count, sample_rate)

        return TTSAudio(
            sample_rate=sample_rate,
            channels=channels,
            sample_format=TTS_SAMPLE_FORMAT,
            chunks=_chunks(),
        )
