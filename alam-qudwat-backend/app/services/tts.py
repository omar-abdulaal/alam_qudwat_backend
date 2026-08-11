"""Text-to-speech via the org's own SILMA model on SageMaker (not OpenAI).

Adapted from the reference implementation (tts_silma_example.py) — same
invocation shape and chunked-delivery behavior, just reading endpoint/
region/voice from AppSettings instead of raw os.environ, and exposed
behind the `TextToSpeech` protocol so tests can substitute a fake.

The SageMaker call itself is synchronous and returns the *entire* audio
in one response (there's no true incremental synthesis happening on the
SILMA side); `_iter_pcm_chunks_minimal` only chunks it for HTTP transport
efficiency afterwards. The module-level lock + boto3 client are shared
across requests on purpose, matching the reference implementation, since
the SageMaker endpoint may not tolerate high call concurrency well.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator, Optional, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import get_app_settings

logger = logging.getLogger("app.tts")

TTS_SAMPLE_RATE = 32000
TTS_CHANNELS = 1
TTS_SAMPLE_FORMAT = "pcm_s16le"
_BYTES_PER_SAMPLE = 2

# 1 MB ~= 16.38s of raw PCM at 32000 Hz int16 mono.
_MAX_CHUNK_BYTES = 1024 * 1024

# Strips inline citation markers like "[1]" / "[١٢]" (ASCII or Arabic-Indic
# digits, optionally preceded by whitespace) — reads badly aloud otherwise.
_CITATION_MARKER = re.compile(r"\s*[\[［][0-9٠-٩]+[\]］]")

_tts_lock = asyncio.Lock()
_sagemaker_runtime = None


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


def _invoke_silma_sagemaker_pcm_32000(text: str, voice_id: str, endpoint_name: str) -> bytes:
    """Calls the SageMaker endpoint; expects raw PCM bytes: int16
    little-endian mono at TTS_SAMPLE_RATE Hz."""
    payload = {
        "text": text,
        "voice_id": voice_id,
        "sample_rate": TTS_SAMPLE_RATE,
        "response_format": "pcm",
    }
    try:
        response = _get_sagemaker_runtime().invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/octet-stream",
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        pcm_bytes = response["Body"].read()
        if not pcm_bytes:
            raise RuntimeError("SageMaker returned empty audio bytes.")
        return pcm_bytes
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        raise RuntimeError(f"SageMaker Silma TTS error: {error_code}: {error_message}") from e


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
    async def speak(self, text: str, *, voice_id: str | None = None) -> AsyncIterator[bytes]:
        """Yield raw PCM (int16 LE mono, TTS_SAMPLE_RATE Hz) chunks."""
        ...


class SilmaSageMakerTTS:
    async def speak(self, text: str, *, voice_id: str | None = None) -> AsyncIterator[bytes]:
        settings = get_app_settings()
        endpoint_name = settings.silma_sagemaker_endpoint_name
        if not endpoint_name:
            raise RuntimeError(
                "SILMA_SAGEMAKER_ENDPOINT_NAME is not set. Add it to your .env file to enable TTS."
            )
        resolved_voice = voice_id or settings.tts_default_voice_id

        try:
            async with _tts_lock:
                pcm_bytes = await asyncio.to_thread(
                    _invoke_silma_sagemaker_pcm_32000, text, resolved_voice, endpoint_name
                )

            chunk_count = 0
            for chunk in _iter_pcm_chunks_minimal(pcm_bytes):
                chunk_count += 1
                yield chunk
                await asyncio.sleep(0)

            logger.info("Silma SageMaker TTS: %d chunk(s)", chunk_count)
        except Exception:
            logger.exception("Silma SageMaker TTS error")
            raise
