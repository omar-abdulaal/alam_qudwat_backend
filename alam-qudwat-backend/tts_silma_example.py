import base64
import io
import json
import os
import wave
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SILMA_SAGEMAKER_ENDPOINT_NAME = os.environ["SILMA_SAGEMAKER_ENDPOINT_NAME"]

# 1 MB ~= 16.38 seconds of raw PCM at 32000 Hz int16 mono (actual size
# per second depends on whatever sample_rate SILMA returns for a call).
MAX_CHUNK_BYTES = 1024 * 1024
BYTES_PER_SAMPLE = 2  # int16 mono

sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
    config=Config(
        connect_timeout=5,
        read_timeout=70,
        retries={
            "max_attempts": 1,
            "mode": "standard",
        },
    ),
)


def _invoke_silma_sagemaker(text: str, voice_id: str = "pixel") -> bytes:
    """
    Calls the SageMaker endpoint (see inference.py). The container always
    returns JSON: {"audio_base64": <base64 of a COMPLETE WAV file>,
    "format": "wav", "sample_rate": <int>} — NOT headerless raw PCM, and
    NOT necessarily at a fixed rate. This returns the decoded WAV bytes;
    callers must parse them (see silma_tts_stream_pcm below) to get raw
    PCM samples plus the real sample_rate/channels/sample_width.
    """

    payload = {"text": text, "voice_id": voice_id}

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SILMA_SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

        body = json.loads(response["Body"].read())
        audio_b64 = body.get("audio_base64")

        if not audio_b64:
            raise RuntimeError("SageMaker response had no audio_base64 field.")

        return base64.b64decode(audio_b64)

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        raise RuntimeError(
            f"SageMaker Silma TTS error: {error_code}: {error_message}"
        ) from e


def _extract_pcm_from_wav(wav_bytes: bytes):
    """Strips the WAV container and returns (pcm_bytes, sample_rate,
    channels, sample_width) from the file's own fmt chunk."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return (
            wf.readframes(wf.getnframes()),
            wf.getframerate(),
            wf.getnchannels(),
            wf.getsampwidth(),
        )


def _iter_pcm_chunks_minimal(
    pcm_bytes: bytes,
    max_chunk_bytes: Optional[int] = None,
):
    """
    Yields audio in the fewest practical chunks.

    If pcm_bytes <= max_chunk_bytes, it yields exactly 1 chunk.
    Otherwise it splits into large aligned chunks.
    """

    if max_chunk_bytes is None:
        max_chunk_bytes = MAX_CHUNK_BYTES

    # Keep chunk size aligned to int16 sample boundaries.
    max_chunk_bytes = max(BYTES_PER_SAMPLE, max_chunk_bytes)
    max_chunk_bytes -= max_chunk_bytes % BYTES_PER_SAMPLE

    if len(pcm_bytes) <= max_chunk_bytes:
        yield pcm_bytes
        return

    for i in range(0, len(pcm_bytes), max_chunk_bytes):
        chunk = pcm_bytes[i:i + max_chunk_bytes]
        if chunk:
            yield chunk


async def silma_tts_stream_pcm(text: str, voice_id: str = "pixel"):
    """
    Async generator yielding raw PCM chunks (headerless — the WAV
    container SageMaker returns is decoded and stripped here), plus the
    real (sample_rate, channels, sample_width) discovered from that WAV
    file so a caller can label the stream correctly instead of assuming
    a fixed rate.
    """

    wav_bytes = _invoke_silma_sagemaker(text, voice_id)
    pcm_bytes, sample_rate, channels, sample_width = _extract_pcm_from_wav(wav_bytes)

    if not pcm_bytes:
        raise RuntimeError("SageMaker returned empty audio.")

    print(f"Silma SageMaker TTS: sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}")

    chunk_count = 0

    for chunk in _iter_pcm_chunks_minimal(pcm_bytes):
        chunk_count += 1
        yield chunk

    print(f"Silma SageMaker TTS chunks: {chunk_count}")
