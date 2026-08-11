import os
import json
import asyncio
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

TARGET_SAMPLE_RATE = 32000
BYTES_PER_SAMPLE = 2  # int16 mono

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SILMA_SAGEMAKER_ENDPOINT_NAME = os.environ["SILMA_SAGEMAKER_ENDPOINT_NAME"]

# 1 MB = about 16.38 seconds of raw PCM at 32000 Hz int16 mono.
# Increase this if your frontend/WebSocket can handle larger binary messages.
MAX_CHUNK_BYTES = str(1024 * 1024)

tts_lock = asyncio.Lock()

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


def _invoke_silma_sagemaker_pcm_32000(text: str, voice_id: str = "pixel") -> bytes:
    """
    Calls SageMaker endpoint and expects raw PCM bytes:
    int16 little-endian mono 32000 Hz.
    """

    payload = {
        "text": text,
        "voice_id": voice_id,
        "sample_rate": TARGET_SAMPLE_RATE,
        "response_format": "pcm",
    }

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SILMA_SAGEMAKER_ENDPOINT_NAME,
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
        raise RuntimeError(
            f"SageMaker Silma TTS error: {error_code}: {error_message}"
        ) from e


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
        max_chunk_bytes = int(MAX_CHUNK_BYTES)

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
    Same external behavior as your old function:
    async generator yielding raw PCM chunks.
    """

    try:
        async with tts_lock:
            pcm_bytes = await asyncio.to_thread(
                _invoke_silma_sagemaker_pcm_32000,
                text,
                voice_id,
            )

        chunk_count = 0

        for chunk in _iter_pcm_chunks_minimal(pcm_bytes):
            chunk_count += 1
            yield chunk
            await asyncio.sleep(0)

        print(f"Silma SageMaker TTS chunks: {chunk_count}")

    except Exception as e:
        print("SageMaker Silma TTS error:", e)
        raise