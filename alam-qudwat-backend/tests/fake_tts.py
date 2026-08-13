"""Deterministic, zero-cost TTS stub for tests — no AWS/SageMaker calls."""
from __future__ import annotations

from typing import AsyncIterator

from app.services.tts import TTS_SAMPLE_FORMAT, TTSAudio


class FakeTTS:
    def __init__(
        self,
        chunks: list[bytes] | None = None,
        raise_error: Exception | None = None,
        sample_rate: int = 32000,
        channels: int = 1,
        sample_format: str = TTS_SAMPLE_FORMAT,
    ):
        self.chunks = chunks if chunks is not None else [b"\x00\x01" * 100]
        self.raise_error = raise_error
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_format = sample_format
        # Each call records the full text collected from the input stream
        # (real callers only ever send a one-shot stream today, but this
        # still works correctly for a multi-piece stream).
        self.calls: list[tuple[str, str | None]] = []

    async def speak(self, text_stream: AsyncIterator[str], *, voice_id: str | None = None) -> TTSAudio:
        text = "".join([piece async for piece in text_stream])
        self.calls.append((text, voice_id))
        if self.raise_error is not None:
            raise self.raise_error

        async def _chunks() -> AsyncIterator[bytes]:
            for chunk in self.chunks:
                yield chunk

        return TTSAudio(
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_format=self.sample_format,
            chunks=_chunks(),
        )
