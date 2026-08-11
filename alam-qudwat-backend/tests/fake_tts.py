"""Deterministic, zero-cost TTS stub for tests — no AWS/SageMaker calls."""
from __future__ import annotations

from typing import AsyncIterator


class FakeTTS:
    def __init__(self, chunks: list[bytes] | None = None, raise_error: Exception | None = None):
        self.chunks = chunks if chunks is not None else [b"\x00\x01" * 100]
        self.raise_error = raise_error
        self.calls: list[tuple[str, str | None]] = []

    async def speak(self, text: str, *, voice_id: str | None = None) -> AsyncIterator[bytes]:
        self.calls.append((text, voice_id))
        if self.raise_error is not None:
            raise self.raise_error
        for chunk in self.chunks:
            yield chunk
