"""Deterministic, zero-cost STT stub for tests."""
from __future__ import annotations


class FakeSTT:
    def __init__(self, text: str = "نص تجريبي من الصوت"):
        self.text = text
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, *, audio_bytes: bytes, filename: str) -> str:
        self.calls.append((audio_bytes, filename))
        return self.text
