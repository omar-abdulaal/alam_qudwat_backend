"""Deterministic, zero-cost diacritizer stub for tests — no OpenAI calls."""
from __future__ import annotations


class FakeDiacritizer:
    def __init__(self, transform=None, raise_error: Exception | None = None):
        # Default transform is visibly distinct from the input so tests
        # can assert the diacritized (not plain) text reached the TTS call.
        self.transform = transform or (lambda text: f"[diacritized]{text}")
        self.raise_error = raise_error
        self.calls: list[str] = []

    async def diacritize(self, text: str) -> str:
        self.calls.append(text)
        if self.raise_error is not None:
            raise self.raise_error
        return self.transform(text)
