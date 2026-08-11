"""Deterministic, zero-cost embedding provider for tests.

Produces stable, content-dependent vectors (same text -> same vector)
without calling any external API, and counts how many times `embed` was
invoked so idempotency tests can assert "no embedding calls on an
unchanged re-ingestion run".
"""
from __future__ import annotations

import hashlib
from typing import Sequence


class FakeEmbeddingProvider:
    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.call_count = 0
        self.embedded_texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self.call_count += 1
        self.embedded_texts.extend(texts)
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat the 32-byte digest to fill `dim` floats in [0, 1).
        values = [(digest[i % len(digest)] / 255.0) for i in range(self.dim)]
        return values
