"""Embedding provider abstraction.

Ingestion and retrieval only depend on this protocol, never on a concrete
SDK, so the backing model can be swapped (e.g. for a local multilingual
sentence-transformers model) without touching either of them.
"""
from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        ...
