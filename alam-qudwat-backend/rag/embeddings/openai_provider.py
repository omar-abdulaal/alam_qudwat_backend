"""OpenAI embedding provider (default backend for EmbeddingProvider)."""
from __future__ import annotations

from typing import Sequence

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from rag.config import get_settings


class OpenAIEmbeddingProvider:
    def __init__(self, model: str | None = None, dim: int | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file before running ingestion "
                "or retrieval — see .env.example."
            )
        self._client = OpenAI(api_key=key)

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        # API preserves input order.
        return [item.embedding for item in response.data]
