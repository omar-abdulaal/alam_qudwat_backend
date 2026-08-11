"""Deterministic, zero-cost chat LLM stub for tests — mirrors
tests/fake_embedder.py's role for the embeddings API."""
from __future__ import annotations

from typing import Any, AsyncIterator


class FakeChatLLM:
    def __init__(
        self,
        response_text: str = "هذا رد تجريبي للاختبار [1].",
        suggestions: list[str] | None = None,
    ):
        self.response_text = response_text
        self.suggestions = suggestions if suggestions is not None else []
        self.calls: list[list[dict[str, str]]] = []
        self.json_calls: list[list[dict[str, str]]] = []

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append(messages)
        for word in self.response_text.split(" "):
            yield word + " "

    async def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        self.json_calls.append(messages)
        return {"suggestions": self.suggestions}
