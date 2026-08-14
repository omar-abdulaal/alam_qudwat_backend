"""Deterministic, zero-cost chat LLM stub for tests — mirrors
tests/fake_embedder.py's role for the embeddings API."""
from __future__ import annotations

from typing import Any, AsyncIterator


class FakeChatLLM:
    def __init__(
        self,
        response_text: str = "هذا رد تجريبي للاختبار [1].",
        suggestions: list[str] | None = None,
        rewritten_query: str | None = None,
    ):
        self.response_text = response_text
        self.suggestions = suggestions if suggestions is not None else []
        # Used only by chat_service.rewrite_retrieval_query() -- distinct
        # from `suggestions` above so one fake can stand in for either
        # complete_json() caller; each reads only its own key from the
        # dict complete_json() returns.
        self.rewritten_query = rewritten_query
        self.calls: list[list[dict[str, str]]] = []
        self.json_calls: list[list[dict[str, str]]] = []

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append(messages)
        for word in self.response_text.split(" "):
            yield word + " "

    async def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        self.json_calls.append(messages)
        return {"suggestions": self.suggestions, "query": self.rewritten_query}
