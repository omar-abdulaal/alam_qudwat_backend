"""The actual grounded-chat LLM call.

rag/generation/prompt.py deliberately builds prompts without calling any
LLM — this module is where that call happens, kept thin (provider could
be swapped later) and async so token streaming doesn't block the event
loop.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Protocol

from openai import AsyncOpenAI

from rag.config import get_settings


class ChatLLM(Protocol):
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield response text incrementally."""
        ...

    async def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Non-streamed call constrained to return a JSON object. Used for
        small structured decisions (e.g. follow-up suggestions) where
        streaming would just add parsing complexity for no benefit."""
        ...


class OpenAIChatLLM:
    def __init__(self, model: str | None = None, temperature: float | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.chat_model
        self.temperature = temperature if temperature is not None else settings.chat_temperature
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file — see .env.example."
            )
        self._client = AsyncOpenAI(api_key=key)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=False,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
