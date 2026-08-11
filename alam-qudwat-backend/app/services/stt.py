"""Speech-to-text: transcribe an audio upload to plain text.

Deliberately does nothing beyond transcription — no chat turn is created,
no LLM answer is generated. The client is expected to take the returned
text and send it through POST /api/v1/chat/stream itself, like any typed
message.
"""
from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI

from rag.config import get_settings


class SpeechToText(Protocol):
    async def transcribe(self, *, audio_bytes: bytes, filename: str) -> str:
        """Return the transcribed text for the given audio file."""
        ...


class OpenAISTT:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.stt_model
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file — see .env.example."
            )
        self._client = AsyncOpenAI(api_key=key)

    async def transcribe(self, *, audio_bytes: bytes, filename: str) -> str:
        response = await self._client.audio.transcriptions.create(
            model=self.model,
            file=(filename, audio_bytes),
        )
        return response.text
