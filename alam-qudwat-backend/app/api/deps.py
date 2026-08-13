"""Shared FastAPI dependencies. Provider functions (get_embedder,
get_chat_llm) are separate, overridable callables so tests can swap in
fakes via app.dependency_overrides without touching route code.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import AppSettings, get_app_settings
from app.db.session import get_db
from app.services.live_generation import LiveGenerationRegistry, get_live_generation_registry
from app.services.llm import ChatLLM, OpenAIChatLLM
from app.services.stt import OpenAISTT, SpeechToText
from app.services.tts import SilmaSageMakerTTS, TextToSpeech
from rag.config import Settings, get_settings
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider


def get_embedder() -> EmbeddingProvider:
    return OpenAIEmbeddingProvider()


def get_chat_llm() -> ChatLLM:
    return OpenAIChatLLM()


def get_stt() -> SpeechToText:
    return OpenAISTT()


def get_tts() -> TextToSpeech:
    return SilmaSageMakerTTS()


DbSession = Annotated[Session, Depends(get_db)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
LLMDep = Annotated[ChatLLM, Depends(get_chat_llm)]
STTDep = Annotated[SpeechToText, Depends(get_stt)]
TTSDep = Annotated[TextToSpeech, Depends(get_tts)]
RagSettingsDep = Annotated[Settings, Depends(get_settings)]
AppSettingsDep = Annotated[AppSettings, Depends(get_app_settings)]
LiveGenerationRegistryDep = Annotated[LiveGenerationRegistry, Depends(get_live_generation_registry)]
