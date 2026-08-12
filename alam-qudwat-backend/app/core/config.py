"""Web-service configuration — pure backend/infra concerns.

Separate from rag.config.Settings (which owns RAG-domain knobs: DB URL,
embedding/chat model, chunking, retrieval). Both read the same .env.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Shared-secret bearer token the Flutter app sends as
    # "Authorization: Bearer <token>". This is the only access control for
    # this MVP — there is no per-user account system yet.
    api_auth_token: str = Field(alias="API_AUTH_TOKEN")

    cors_allow_origins: str = Field(default="*", alias="CORS_ALLOW_ORIGINS")

    # How many prior messages (user+assistant combined) to load from
    # Postgres as conversation context on each new chat turn.
    history_max_messages: int = Field(default=8, alias="HISTORY_MAX_MESSAGES")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    max_message_length: int = Field(default=2000, alias="MAX_MESSAGE_LENGTH")

    # --- Speech-to-text (app/services/stt.py) ---
    stt_max_audio_bytes: int = Field(default=25 * 1024 * 1024, alias="STT_MAX_AUDIO_BYTES")

    # --- Text-to-speech / SILMA on SageMaker (app/services/tts.py) ---
    # silma_sagemaker_endpoint_name is intentionally optional: the app must
    # still boot without it in environments that don't use TTS yet. Only
    # POST /api/v1/tts/speak fails (clearly) if it's unset when called.
    silma_sagemaker_endpoint_name: str | None = Field(default=None, alias="SILMA_SAGEMAKER_ENDPOINT_NAME")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    tts_default_voice_id: str = Field(default="pixel", alias="TTS_DEFAULT_VOICE_ID")
    tts_max_text_length: int = Field(default=1000, alias="TTS_MAX_TEXT_LENGTH")

    # --- Startup RAG sync (app/services/rag_sync.py) ---
    # Runs rag.ingestion.ingest.ingest_missing_characters() in a background
    # thread on startup so any character present in data/raw/* but not yet
    # in the DB gets added automatically, without blocking the API from
    # serving requests. Disable for a scaled-out/read-only API instance,
    # or local dev where an automatic OpenAI bill on every reload isn't
    # wanted.
    auto_ingest_on_startup: bool = Field(default=True, alias="AUTO_INGEST_ON_STARTUP")

    # --- Startup character-classification sync (app/services/character_classification_sync.py) ---
    # Applies data/generated/character_classifications.json (produced
    # locally by scripts/export_character_classifications.py, committed to
    # the repo) to `characters.categories`/`short_description` on every
    # startup. Unlike auto_ingest_on_startup, this makes no external API
    # calls — a server never needs OPENAI_API_KEY or the Batch pipeline for
    # this, just the committed snapshot file.
    auto_sync_classifications_on_startup: bool = Field(
        default=True, alias="AUTO_SYNC_CLASSIFICATIONS_ON_STARTUP"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings(API_AUTH_TOKEN=os.getenv("API_AUTH_TOKEN", "alam-qudwat-dev-secret-123"))
