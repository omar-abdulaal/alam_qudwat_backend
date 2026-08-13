"""Central configuration for the RAG pipeline.

All environment-specific / secret values are read from the environment
(optionally via a local ``.env`` file, never committed). Nothing here
should hardcode credentials or deployment-specific values.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Rashidun caliphs currently produced by scraper/shamela_scraper.py.
# New characters/books/eras can be added here (or, at larger scale, moved
# into a small lookup table in the DB) without touching ingestion code.
CHARACTER_ERA_MAP: dict[str, str] = {
    "abu_bakr": "الخلافة الراشدة",
    "umar": "الخلافة الراشدة",
    "uthman": "الخلافة الراشدة",
    "ali": "الخلافة الراشدة",
}

# Fallback for datasets with no meaningful per-character era (e.g. hundreds
# of Companions spanning many periods, where periodizing each one
# individually would be inventing history rather than reading it). Keyed by
# the page's `collection` value and set to that same source-provided label
# — never a fabricated period.
COLLECTION_ERA_MAP: dict[str, str] = {
    "الصحابة": "الصحابة",
}

DEFAULT_ERA = "غير محدد"  # "unspecified" — fallback for future/unknown characters


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(alias="DATABASE_URL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")

    chunk_token_size: int = Field(default=400, alias="CHUNK_TOKEN_SIZE")
    chunk_token_overlap: int = Field(default=60, alias="CHUNK_TOKEN_OVERLAP")
    chunk_min_token_size: int = Field(default=40, alias="CHUNK_MIN_TOKEN_SIZE")

    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    # Grounded chat generation (used by app/services/llm.py + chat_service.py).
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    chat_temperature: float = Field(default=0.3, alias="CHAT_TEMPERATURE")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_min_score: float = Field(default=0.28, alias="RETRIEVAL_MIN_SCORE")

    # Speech-to-text (app/services/stt.py). Same "which OpenAI model" domain
    # as chat_model/embedding_model above.
    stt_model: str = Field(default="gpt-4o-mini-transcribe", alias="STT_MODEL")

    # Arabic diacritization for TTS pronunciation only (app/services/
    # diacritization.py) — same "which OpenAI model" domain as above.
    diacritization_model: str = Field(default="gpt-4o-mini", alias="DIACRITIZATION_MODEL")

    # Root directory containing dataset subdirectories (e.g. data/raw/rashidun,
    # data/raw/companions_tier1) — used by ingest_missing_characters() to
    # discover what's available to ingest. Not read by the manual `ingest`
    # CLI, which takes an explicit --input directory instead.
    data_dir: Path = Field(default=Path("data/raw"), alias="RAG_DATA_DIR")

    def era_for_page(self, character_id: str, collection: str) -> str:
        """Exact character match first (Rashidun); else fall back to a
        collection-level label (e.g. Companions -> "الصحابة"); else
        DEFAULT_ERA. Never fabricates a period the source doesn't assert."""
        if character_id in CHARACTER_ERA_MAP:
            return CHARACTER_ERA_MAP[character_id]
        return COLLECTION_ERA_MAP.get(collection, DEFAULT_ERA)


@lru_cache
def get_settings() -> Settings:
    return Settings()
