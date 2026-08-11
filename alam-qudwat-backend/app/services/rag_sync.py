"""Background RAG sync run at backend startup (see app/main.py's lifespan).

Adds any character present under data/raw/* that isn't in the DB yet —
via rag.ingestion.ingest.ingest_missing_characters(), the same ingestion
primitives the manual CLI uses, just filtered to "missing" characters
instead of full content-hash diffing (see that function's docstring).

Deliberately synchronous (DB + OpenAI HTTP calls) and run on its own OS
thread by the caller — never awaited directly from the event loop, so it
can never block request handling. Never raises: any failure here (missing
OPENAI_API_KEY, DB unreachable, etc.) is logged and swallowed so it can
never crash the API process — the app keeps serving regardless.
"""
from __future__ import annotations

import logging

from rag.config import get_settings
from rag.db.session import session_scope
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.ingestion.ingest import ingest_missing_characters

logger = logging.getLogger("app.rag_sync")


def run_background_rag_sync() -> None:
    logger.info("startup RAG sync starting")
    try:
        settings = get_settings()
        embedder = OpenAIEmbeddingProvider()
        with session_scope() as session:
            stats = ingest_missing_characters(session, embedder, settings)
        logger.info(
            "startup RAG sync finished: %d character(s) added (%d checked, %d failed) — %s",
            stats.characters_added,
            stats.characters_checked,
            stats.characters_failed,
            stats.summary(),
        )
    except Exception:
        logger.exception("startup RAG sync failed; API continues serving without it")
