"""FastAPI application entrypoint.

Run locally with: uvicorn app.main:app --reload
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, characters, conversations, health, stt, story, tts
from app.core.config import get_app_settings
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.core.security import require_api_token
from app.services.rag_sync import run_background_rag_sync

configure_logging()
logger = logging.getLogger("app")

app_settings = get_app_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if app_settings.auto_ingest_on_startup:
        # A plain OS thread, not an asyncio task: ingest_missing_characters()
        # is fully synchronous (DB + OpenAI HTTP calls), so running it
        # in-loop would block every concurrent request. yield below happens
        # immediately regardless — the API starts serving without waiting
        # for this to finish.
        threading.Thread(target=run_background_rag_sync, name="rag-startup-sync", daemon=True).start()
    else:
        logger.info("AUTO_INGEST_ON_STARTUP is disabled; skipping startup RAG sync")
    yield


app = FastAPI(
    title="Alam Qudwat API",
    description="Grounded, citable storytelling chat API over the Rashidun-caliphs RAG knowledge base.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# /health is intentionally unauthenticated (load-balancer / container health checks).
app.include_router(health.router)

_authed = [Depends(require_api_token)]
app.include_router(characters.router, dependencies=_authed)
app.include_router(story.router, dependencies=_authed)
app.include_router(conversations.router, dependencies=_authed)
app.include_router(chat.router, dependencies=_authed)
app.include_router(stt.router, dependencies=_authed)
app.include_router(tts.router, dependencies=_authed)
