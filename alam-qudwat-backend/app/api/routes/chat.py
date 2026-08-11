"""The main chat/storytelling endpoint: retrieval -> grounded generation ->
streaming response with citations.

Flow: Flutter POSTs a message (+ optional conversation_id/character_slug/
mode) -> chat_service.prepare_turn() resolves the conversation, loads
recent history from Postgres, runs RAG retrieval (rag.retrieval.retriever,
unmodified) and builds the grounded prompt (rag.generation.prompt,
unmodified) -> if sources are insufficient, a fixed honest fallback is
streamed instead of calling the LLM -> otherwise the LLM is streamed
token-by-token as Server-Sent Events -> the assistant turn + citations are
persisted only after the stream completes successfully.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import AppSettingsDep, DbSession, EmbedderDep, LLMDep, RagSettingsDep
from app.schemas.chat import ChatRequest
from app.services import chat_service

logger = logging.getLogger("app.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # disable nginx/ALB response buffering for SSE
}


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    session: DbSession,
    embedder: EmbedderDep,
    llm: LLMDep,
    settings: RagSettingsDep,
    app_settings: AppSettingsDep,
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[bytes]:
        try:
            turn = await asyncio.to_thread(
                chat_service.prepare_turn,
                session,
                req,
                embedder,
                settings,
                max_message_length=app_settings.max_message_length,
                history_max_messages=app_settings.history_max_messages,
            )
        except HTTPException as exc:
            yield _sse("error", {"message": str(exc.detail)})
            return
        except Exception:
            logger.exception("failed to prepare chat turn")
            yield _sse("error", {"message": "internal error"})
            return

        yield _sse("conversation", {"conversation_id": str(turn.conversation_id)})

        if not turn.grounded:
            yield _sse("delta", {"text": turn.fallback_text})
            await asyncio.to_thread(
                chat_service.save_assistant_message, session, turn.conversation_id, turn.fallback_text, []
            )
            yield _sse("citations", {"citations": []})
            yield _sse("suggestions", {"suggestions": []})
            yield _sse("done", {})
            return

        text_parts: list[str] = []
        try:
            async for token in llm.stream(turn.llm_messages):
                text_parts.append(token)
                yield _sse("delta", {"text": token})
        except Exception:
            logger.exception("LLM streaming failed")
            yield _sse("error", {"message": "generation failed"})
            return

        full_text = "".join(text_parts)
        suggestions = await chat_service.generate_suggestions(llm, turn, full_text)
        extra = {"suggestions": suggestions} if suggestions else None

        await asyncio.to_thread(
            chat_service.save_assistant_message,
            session,
            turn.conversation_id,
            full_text,
            turn.citations,
            extra,
        )
        yield _sse("citations", {"citations": [c.model_dump() for c in turn.citations]})
        yield _sse("suggestions", {"suggestions": suggestions})
        yield _sse("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)
