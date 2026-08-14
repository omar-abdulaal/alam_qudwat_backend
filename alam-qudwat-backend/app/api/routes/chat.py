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
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import AppSettingsDep, DbSession, EmbedderDep, LiveGenerationRegistryDep, LLMDep, RagSettingsDep
from app.schemas.chat import ChatRequest
from app.services import chat_service
from app.services.diacritization import strip_diacritics
from rag.generation.prompt import CLOSING_QUESTION, CLOSING_QUESTION_DIACRITIZED

logger = logging.getLogger("app.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # disable nginx/ALB response buffering for SSE
}


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class _GenerationFailed(RuntimeError):
    """Marks a broadcast as failed without leaking internal exception
    details to whatever's subscribed to it (POST /api/v1/tts/speak/live)."""


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    session: DbSession,
    embedder: EmbedderDep,
    llm: LLMDep,
    settings: RagSettingsDep,
    app_settings: AppSettingsDep,
    live_generations: LiveGenerationRegistryDep,
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[bytes]:
        # Registered up front (before retrieval/generation even starts) so
        # the "conversation" event's generation_id is valid the instant
        # it's sent — the Flutter app can fire POST /api/v1/tts/speak/live
        # with it right away and simply wait on the subscription for the
        # first token, rather than needing to know generation is under way.
        generation_id, broadcast = live_generations.create()
        broadcast_error: Optional[BaseException] = None
        try:
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
                broadcast_error = _GenerationFailed("chat generation failed")
                yield _sse("error", {"message": str(exc.detail)})
                return
            except Exception:
                logger.exception("failed to prepare chat turn")
                broadcast_error = _GenerationFailed("chat generation failed")
                yield _sse("error", {"message": "internal error"})
                return

            yield _sse(
                "conversation",
                {"conversation_id": str(turn.conversation_id), "generation_id": str(generation_id)},
            )

            # One extra chance to recover before giving up on this
            # question: if the literal-text retrieval in prepare_turn()
            # failed the grounding gate (e.g. a vague "حدثني عن هذه
            # الشخصية" that never names the character), ask the LLM for a
            # better *search* query — never an answer — using the
            # character and recent conversation context, and retry
            # retrieval once with that instead. Only reached on the
            # failure path, so a question that already retrieves well
            # never pays for this extra call. Configurable via
            # RETRIEVAL_QUERY_REWRITE_ON_FALLBACK (rag/config.py),
            # default enabled.
            if not turn.grounded and settings.retrieval_query_rewrite_on_fallback:
                rewritten_query = await chat_service.rewrite_retrieval_query(
                    llm, turn.question, turn.character_name, turn.history
                )
                if rewritten_query:
                    turn = await asyncio.to_thread(
                        chat_service.retry_retrieval, session, turn, rewritten_query, embedder, settings
                    )

            if not turn.grounded:
                yield _sse("delta", {"text": turn.fallback_text})
                await broadcast.publish(turn.fallback_text)
                await asyncio.to_thread(
                    chat_service.save_assistant_message, session, turn.conversation_id, turn.fallback_text, []
                )
                yield _sse("citations", {"citations": []})
                yield _sse("suggestions", {"suggestions": []})
                yield _sse("done", {})
                return

            # DIACRITIZATION_RULE (rag/generation/prompt.py) has the LLM
            # write this answer already fully diacritized — one generation
            # call doing double duty instead of a second, separate LLM
            # call to diacritize the same answer later just to synthesize
            # speech for it (that used to add a full extra round-trip of
            # latency to every "play audio" request). `raw_parts` keeps
            # that diacritized text (for TTS use only, see
            # app/api/routes/tts.py); every delta sent to the client has
            # diacritics stripped in real time, so the user never sees a
            # tashkeel mark even transiently mid-stream. Stripping is a
            # plain character-class filter with no cross-character
            # dependencies, so stripping token-by-token and concatenating
            # gives byte-for-byte the same result as stripping the joined
            # text once at the end. Each raw (diacritized) token is also
            # published to `broadcast` as it arrives — this is what lets
            # POST /api/v1/tts/speak/live start synthesizing speech for
            # this exact answer while it's still being generated, instead
            # of waiting for `done` and a persisted message_id.
            raw_parts: list[str] = []
            try:
                async for token in llm.stream(turn.llm_messages):
                    raw_parts.append(token)
                    await broadcast.publish(token)
                    yield _sse("delta", {"text": strip_diacritics(token)})
            except Exception:
                logger.exception("LLM streaming failed")
                broadcast_error = _GenerationFailed("chat generation failed")
                yield _sse("error", {"message": "generation failed"})
                return

            # Every answer ends with exactly this question — appended here
            # rather than left to the LLM to reproduce verbatim (the system
            # prompt separately tells it not to write its own version; see
            # rag.generation.prompt.NO_OWN_CLOSING_RULE), so it's guaranteed
            # byte-for-byte rather than merely instructed.
            #
            # full_text's closing is the literal CLOSING_QUESTION constant,
            # appended directly — NOT derived by stripping
            # CLOSING_QUESTION_DIACRITIZED. Those two don't round-trip through
            # strip_diacritics() into each other (CLOSING_QUESTION itself
            # contains a tanween mark — "أيضاً" — that strip_diacritics()
            # would remove like any other diacritic, same as it does for the
            # LLM's own output); deriving one from the other here would
            # silently break "ends with exactly ...".
            full_text = strip_diacritics("".join(raw_parts)) + "\n\n" + CLOSING_QUESTION
            full_diacritized_text = "".join(raw_parts) + "\n\n" + CLOSING_QUESTION_DIACRITIZED
            await broadcast.publish("\n\n" + CLOSING_QUESTION_DIACRITIZED)
            yield _sse("delta", {"text": "\n\n" + CLOSING_QUESTION})

            suggestions = await chat_service.generate_suggestions(llm, turn, full_text)

            extra: dict = {}
            if suggestions:
                extra["suggestions"] = suggestions
            # Kept so a later POST /api/v1/tts/speak?message_id=... can
            # synthesize speech for this exact answer without a second LLM
            # call — see app/api/routes/tts.py.
            extra["diacritized_content"] = full_diacritized_text

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
        finally:
            # Always unblocks anything subscribed via /speak/live, even on
            # an early return above — otherwise a live TTS request would
            # hang forever waiting for text that will never arrive.
            await broadcast.finish(broadcast_error)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)
