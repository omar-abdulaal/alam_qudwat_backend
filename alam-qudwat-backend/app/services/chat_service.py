"""Chat orchestration: load/create conversation -> load history -> RAG
retrieval (reused as-is from rag.retrieval.retriever) -> build grounded
prompt (reused from rag.generation.prompt) -> [caller streams the LLM] ->
persist.

Split into a synchronous "prepare" step (DB + retrieval + prompt-building,
all fast, run via asyncio.to_thread from the async route) and a separate
"save" step for after the LLM stream completes, so a failed/interrupted
generation never leaves a bogus assistant message in history while the
user's own message is still durably saved.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Character, Conversation, Message
from app.schemas.chat import ChatRequest, CitationOut
from app.services.llm import ChatLLM
from rag.config import Settings
from rag.embeddings.base import EmbeddingProvider
from rag.generation.prompt import build_chat_messages, build_suggestions_prompt
from rag.retrieval.retriever import RetrievedChunk, retrieve

MAX_SUGGESTIONS = 3

logger = logging.getLogger("app.chat_service")

FALLBACK_TEXT = {
    "kids": (
        "لا تتوفر لديّ معلومات كافية في المصادر المتاحة للإجابة عن هذا السؤال بدقة. "
        "جرّب أن تسأل عن جانب آخر من القصة!"
    ),
    "adults": (
        "لا تتوفر في المصادر التاريخية المسترجعة معلومات كافية للإجابة عن هذا السؤال بدقة. "
        "يمكنك إعادة صياغة السؤال أو طرح سؤال آخر متعلق بهذه الشخصية."
    ),
}


@dataclass
class ChatTurn:
    conversation_id: uuid.UUID
    mode: str
    grounded: bool
    llm_messages: list[dict[str, str]]  # only meaningful when grounded
    fallback_text: str  # only meaningful when not grounded
    citations: list[CitationOut]
    # Kept for the post-stream suggestions call (generate_suggestions) —
    # not needed once the stream itself is built.
    question: str
    character_name: str
    chunks: list[RetrievedChunk]


def _get_or_create_conversation(session: Session, req: ChatRequest) -> Conversation:
    if req.conversation_id is not None:
        conversation = session.get(Conversation, req.conversation_id)
        if conversation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
        if req.character_slug is not None and req.character_slug != conversation.character_slug:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "character_slug does not match the existing conversation"
            )
        if req.mode is not None and req.mode != conversation.narrator_mode:
            conversation.narrator_mode = req.mode
        return conversation

    if not req.character_slug:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "character_slug is required to start a new conversation")
    if not req.mode:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "mode is required to start a new conversation")

    character = session.get(Character, req.character_slug)
    if character is None or not character.is_active:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown character: {req.character_slug}")

    conversation = Conversation(id=uuid.uuid4(), character_slug=req.character_slug, narrator_mode=req.mode)
    session.add(conversation)
    session.flush()
    return conversation


def prepare_turn(
    session: Session,
    req: ChatRequest,
    embedder: EmbeddingProvider,
    settings: Settings,
    *,
    max_message_length: int,
    history_max_messages: int,
) -> ChatTurn:
    message = req.message.strip()
    if not message:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "message must not be empty")
    if len(message) > max_message_length:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"message too long (max {max_message_length} chars)")

    conversation = _get_or_create_conversation(session, req)
    character = session.get(Character, conversation.character_slug)

    history_rows = (
        session.query(Message)
        .filter_by(conversation_id=conversation.id)
        .order_by(Message.created_at.desc())
        .limit(history_max_messages)
        .all()
    )
    history_rows.reverse()
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    # Persist the user's turn immediately so it survives even if generation
    # fails or the client disconnects mid-stream.
    session.add(Message(id=uuid.uuid4(), conversation_id=conversation.id, role="user", content=message))
    session.commit()

    # Vague follow-ups ("and what happened next?") carry little topical
    # signal on their own and retrieve poorly in isolation, even though the
    # LLM has full history context. Folding in the prior user turn gives
    # the embedding enough context to retrieve the right chunks, without an
    # extra LLM call to rewrite the query.
    last_user_turn = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
    retrieval_query = f"{last_user_turn}\n{message}" if last_user_turn else message

    chunks = retrieve(
        session, retrieval_query, embedder, character=conversation.character_slug, top_k=settings.retrieval_top_k
    )
    grounded = bool(chunks) and chunks[0].score >= settings.retrieval_min_score

    citations = (
        [
            CitationOut(
                index=i,
                book_title=c.book_title,
                author=c.author,
                character=c.character,
                character_name=c.caliph_name,
                page=c.printed_page or str(c.page_id),
                source_url=c.source_url,
                score=c.score,
            )
            for i, c in enumerate(chunks, start=1)
        ]
        if grounded
        else []
    )

    llm_messages = (
        build_chat_messages(
            message,
            chunks,
            mode=conversation.narrator_mode,
            character_name=character.name_ar,
            history=history,
        )
        if grounded
        else []
    )

    return ChatTurn(
        conversation_id=conversation.id,
        mode=conversation.narrator_mode,
        grounded=grounded,
        llm_messages=llm_messages,
        fallback_text=FALLBACK_TEXT[conversation.narrator_mode],
        citations=citations,
        question=message,
        character_name=character.name_ar,
        chunks=chunks if grounded else [],
    )


def save_assistant_message(
    session: Session,
    conversation_id: uuid.UUID,
    content: str,
    citations: list[CitationOut],
    extra: dict | None = None,
) -> None:
    session.add(
        Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            citations=[c.model_dump() for c in citations] or None,
            extra=extra,
        )
    )
    conversation = session.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.updated_at = datetime.now(timezone.utc)
    session.commit()


async def generate_suggestions(llm: ChatLLM, turn: ChatTurn, answer_text: str) -> list[str]:
    """Ask the LLM whether natural follow-up questions exist for this
    answer; the model itself decides (see
    rag.generation.prompt.build_suggestions_prompt) — this can and often
    should return an empty list. Never raises: any failure (API error,
    malformed JSON, wrong shape) just yields no suggestions rather than
    breaking an otherwise-successful answer."""
    messages = build_suggestions_prompt(
        turn.question,
        answer_text,
        turn.chunks,
        mode=turn.mode,
        character_name=turn.character_name,
    )
    try:
        result = await llm.complete_json(messages)
    except Exception:
        logger.exception("suggestions generation failed; continuing without suggestions")
        return []

    raw = result.get("suggestions")
    if not isinstance(raw, list):
        return []

    suggestions = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    return suggestions[:MAX_SUGGESTIONS]
