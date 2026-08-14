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
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Character, Conversation, Message
from app.schemas.chat import ChatRequest, CitationOut
from app.services.llm import ChatLLM
from app.services.suggestions import MAX_SUGGESTIONS, filter_unused, predefined_suggestions
from rag.config import Settings
from rag.embeddings.base import EmbeddingProvider
from rag.generation.prompt import (
    CLOSING_QUESTION,
    build_chat_messages,
    build_retrieval_query_rewrite_prompt,
    build_suggestions_prompt,
)
from rag.retrieval.retriever import RetrievedChunk, retrieve

logger = logging.getLogger("app.chat_service")

# The fixed "not enough information" answers deliberately do NOT carry
# CLOSING_QUESTION — that invitation to keep exploring doesn't make sense
# right after telling the user the sources don't cover their question,
# and (see _strip_trailing_closing_question below) baking it in here
# previously polluted the LLM's own history with a pattern it would
# sometimes echo back into later answers.
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

_CLOSING_SUFFIX = "\n\n" + CLOSING_QUESTION


def _strip_trailing_closing_question(content: str) -> str:
    """Assistant messages are stored with CLOSING_QUESTION appended
    (app/api/routes/chat.py). Feeding that back to the LLM verbatim as its
    own conversation history taught it a pattern it would sometimes echo
    mid-answer (a visible duplicate closing question) or reuse in a later
    turn even when no longer appropriate — so history never includes it,
    even though storage/display still does."""
    if content.endswith(_CLOSING_SUFFIX):
        return content[: -len(_CLOSING_SUFFIX)]
    return content


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
    character_categories: list[str]
    # Every message the user has ever sent in this conversation (including
    # the current one) — a suggestion matching any of these must never be
    # offered, predefined or LLM-generated (see app/services/suggestions.py).
    already_asked: set[str]
    # The two fields below exist only to support the retrieval-query-rewrite
    # fallback (rewrite_retrieval_query/retry_retrieval below), called from
    # app/api/routes/chat.py when `grounded` comes back False — not needed
    # by the main streaming/persistence flow otherwise.
    character_slug: str
    history: list[dict[str, str]]


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


def _grade_and_build(
    session: Session,
    query_text: str,
    message: str,
    embedder: EmbeddingProvider,
    settings: Settings,
    *,
    character_slug: str,
    character_name: str,
    mode: str,
    history: list[dict[str, str]],
    is_first_message: bool,
) -> tuple[bool, list[RetrievedChunk], list[CitationOut], list[dict[str, str]]]:
    """Runs one retrieve() call plus the grounding-gate decision and (only
    if grounded) prompt building. `query_text` is what gets embedded for
    retrieval; `message` is always the user's actual, literal question —
    the one shown to the narrator LLM as "the question" and to the user —
    the two differ only when called from retry_retrieval() with an
    LLM-rewritten query_text, never elsewhere. Shared by prepare_turn's
    initial attempt and that retry so the grounding/prompt-building logic
    itself only lives in one place."""
    chunks = retrieve(session, query_text, embedder, character=character_slug, top_k=settings.retrieval_top_k)
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
            mode=mode,
            character_name=character_name,
            history=history,
            is_first_message=is_first_message,
        )
        if grounded
        else []
    )

    return grounded, chunks, citations, llm_messages


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
    history = [
        {
            "role": m.role,
            "content": _strip_trailing_closing_question(m.content) if m.role == "assistant" else m.content,
        }
        for m in history_rows
    ]

    # True only for a conversation's very first assistant answer. Safe to
    # derive from the (capped) history window: if the conversation were
    # long enough to have already fallen out of that cap, it would
    # necessarily already contain an assistant turn within it too.
    is_first_message = not any(h["role"] == "assistant" for h in history)

    # Persist the user's turn immediately so it survives even if generation
    # fails or the client disconnects mid-stream.
    session.add(Message(id=uuid.uuid4(), conversation_id=conversation.id, role="user", content=message))
    session.commit()

    # Every user message in this conversation so far (uncapped, unlike
    # `history` above) — a suggestion must never repeat one of these, no
    # matter how far back it was asked (see app/services/suggestions.py).
    already_asked = {
        content
        for (content,) in session.query(Message.content)
        .filter_by(conversation_id=conversation.id, role="user")
        .all()
    }

    # Vague follow-ups ("and what happened next?") carry little topical
    # signal on their own and retrieve poorly in isolation, even though the
    # LLM has full history context. Folding in the prior user turn gives
    # the embedding enough context to retrieve the right chunks, without an
    # extra LLM call to rewrite the query.
    last_user_turn = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
    retrieval_query = f"{last_user_turn}\n{message}" if last_user_turn else message

    grounded, chunks, citations, llm_messages = _grade_and_build(
        session,
        retrieval_query,
        message,
        embedder,
        settings,
        character_slug=conversation.character_slug,
        character_name=character.name_ar,
        mode=conversation.narrator_mode,
        history=history,
        is_first_message=is_first_message,
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
        character_categories=character.categories,
        already_asked=already_asked,
        character_slug=conversation.character_slug,
        history=history,
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


async def rewrite_retrieval_query(
    llm: ChatLLM, question: str, character_name: str, history: list[dict[str, str]]
) -> str | None:
    """Only called from app/api/routes/chat.py when the initial retrieve()
    attempt in prepare_turn() failed the grounding gate (e.g. a vague
    first message like "حدثني عن هذه الشخصية" that never mentions the
    character by name, so its embedding carries almost no topical
    signal) — asks the LLM for a better *search* query using the
    character and recent conversation context, never an answer (see the
    prompt's own rule 1 in rag/generation/prompt.py). Gated by
    Settings.retrieval_query_rewrite_on_fallback (rag/config.py,
    RETRIEVAL_QUERY_REWRITE_ON_FALLBACK env var, default enabled) at the
    call site — a question that already retrieves well never reaches
    this function, so it adds no latency/cost to the common path.

    Never raises: any failure here just means retry_retrieval() below is
    skipped and the existing "sources don't cover this" fallback proceeds
    exactly as it did before this feature existed."""
    messages = build_retrieval_query_rewrite_prompt(question, character_name, history)
    try:
        result = await llm.complete_json(messages)
    except Exception:
        logger.exception("retrieval query rewrite failed; keeping the original ungrounded result")
        return None
    query = result.get("query")
    return query.strip() if isinstance(query, str) and query.strip() else None


def retry_retrieval(
    session: Session,
    turn: ChatTurn,
    rewritten_query: str,
    embedder: EmbeddingProvider,
    settings: Settings,
) -> ChatTurn:
    """Re-runs retrieval with an LLM-rewritten query (rewrite_retrieval_query
    above) after the original attempt failed the grounding gate. Returns
    `turn` unchanged (still ungrounded) if the retry doesn't clear the gate
    either — callers only ever need to branch on turn.grounded, never on
    whether a retry happened."""
    is_first_message = not any(h["role"] == "assistant" for h in turn.history)
    grounded, chunks, citations, llm_messages = _grade_and_build(
        session,
        rewritten_query,
        turn.question,
        embedder,
        settings,
        character_slug=turn.character_slug,
        character_name=turn.character_name,
        mode=turn.mode,
        history=turn.history,
        is_first_message=is_first_message,
    )
    if not grounded:
        return turn
    return replace(turn, grounded=True, chunks=chunks, citations=citations, llm_messages=llm_messages)


async def generate_suggestions(llm: ChatLLM, turn: ChatTurn, answer_text: str) -> list[str]:
    """Up to MAX_SUGGESTIONS (3) suggestions: predefined ones first (no
    LLM call — app.services.suggestions.predefined_suggestions, by
    NarratorMode and, for adults, the character's role), then — only if
    slots remain — LLM-generated ones filling the rest, which the model
    itself decides whether to offer at all (see
    rag.generation.prompt.build_suggestions_prompt) and can return fewer
    than requested, or none.

    A suggestion the user has already sent as a message in this
    conversation (turn.already_asked) is excluded — predefined ones by
    exact match here, LLM ones by both exact match (post-filtered below)
    and instruction (the prompt is told the same list, to avoid
    paraphrase-level repeats too). Never raises: any LLM failure just
    falls back to the predefined suggestions rather than breaking an
    otherwise-successful answer."""
    predefined = filter_unused(
        predefined_suggestions(turn.mode, turn.character_categories), turn.already_asked
    )
    remaining_slots = MAX_SUGGESTIONS - len(predefined)
    if remaining_slots <= 0:
        return predefined[:MAX_SUGGESTIONS]

    messages = build_suggestions_prompt(
        turn.question,
        answer_text,
        turn.chunks,
        mode=turn.mode,
        character_name=turn.character_name,
        already_asked=sorted(turn.already_asked),
        max_suggestions=remaining_slots,
    )
    try:
        result = await llm.complete_json(messages)
    except Exception:
        logger.exception("suggestions generation failed; continuing with predefined suggestions only")
        return predefined

    raw = result.get("suggestions")
    llm_suggestions = [s.strip() for s in raw if isinstance(s, str) and s.strip()] if isinstance(raw, list) else []
    llm_suggestions = filter_unused(llm_suggestions, turn.already_asked | set(predefined))

    return (predefined + llm_suggestions)[:MAX_SUGGESTIONS]
