"""Unit tests for the retrieval-query-rewrite fallback
(app/services/chat_service.py rewrite_retrieval_query / retry_retrieval)
-- the "one more chance" mechanism that asks the LLM for a better search
query when the literal user message fails the grounding gate (e.g. a
vague "حدثني عن هذه الشخصية" that never names the character).

No DB, no real OpenAI calls: chat_service.retrieve is monkeypatched so
behavior is fully controlled by the query text passed in, independent of
any embedding provider's actual semantics -- and retry_retrieval/
rewrite_retrieval_query are exercised directly rather than through the
full HTTP endpoint, since this is really testing the fallback's decision
logic, not the surrounding chat flow.

No pytest-asyncio in this project -- async cases are driven with plain
asyncio.run() (see tests/test_tts_segmenting.py for the same pattern).
"""
from __future__ import annotations

import asyncio
import uuid

from app.services import chat_service
from rag.config import get_settings
from rag.retrieval.retriever import RetrievedChunk
from tests.fake_llm import FakeChatLLM


def _settings(**overrides):
    return get_settings().model_copy(update=overrides)


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        text="نص تجريبي",
        score=score,
        character="abu_bakr",
        caliph_name="أبو بكر الصديق",
        book_title="كتاب",
        author="مؤلف",
        era="الخلافة الراشدة",
        page_id=1,
        printed_page=None,
        source_url="https://example.com",
    )


def _base_turn(**overrides) -> chat_service.ChatTurn:
    defaults = dict(
        conversation_id=uuid.uuid4(),
        mode="adults",
        grounded=False,
        llm_messages=[],
        fallback_text="لا تتوفر معلومات كافية.",
        citations=[],
        question="حدثني عن هذه الشخصية",
        character_name="أبو بكر الصديق",
        chunks=[],
        character_categories=["خليفة"],
        already_asked=set(),
        character_slug="abu_bakr",
        history=[],
    )
    defaults.update(overrides)
    return chat_service.ChatTurn(**defaults)


def test_retry_retrieval_grounds_when_the_rewritten_query_scores_high_enough(monkeypatch):
    seen_queries = []

    def fake_retrieve(session, query, embedder, *, character, top_k):
        seen_queries.append(query)
        if query == "استعلام محسّن يذكر أبو بكر الصديق":
            return [_chunk(0.9)]
        return []

    monkeypatch.setattr(chat_service, "retrieve", fake_retrieve)

    turn = _base_turn()
    updated = chat_service.retry_retrieval(
        session=None,
        turn=turn,
        rewritten_query="استعلام محسّن يذكر أبو بكر الصديق",
        embedder=None,
        settings=_settings(),
    )

    assert seen_queries == ["استعلام محسّن يذكر أبو بكر الصديق"]
    assert updated.grounded is True
    assert len(updated.chunks) == 1
    assert updated.llm_messages  # a grounded prompt was actually built this time
    assert updated.question == turn.question  # the user's literal question is never replaced


def test_retry_retrieval_returns_the_original_turn_unchanged_when_still_ungrounded(monkeypatch):
    monkeypatch.setattr(chat_service, "retrieve", lambda *a, **k: [])

    turn = _base_turn()
    updated = chat_service.retry_retrieval(
        session=None, turn=turn, rewritten_query="ما زال غامضًا", embedder=None, settings=_settings()
    )

    assert updated is turn
    assert updated.grounded is False
    assert updated.llm_messages == []


def test_rewrite_retrieval_query_returns_the_llms_query_not_an_answer():
    llm = FakeChatLLM(rewritten_query="من هو أبو بكر الصديق")

    async def run():
        return await chat_service.rewrite_retrieval_query(llm, "حدثني عن هذه الشخصية", "أبو بكر الصديق", [])

    query = asyncio.run(run())

    assert query == "من هو أبو بكر الصديق"
    assert len(llm.json_calls) == 1
    prompt_text = " ".join(m["content"] for m in llm.json_calls[0])
    assert "أبو بكر الصديق" in prompt_text
    assert "لا تُجب عن السؤال" in prompt_text  # the system prompt's own "never answer" rule


def test_rewrite_retrieval_query_includes_recent_history_context():
    llm = FakeChatLLM(rewritten_query="أي استعلام")
    history = [
        {"role": "user", "content": "من هو الخليفة الأول؟"},
        {"role": "assistant", "content": "هو أبو بكر الصديق."},
    ]

    async def run():
        return await chat_service.rewrite_retrieval_query(llm, "وماذا فعل بعد ذلك؟", "أبو بكر الصديق", history)

    asyncio.run(run())

    prompt_text = " ".join(m["content"] for m in llm.json_calls[0])
    assert "من هو الخليفة الأول؟" in prompt_text


def test_rewrite_retrieval_query_returns_none_on_llm_failure():
    class FailingLLM:
        async def complete_json(self, messages: list[dict[str, str]]) -> dict:
            raise RuntimeError("boom")

    async def run():
        return await chat_service.rewrite_retrieval_query(FailingLLM(), "سؤال", "شخصية", [])

    assert asyncio.run(run()) is None


def test_rewrite_retrieval_query_returns_none_for_a_blank_query():
    llm = FakeChatLLM(rewritten_query="   ")

    async def run():
        return await chat_service.rewrite_retrieval_query(llm, "سؤال", "شخصية", [])

    assert asyncio.run(run()) is None
