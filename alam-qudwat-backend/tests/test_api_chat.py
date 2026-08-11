"""Tests the main chat flow end-to-end at the API layer, with a fake
embedder (tests/fake_embedder.py) and a fake LLM (tests/fake_llm.py) so no
real OpenAI calls/costs are involved. Because the fake embedder returns
content-hash-based vectors unrelated to the real, committed embeddings in
`chunks`, similarity scores against it are noise — so:

- The "not grounded" tests use it as-is (real min-score threshold), which
  reliably falls below threshold and exercises the no-LLM-call fallback.
- The "grounded" tests override rag.config.get_settings with a very low
  retrieval_min_score so the (semantically meaningless but structurally
  valid) top hit still passes the gate, letting us test the actual
  LLM-streaming/citation/persistence path deterministically.
"""
from __future__ import annotations

import uuid

from app.api.deps import get_chat_llm, get_embedder
from app.db.models import Conversation, Message
from app.main import app
from rag.config import get_settings
from tests.fake_embedder import FakeEmbeddingProvider
from tests.fake_llm import FakeChatLLM


def _lenient_settings():
    return get_settings().model_copy(update={"retrieval_min_score": -1.0})


def _install_fakes(llm: FakeChatLLM, *, lenient: bool = False) -> None:
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_chat_llm] = lambda: llm
    if lenient:
        app.dependency_overrides[get_settings] = _lenient_settings


def _parse_sse(body: str) -> list[tuple[str, str]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) == 2 and lines[0].startswith("event: ") and lines[1].startswith("data: "):
            events.append((lines[0][len("event: "):], lines[1][len("data: "):]))
    return events


def test_new_conversation_requires_character_slug_and_mode(client):
    resp = client.post("/api/v1/chat/stream", json={"message": "مرحبا"})
    events = _parse_sse(resp.text)
    assert events[0][0] == "error"
    assert "character_slug" in events[0][1]


def test_unknown_character_returns_error_event(client):
    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "مرحبا", "character_slug": "not-a-real-character", "mode": "adults"},
    )
    events = _parse_sse(resp.text)
    assert events[0][0] == "error"


def test_off_topic_message_gets_fallback_without_calling_llm(client, db_session):
    llm = FakeChatLLM()
    _install_fakes(llm)  # default (real) min-score threshold, noise-scored fake embeddings

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "ما هي أفضل وصفة للبيتزا؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)
    kinds = [e[0] for e in events]

    assert kinds == ["conversation", "delta", "citations", "suggestions", "done"]
    assert '"citations": []' in dict(events)["citations"]
    assert '"suggestions": []' in dict(events)["suggestions"]
    assert llm.calls == []  # the LLM must never be called for an ungrounded turn
    assert llm.json_calls == []  # nor the suggestions call


def test_grounded_message_streams_llm_response_with_citations_and_persists(client, db_session):
    llm = FakeChatLLM("هذا رد تجريبي يعتمد على المصادر [1].")
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)
    kinds = [e[0] for e in events]

    assert kinds[0] == "conversation"
    assert "delta" in kinds
    assert kinds[-3:] == ["citations", "suggestions", "done"]
    assert len(llm.calls) == 1
    assert len(llm.json_calls) == 1  # the suggestions call happened

    import json

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    conversation = db_session.get(Conversation, conv_id)
    assert conversation is not None
    assert conversation.character_slug == "abu_bakr"
    assert conversation.narrator_mode == "adults"

    messages = db_session.query(Message).filter_by(conversation_id=conv_id).order_by(Message.created_at).all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].citations  # structured citations were persisted


def test_suggestions_are_streamed_and_persisted_when_llm_provides_them(client, db_session):
    llm = FakeChatLLM("رد [1].", suggestions=["سؤال متابعة أول؟", "سؤال متابعة ثانٍ؟"])
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    import json

    suggestions_payload = json.loads(dict(events)["suggestions"])
    assert suggestions_payload["suggestions"] == ["سؤال متابعة أول؟", "سؤال متابعة ثانٍ؟"]

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    assistant_message = (
        db_session.query(Message).filter_by(conversation_id=conv_id, role="assistant").one()
    )
    assert assistant_message.extra == {"suggestions": ["سؤال متابعة أول؟", "سؤال متابعة ثانٍ؟"]}


def test_no_suggestions_when_llm_decides_against_it(client, db_session):
    llm = FakeChatLLM("رد [1].", suggestions=[])
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    import json

    assert json.loads(dict(events)["suggestions"])["suggestions"] == []

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    assistant_message = (
        db_session.query(Message).filter_by(conversation_id=conv_id, role="assistant").one()
    )
    assert assistant_message.extra is None


def test_followup_reuses_conversation_and_loads_history(client, db_session):
    llm = FakeChatLLM("رد أول [1].")
    _install_fakes(llm, lenient=True)

    first = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    import json

    conv_id = json.loads(dict(_parse_sse(first.text))["conversation"])["conversation_id"]

    llm.response_text = "رد ثانٍ [1]."
    second = client.post(
        "/api/v1/chat/stream",
        json={"message": "وماذا حدث بعد ذلك؟", "conversation_id": conv_id},
    )
    events = _parse_sse(second.text)
    assert dict(events)["conversation"] == json.dumps({"conversation_id": conv_id})

    # The second LLM call's messages must include the first turn's history.
    assert len(llm.calls) == 2
    second_call_messages = llm.calls[1]
    roles = [m["role"] for m in second_call_messages]
    assert roles.count("user") >= 2  # prior user turn (history) + current turn
    assert any("من هو أبو بكر" in m["content"] for m in second_call_messages)


def test_mismatched_character_slug_for_existing_conversation_is_rejected(client):
    llm = FakeChatLLM()
    _install_fakes(llm, lenient=True)

    first = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو عمر؟", "character_slug": "umar", "mode": "adults"},
    )
    import json

    conv_id = json.loads(dict(_parse_sse(first.text))["conversation"])["conversation_id"]

    second = client.post(
        "/api/v1/chat/stream",
        json={"message": "سؤال آخر", "conversation_id": conv_id, "character_slug": "abu_bakr"},
    )
    events = _parse_sse(second.text)
    assert events[0][0] == "error"
