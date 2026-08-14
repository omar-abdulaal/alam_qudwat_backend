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
from app.db.models import Character, Conversation, Message
from app.main import app
from app.services import chat_service
from app.services.suggestions import ADULTS_FIRST_SUGGESTION
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


def test_off_topic_message_gets_fallback_without_calling_the_narrator_llm(client, db_session):
    llm = FakeChatLLM()  # rewritten_query=None -- simulates the rewrite LLM declining to suggest one
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
    assert llm.calls == []  # the narrator LLM must never be called for an ungrounded turn
    # One retrieval-query-rewrite attempt happens (RETRIEVAL_QUERY_REWRITE_ON_FALLBACK
    # defaults to enabled) -- it's a real recovery attempt, not a suggestions call
    # (there's no narrator answer yet to base suggestions on), and it correctly
    # can't rescue a genuinely off-topic question either, so the fallback still applies.
    assert len(llm.json_calls) == 1

    import json

    from rag.generation.prompt import CLOSING_QUESTION

    # The fallback is a plain "sources don't cover this" statement -- no
    # invitation to keep exploring right after telling the user we don't
    # know, and (separately) not something that should get echoed as a
    # pattern into the LLM's own history on a later turn.
    fallback_delta = json.loads(dict(events)["delta"])["text"]
    assert CLOSING_QUESTION not in fallback_delta


def test_query_rewrite_recovers_a_vague_first_message_that_would_otherwise_fall_back(client, monkeypatch):
    """"حدثني عن هذه الشخصية" carries no character-identifying signal on
    its own -- the fake retrieve() below simulates that first attempt
    failing, and the LLM-rewritten query (RETRIEVAL_QUERY_REWRITE_ON_
    FALLBACK, default enabled) recovering it, without the fallback text
    or a fabricated embedding score being involved."""
    from rag.retrieval.retriever import RetrievedChunk

    seen_queries: list[str] = []

    def fake_retrieve(session, query, embedder, *, character, top_k):
        seen_queries.append(query)
        if query == "استعلام محسّن يذكر أبو بكر":
            return [
                RetrievedChunk(
                    chunk_id="c1",
                    text="نص تاريخي عن أبي بكر",
                    score=0.9,
                    character="abu_bakr",
                    caliph_name="أبو بكر الصديق",
                    book_title="سير أعلام النبلاء",
                    author="الذهبي",
                    era="الخلافة الراشدة",
                    page_id=1,
                    printed_page="10",
                    source_url="https://example.com",
                )
            ]
        return []

    monkeypatch.setattr(chat_service, "retrieve", fake_retrieve)

    llm = FakeChatLLM("رد نهائي بعد إعادة صياغة الاستعلام [1].", rewritten_query="استعلام محسّن يذكر أبو بكر")
    _install_fakes(llm)  # default (real) min-score threshold -- irrelevant here since retrieve() itself is mocked

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "حدثني عن هذه الشخصية", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    assert seen_queries == ["حدثني عن هذه الشخصية", "استعلام محسّن يذكر أبو بكر"]
    assert len(llm.calls) == 1  # the narrator LLM was actually called -- proves this isn't the fallback path

    import json

    all_delta_text = "".join(json.loads(d)["text"] for k, d in events if k == "delta")
    assert "رد نهائي بعد إعادة صياغة الاستعلام" in all_delta_text


def test_query_rewrite_fallback_can_be_disabled_via_settings(client, monkeypatch):
    def fake_retrieve(session, query, embedder, *, character, top_k):
        return []  # ungrounded regardless of query -- proves no retry is even attempted

    monkeypatch.setattr(chat_service, "retrieve", fake_retrieve)
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(
        update={"retrieval_query_rewrite_on_fallback": False}
    )

    llm = FakeChatLLM(rewritten_query="لن يُستخدم أبدًا")
    _install_fakes(llm)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "حدثني عن هذه الشخصية", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)
    kinds = [e[0] for e in events]

    assert kinds == ["conversation", "delta", "citations", "suggestions", "done"]
    assert llm.json_calls == []  # the rewrite call must never happen when the flag is off
    assert llm.calls == []  # nor the main narrator call -- still the plain fallback


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


def test_answer_ends_with_the_fixed_closing_question(client, db_session):
    from rag.generation.prompt import CLOSING_QUESTION

    llm = FakeChatLLM("هذا رد تجريبي يعتمد على المصادر [1].")
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    import json

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    assistant_message = db_session.query(Message).filter_by(conversation_id=conv_id, role="assistant").one()
    assert assistant_message.content.endswith(CLOSING_QUESTION)

    # ...and it must have actually streamed as (part of) a delta event too,
    # not just been silently persisted.
    all_delta_text = "".join(json.loads(d)["text"] for k, d in events if k == "delta")
    assert all_delta_text.endswith(CLOSING_QUESTION)


def test_streamed_and_stored_text_never_contains_diacritics(client, db_session):
    """DIACRITIZATION_RULE has the LLM write its answer with tashkeel —
    the fake LLM here simulates that by actually returning diacritized
    text, so this proves stripping genuinely happens rather than passing
    vacuously because the fake never emits diacritics."""
    diacritized_answer = "هَذَا رَدٌّ مُشَكَّلٌ يَعْتَمِدُ عَلَى الْمَصَادِرِ [1]."
    llm = FakeChatLLM(diacritized_answer)
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    import json

    all_delta_text = "".join(json.loads(d)["text"] for k, d in events if k == "delta")
    assert "هَذَا" not in all_delta_text  # diacritized form of the LLM's word must not leak through
    assert "هذا" in all_delta_text  # the plain word is still there, just undiacritized

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    assistant_message = db_session.query(Message).filter_by(conversation_id=conv_id, role="assistant").one()
    assert "هَذَا" not in assistant_message.content
    assert "هذا" in assistant_message.content


def test_diacritized_content_is_stored_for_tts_and_strips_back_to_the_plain_answer(client, db_session):
    """The LLM's raw diacritized portion must round-trip back to the plain
    text via strip_diacritics(); the appended closing question is checked
    separately (test_answer_ends_with_the_fixed_closing_question) since
    CLOSING_QUESTION/CLOSING_QUESTION_DIACRITIZED are independent fixed
    strings by design, not round-trip-derived from one another (see
    app/api/routes/chat.py)."""
    from app.services.diacritization import strip_diacritics

    diacritized_answer = "هَذَا رَدٌّ مُشَكَّلٌ [1]."
    llm = FakeChatLLM(diacritized_answer)
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    import json

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    assistant_message = db_session.query(Message).filter_by(conversation_id=conv_id, role="assistant").one()

    diacritized_content = assistant_message.extra["diacritized_content"]
    assert diacritized_answer in diacritized_content  # the LLM's raw (diacritized) output, untouched
    assert strip_diacritics(diacritized_answer) in assistant_message.content


def test_first_message_in_conversation_gets_the_conciseness_instruction(client, db_session):
    llm = FakeChatLLM("رد أول [1].")
    _install_fakes(llm, lenient=True)

    first = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    import json

    conv_id = json.loads(dict(_parse_sse(first.text))["conversation"])["conversation_id"]
    first_system_prompt = llm.calls[0][0]["content"]
    assert "أول إجابة" in first_system_prompt

    llm.response_text = "رد ثانٍ [1]."
    client.post(
        "/api/v1/chat/stream",
        json={"message": "وماذا حدث بعد ذلك؟", "conversation_id": conv_id},
    )
    second_system_prompt = llm.calls[1][0]["content"]
    assert "أول إجابة" not in second_system_prompt


def test_kids_predefined_suggestions_ignore_character_categories(client, db_session):
    from app.services.suggestions import KIDS_PREDEFINED_SUGGESTIONS

    character = db_session.get(Character, "abu_bakr")
    character.categories = ["قائد عسكري"]  # must not leak an adult-role question into kids mode
    db_session.commit()

    llm = FakeChatLLM("رد [1].", suggestions=[])
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "kids"},
    )
    events = _parse_sse(resp.text)

    import json

    assert json.loads(dict(events)["suggestions"])["suggestions"] == list(KIDS_PREDEFINED_SUGGESTIONS)


def test_suggestions_combine_predefined_and_llm_generated(client, db_session):
    """No role match (categories=[]) leaves 1 predefined + 2 open slots,
    which the LLM's suggestions fill, in order: predefined first."""
    character = db_session.get(Character, "abu_bakr")
    character.categories = []
    db_session.commit()

    llm = FakeChatLLM("رد [1].", suggestions=["سؤال إضافي من الذكاء الاصطناعي؟"])
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    import json

    suggestions_payload = json.loads(dict(events)["suggestions"])
    expected = [ADULTS_FIRST_SUGGESTION, "سؤال إضافي من الذكاء الاصطناعي؟"]
    assert suggestions_payload["suggestions"] == expected

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    assistant_message = (
        db_session.query(Message).filter_by(conversation_id=conv_id, role="assistant").one()
    )
    assert assistant_message.extra["suggestions"] == expected


def test_predefined_suggestions_still_shown_when_llm_declines(client, db_session):
    """Predefined suggestions never depend on the LLM's suggestions call —
    they're shown regardless of whether it offers anything extra."""
    character = db_session.get(Character, "abu_bakr")
    character.categories = ["قائد عسكري"]
    db_session.commit()

    llm = FakeChatLLM("رد [1].", suggestions=[])
    _install_fakes(llm, lenient=True)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    events = _parse_sse(resp.text)

    import json

    expected = [ADULTS_FIRST_SUGGESTION, "كيف تعامل مع تحديات المعارك والقيادة؟"]
    assert json.loads(dict(events)["suggestions"])["suggestions"] == expected

    conv_id = uuid.UUID(json.loads(dict(events)["conversation"])["conversation_id"])
    assistant_message = (
        db_session.query(Message).filter_by(conversation_id=conv_id, role="assistant").one()
    )
    assert assistant_message.extra["suggestions"] == expected


def test_used_predefined_suggestion_never_reappears(client, db_session):
    llm = FakeChatLLM("رد [1].", suggestions=[])
    _install_fakes(llm, lenient=True)

    import json

    first = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    conv_id = json.loads(dict(_parse_sse(first.text))["conversation"])["conversation_id"]

    # The user "selects" the fixed first suggestion by sending its exact text.
    second = client.post(
        "/api/v1/chat/stream",
        json={"message": ADULTS_FIRST_SUGGESTION, "conversation_id": conv_id},
    )
    events = _parse_sse(second.text)
    suggestions_payload = json.loads(dict(events)["suggestions"])["suggestions"]

    assert ADULTS_FIRST_SUGGESTION not in suggestions_payload


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
    conversation_event = json.loads(dict(events)["conversation"])
    assert conversation_event["conversation_id"] == conv_id
    assert "generation_id" in conversation_event

    # The second LLM call's messages must include the first turn's history.
    assert len(llm.calls) == 2
    second_call_messages = llm.calls[1]
    roles = [m["role"] for m in second_call_messages]
    assert roles.count("user") >= 2  # prior user turn (history) + current turn
    assert any("من هو أبو بكر" in m["content"] for m in second_call_messages)


def test_closing_question_is_not_fed_back_to_the_llm_as_history(client, db_session):
    """The stored first answer ends with CLOSING_QUESTION, but the second
    call's history must not contain it -- otherwise the model picks up the
    pattern from its own "prior turn" and starts echoing/repeating it."""
    from rag.generation.prompt import CLOSING_QUESTION

    llm = FakeChatLLM("رد أول [1].")
    _install_fakes(llm, lenient=True)

    first = client.post(
        "/api/v1/chat/stream",
        json={"message": "من هو أبو بكر؟", "character_slug": "abu_bakr", "mode": "adults"},
    )
    import json

    conv_id = json.loads(dict(_parse_sse(first.text))["conversation"])["conversation_id"]

    # Confirm the premise: what's actually stored does end with it.
    assistant_message = (
        db_session.query(Message)
        .filter_by(conversation_id=uuid.UUID(conv_id), role="assistant")
        .one()
    )
    assert assistant_message.content.endswith(CLOSING_QUESTION)

    llm.response_text = "رد ثانٍ [1]."
    client.post(
        "/api/v1/chat/stream",
        json={"message": "وماذا حدث بعد ذلك؟", "conversation_id": conv_id},
    )

    second_call_messages = llm.calls[1]
    assert not any(CLOSING_QUESTION in m["content"] for m in second_call_messages)


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
