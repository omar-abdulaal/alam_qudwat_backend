from rag.generation.prompt import GROUNDING_RULES, NEVER_IMPERSONATE_RULE, build_chat_messages, narrator_system_prompt
from rag.retrieval.retriever import RetrievedChunk

CHUNK = RetrievedChunk(
    chunk_id="c1",
    text="نص تجريبي عن الصديق.",
    score=0.9,
    character="abu_bakr",
    caliph_name="أبو بكر الصديق",
    book_title="كتاب تجريبي",
    author="مؤلف",
    era="الخلافة الراشدة",
    page_id=1,
    printed_page="1",
    source_url="https://example.com/1",
)


def test_kids_and_adults_prompts_both_carry_grounding_and_impersonation_rules():
    kids = narrator_system_prompt("kids", "أبو بكر الصديق")
    adults = narrator_system_prompt("adults", "أبو بكر الصديق")

    for prompt in (kids, adults):
        assert GROUNDING_RULES in prompt
        assert NEVER_IMPERSONATE_RULE in prompt
        assert "أبو بكر الصديق" in prompt

    assert kids != adults


def test_build_chat_messages_shape_and_ordering():
    history = [{"role": "user", "content": "سؤال سابق"}, {"role": "assistant", "content": "رد سابق"}]
    messages = build_chat_messages(
        "سؤال جديد", [CHUNK], mode="adults", character_name="أبو بكر الصديق", history=history
    )

    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1]["role"] == "user"
    assert "سؤال جديد" in messages[-1]["content"]
    assert "نص تجريبي عن الصديق" in messages[-1]["content"]


def test_build_chat_messages_without_chunks_says_no_sources_available():
    messages = build_chat_messages("سؤال", [], mode="kids", character_name="X", history=None)
    assert "لا توجد مصادر" in messages[-1]["content"]
    assert len(messages) == 2  # system + the single grounded user turn, no history
