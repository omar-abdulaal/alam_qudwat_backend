from rag.generation.prompt import build_suggestions_prompt
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


def test_build_suggestions_prompt_shape():
    messages = build_suggestions_prompt(
        "سؤال المستخدم",
        "نص الإجابة السردية",
        [CHUNK],
        mode="adults",
        character_name="أبو بكر الصديق",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "سؤال المستخدم" in messages[1]["content"]
    assert "نص الإجابة السردية" in messages[1]["content"]
    assert "نص تجريبي عن الصديق" in messages[1]["content"]


def test_build_suggestions_prompt_instructs_llm_that_empty_is_valid():
    messages = build_suggestions_prompt(
        "سؤال", "إجابة", [], mode="kids", character_name="X"
    )
    assert "قائمة فارغة" in messages[0]["content"]
