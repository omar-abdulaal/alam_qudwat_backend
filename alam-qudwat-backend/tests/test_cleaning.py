from rag.ingestion.cleaning import clean_text


def test_strips_bracket_heading_artifacts_without_touching_words():
    raw = (
        "[[سير الخلفاء الراشدين]\n]\n[-\n"
        "أبو بكر الصديق خليفة رسول الله صلى الله عليه وسلم:\n]\n"
        "اسمه عبد الله -\nويقال:\nعتيق."
    )
    cleaned = clean_text(raw)

    assert "[" not in cleaned
    assert "]" not in cleaned
    assert "سير الخلفاء الراشدين" in cleaned
    assert "أبو بكر الصديق خليفة رسول الله صلى الله عليه وسلم:" in cleaned
    assert "اسمه عبد الله -" in cleaned
    assert "عتيق." in cleaned


def test_collapses_excess_blank_lines():
    raw = "سطر واحد\n\n\n\nسطر آخر"
    cleaned = clean_text(raw)
    assert "\n\n\n" not in cleaned


def test_plain_text_passes_through_unchanged_content():
    raw = "وعن عائشة،\nقالت:\nما أسلم أبو أحد من المهاجرين إلا أبو بكر."
    cleaned = clean_text(raw)
    for word in ["وعن", "عائشة", "قالت", "أسلم", "المهاجرين", "بكر"]:
        assert word in cleaned
