from rag.ingestion.chunker import chunk_text

SENTENCE = "وقال ابن معين إن أبا بكر الصديق رضي الله عنه كان أول من آمن من الرجال."


def _make_long_text(n_sentences: int) -> str:
    return " ".join(SENTENCE for _ in range(n_sentences))


def test_short_text_is_a_single_chunk():
    chunks = chunk_text(SENTENCE, max_tokens=400, overlap_tokens=60, min_tokens=40)
    assert len(chunks) == 1
    assert chunks[0].text.strip().startswith("وقال ابن معين")


def test_long_text_splits_into_multiple_chunks_within_token_budget():
    text = _make_long_text(200)
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20, min_tokens=20)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 100 + 5  # small slack for the merge-back-when-tiny path
        assert c.text.strip() != ""


def test_chunk_indices_are_sequential():
    text = _make_long_text(200)
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20, min_tokens=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    text = _make_long_text(200)
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=30, min_tokens=20)

    assert len(chunks) > 1
    # The tail of chunk N should share text with the head of chunk N+1.
    first_words_of_next = chunks[1].text.split()[:3]
    assert any(w in chunks[0].text for w in first_words_of_next)


def test_no_words_are_altered_only_split():
    text = _make_long_text(50)
    chunks = chunk_text(text, max_tokens=80, overlap_tokens=10, min_tokens=20)
    reassembled_words = set(" ".join(c.text for c in chunks).split())
    original_words = set(text.split())
    # Every word appearing in the source must still appear verbatim.
    assert original_words.issubset(reassembled_words)


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n  ", max_tokens=400, overlap_tokens=60, min_tokens=40) == []
