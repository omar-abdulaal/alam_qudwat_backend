from rag.ingestion.hashing import content_hash


def test_same_text_same_hash():
    assert content_hash("نص عربي") == content_hash("نص عربي")


def test_different_text_different_hash():
    assert content_hash("نص عربي") != content_hash("نص عربي آخر")


def test_hash_is_hex_sha256_length():
    h = content_hash("أي نص")
    assert len(h) == 64
    int(h, 16)  # raises if not valid hex
