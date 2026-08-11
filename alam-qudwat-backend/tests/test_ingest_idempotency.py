import json
from pathlib import Path

from rag.config import Settings
from rag.db.models import Chunk, Document
from rag.ingestion.ingest import run_ingestion
from tests.fake_embedder import FakeEmbeddingProvider

# Fixture data lives in the same (real) database as any already-ingested
# production data (see tests/conftest.py — isolation is transactional, not
# a separate DB, so committed real rows are visible within the test
# transaction). A book_id that can never collide with real scraped data
# (real data uses Shamela book_id 10906) keeps assertions exact regardless
# of what's already ingested.
FIXTURE_BOOK_ID = 999999

FIXTURE_RECORDS = [
    {
        "book_id": FIXTURE_BOOK_ID,
        "book_title": "__TEST_FIXTURE_BOOK__",
        "author": "مؤلف تجريبي",
        "collection": "مجموعة تجريبية",
        "caliph_id": "abu_bakr",
        "caliph_name": "أبو بكر الصديق",
        "page_id": 1,
        "printed_page": "7",
        "url": f"https://read.shamela.ws/book/{FIXTURE_BOOK_ID}/1",
        "text": "[[سير الخلفاء الراشدين]\n]\nاسمه عبد الله، ويقال عتيق بن أبي قحافة رضي الله عنه.",
    },
    {
        "book_id": FIXTURE_BOOK_ID,
        "book_title": "__TEST_FIXTURE_BOOK__",
        "author": "مؤلف تجريبي",
        "collection": "مجموعة تجريبية",
        "caliph_id": "abu_bakr",
        "caliph_name": "أبو بكر الصديق",
        "page_id": 2,
        "printed_page": "8",
        "url": f"https://read.shamela.ws/book/{FIXTURE_BOOK_ID}/2",
        "text": "وعن عائشة قالت: ما أسلم أبو أحد من المهاجرين إلا أبو بكر.",
    },
]


def _write_fixture(tmp_path: Path) -> Path:
    input_dir = tmp_path / "rashidun"
    input_dir.mkdir()
    with (input_dir / "abu_bakr_pages.jsonl").open("w", encoding="utf-8") as fh:
        for record in FIXTURE_RECORDS:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return input_dir


def _settings() -> Settings:
    return Settings(chunk_token_size=200, chunk_token_overlap=20, chunk_min_token_size=5, embedding_batch_size=8)


def test_first_run_creates_documents_and_chunks_and_embeds(tmp_path, db_session):
    input_dir = _write_fixture(tmp_path)
    embedder = FakeEmbeddingProvider()

    stats = run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()

    assert stats.pages_new == 2
    assert stats.pages_unchanged == 0
    assert stats.chunks_new >= 2
    assert embedder.call_count >= 1
    fixture_docs = db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID).all()
    fixture_chunks = db_session.query(Chunk).filter_by(book_title="__TEST_FIXTURE_BOOK__").all()
    assert len(fixture_docs) == 2
    assert len(fixture_chunks) == stats.chunks_new
    for chunk in fixture_chunks:
        assert chunk.embedding is not None


def test_second_run_on_unchanged_data_is_a_no_op_for_embeddings(tmp_path, db_session):
    input_dir = _write_fixture(tmp_path)
    embedder = FakeEmbeddingProvider()

    run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()

    doc_count_after_first = db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID).count()
    chunk_count_after_first = db_session.query(Chunk).filter_by(book_title="__TEST_FIXTURE_BOOK__").count()

    stats = run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()

    assert stats.pages_unchanged == 2
    assert stats.pages_new == 0
    assert stats.chunks_new == 0
    assert stats.chunks_updated == 0
    assert stats.embedded_chunks == 0
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID).count() == doc_count_after_first
    assert db_session.query(Chunk).filter_by(book_title="__TEST_FIXTURE_BOOK__").count() == chunk_count_after_first


def test_changed_page_text_triggers_reembedding_of_only_that_page(tmp_path, db_session):
    input_dir = _write_fixture(tmp_path)
    embedder = FakeEmbeddingProvider()
    run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()
    embedder.call_count = 0

    # Modify only the first record's text.
    modified = [dict(r) for r in FIXTURE_RECORDS]
    modified[0]["text"] = modified[0]["text"] + " نص إضافي تم تصحيحه من المصدر."
    with (input_dir / "abu_bakr_pages.jsonl").open("w", encoding="utf-8") as fh:
        for record in modified:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()

    assert stats.pages_updated == 1
    assert stats.pages_unchanged == 1
    assert embedder.call_count >= 1
