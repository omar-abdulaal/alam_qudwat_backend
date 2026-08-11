import json
from pathlib import Path

from rag.config import Settings
from rag.ingestion.ingest import run_ingestion
from rag.retrieval.retriever import retrieve
from tests.fake_embedder import FakeEmbeddingProvider

# Fixture data lives in the same (real) database as any already-ingested
# production data (see tests/conftest.py — isolation is transactional, not
# a separate DB, so committed real rows are visible within the test
# transaction). A book_id/book_title that can never collide with real
# scraped data keeps fixture assertions exact regardless of what's already
# ingested.
FIXTURE_BOOK_ID = 999999
FIXTURE_BOOK_TITLE = "__TEST_FIXTURE_BOOK__"

RECORDS = [
    {
        "book_id": FIXTURE_BOOK_ID,
        "book_title": FIXTURE_BOOK_TITLE,
        "author": "مؤلف تجريبي",
        "collection": "مجموعة تجريبية",
        "caliph_id": "abu_bakr",
        "caliph_name": "أبو بكر الصديق",
        "page_id": 1,
        "printed_page": "7",
        "url": f"https://read.shamela.ws/book/{FIXTURE_BOOK_ID}/1",
        "text": "اسمه عبد الله، ويقال عتيق بن أبي قحافة رضي الله عنه، وكان أول من آمن من الرجال.",
    },
    {
        "book_id": FIXTURE_BOOK_ID,
        "book_title": FIXTURE_BOOK_TITLE,
        "author": "مؤلف تجريبي",
        "collection": "مجموعة تجريبية",
        "caliph_id": "umar",
        "caliph_name": "عمر بن الخطاب",
        "page_id": 2,
        "printed_page": "70",
        "url": f"https://read.shamela.ws/book/{FIXTURE_BOOK_ID}/2",
        "text": "عمر بن الخطاب بن نفيل العدوي أمير المؤمنين، أسلم بعد نحو خمسين رجلا وإحدى عشرة امرأة.",
    },
]


def _write_fixture(tmp_path: Path) -> Path:
    input_dir = tmp_path / "rashidun"
    input_dir.mkdir()
    with (input_dir / "abu_bakr_pages.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(RECORDS[0], ensure_ascii=False) + "\n")
    with (input_dir / "umar_pages.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(RECORDS[1], ensure_ascii=False) + "\n")
    return input_dir


def _settings() -> Settings:
    return Settings(chunk_token_size=200, chunk_token_overlap=20, chunk_min_token_size=5, embedding_batch_size=8)


def test_retrieve_returns_hits_with_full_citation_metadata(tmp_path, db_session):
    input_dir = _write_fixture(tmp_path)
    embedder = FakeEmbeddingProvider()
    run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()

    results = retrieve(db_session, "من هو أبو بكر", embedder, top_k=5, book_title=FIXTURE_BOOK_TITLE)

    assert len(results) >= 1
    hit = results[0]
    assert hit.book_title == FIXTURE_BOOK_TITLE
    assert hit.author == "مؤلف تجريبي"
    assert hit.source_url.startswith(f"https://read.shamela.ws/book/{FIXTURE_BOOK_ID}/")
    assert hit.character in {"abu_bakr", "umar"}
    assert hit.printed_page in {"7", "70"}


def test_retrieve_filters_by_character(tmp_path, db_session):
    input_dir = _write_fixture(tmp_path)
    embedder = FakeEmbeddingProvider()
    run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()

    results = retrieve(db_session, "من هو", embedder, top_k=10, character="umar", book_title=FIXTURE_BOOK_TITLE)

    assert len(results) >= 1
    assert all(r.character == "umar" for r in results)


def test_retrieve_filters_by_era(tmp_path, db_session):
    input_dir = _write_fixture(tmp_path)
    embedder = FakeEmbeddingProvider()
    run_ingestion(input_dir, db_session, embedder, settings=_settings())
    db_session.commit()

    results = retrieve(
        db_session, "من هو", embedder, top_k=10, era="الخلافة الراشدة", book_title=FIXTURE_BOOK_TITLE
    )

    assert len(results) == 2
