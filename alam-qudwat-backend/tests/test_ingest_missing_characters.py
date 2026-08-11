import json
import uuid
from pathlib import Path

from rag.config import Settings
from rag.db.models import Chunk, Document
from rag.ingestion.hashing import content_hash
from rag.ingestion.ingest import ingest_missing_characters
from tests.fake_embedder import FakeEmbeddingProvider

# Same convention as tests/test_ingest_idempotency.py: a book_id that can
# never collide with real Shamela data, so assertions stay exact even
# though real committed rows are visible (read-only) in the test
# transaction.
FIXTURE_BOOK_ID = 999999


def _settings() -> Settings:
    return Settings(chunk_token_size=200, chunk_token_overlap=20, chunk_min_token_size=5, embedding_batch_size=8)


def _write_jsonl_dataset(dataset_dir: Path, character_id: str, character_name: str, pages: dict[int, str]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / f"{character_id}_pages.jsonl").open("w", encoding="utf-8") as fh:
        for page_id, text in pages.items():
            record = {
                "book_id": FIXTURE_BOOK_ID,
                "book_title": "__TEST_FIXTURE_BOOK__",
                "author": "مؤلف تجريبي",
                "collection": "مجموعة تجريبية",
                "caliph_id": character_id,
                "caliph_name": character_name,
                "page_id": page_id,
                "printed_page": str(page_id),
                "url": f"https://read.shamela.ws/book/{FIXTURE_BOOK_ID}/{page_id}",
                "text": text,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _preseed_document(session, character_id: str, character_name: str, page_id: int) -> None:
    """Directly insert a Document row, bypassing ingestion — simulates a
    character that was already ingested in a prior run."""
    raw_text = f"نص موجود مسبقًا لصفحة {page_id}."
    session.add(
        Document(
            id=uuid.uuid4(),
            book_id=FIXTURE_BOOK_ID,
            book_title="__TEST_FIXTURE_BOOK__",
            author="مؤلف تجريبي",
            collection="مجموعة تجريبية",
            caliph_id=character_id,
            caliph_name=character_name,
            era="غير محدد",
            page_id=page_id,
            printed_page=str(page_id),
            source_url=f"https://read.shamela.ws/book/{FIXTURE_BOOK_ID}/{page_id}",
            raw_text=raw_text,
            content_hash=content_hash(raw_text),
        )
    )
    session.flush()


def test_missing_character_gets_ingested(tmp_path, db_session):
    data_dir = tmp_path / "data_root"
    _write_jsonl_dataset(data_dir / "testset", "char_a", "الشخصية أ", {1: "نص الشخصية الأولى."})
    embedder = FakeEmbeddingProvider()

    stats = ingest_missing_characters(db_session, embedder, settings=_settings(), data_dir=data_dir)
    db_session.commit()

    assert stats.pages_new == 1
    assert embedder.call_count >= 1
    assert stats.characters_checked == 1
    assert stats.characters_added == 1
    assert stats.characters_failed == 0
    docs = db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_a").all()
    assert len(docs) == 1
    chunks = db_session.query(Chunk).filter_by(character="char_a").all()
    assert len(chunks) >= 1
    assert all(c.embedding is not None for c in chunks)


def test_already_present_character_is_skipped_with_zero_embedding_calls(tmp_path, db_session):
    _preseed_document(db_session, "char_b", "الشخصية ب", page_id=99)
    db_session.commit()

    data_dir = tmp_path / "data_root"
    # A *different* page for the same character — if the pre-seeding
    # weren't respected, this would look like new content to ingest.
    _write_jsonl_dataset(data_dir / "testset", "char_b", "الشخصية ب", {1: "نص جديد لم يُعالج مطلقًا."})
    embedder = FakeEmbeddingProvider()

    stats = ingest_missing_characters(db_session, embedder, settings=_settings(), data_dir=data_dir)
    db_session.commit()

    assert stats.pages_seen == 0  # never even read/hash-compared
    assert embedder.call_count == 0
    assert stats.characters_checked == 0  # skipped before ever being "checked"
    assert stats.characters_added == 0
    docs = db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_b").all()
    assert len(docs) == 1  # still just the pre-seeded one — nothing duplicated
    assert docs[0].page_id == 99


def test_rerun_is_idempotent_no_duplicates(tmp_path, db_session):
    data_dir = tmp_path / "data_root"
    _write_jsonl_dataset(data_dir / "testset", "char_c", "الشخصية ج", {1: "نص الشخصية الثالثة."})
    embedder = FakeEmbeddingProvider()

    ingest_missing_characters(db_session, embedder, settings=_settings(), data_dir=data_dir)
    db_session.commit()
    doc_count = db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_c").count()
    chunk_count = db_session.query(Chunk).filter_by(character="char_c").count()

    stats = ingest_missing_characters(db_session, embedder, settings=_settings(), data_dir=data_dir)
    db_session.commit()

    assert stats.pages_seen == 0
    assert stats.pages_new == 0
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_c").count() == doc_count
    assert db_session.query(Chunk).filter_by(character="char_c").count() == chunk_count


def test_one_failing_character_does_not_abort_the_others(tmp_path, db_session):
    class _FlakyEmbedder(FakeEmbeddingProvider):
        def embed(self, texts):
            if any("BOOM" in t for t in texts):
                raise RuntimeError("simulated transient embedding failure")
            return super().embed(texts)

    data_dir = tmp_path / "data_root"
    # Distinct page_ids across characters — book_id+page_id is globally
    # unique regardless of which character a page nominally belongs to
    # (matches the real DB constraint); reusing page_id=1 for all three
    # would make each "character" overwrite the same row.
    _write_jsonl_dataset(data_dir / "testset", "char_ok1", "شخصية سليمة ١", {1: "نص سليم أول."})
    _write_jsonl_dataset(data_dir / "testset2", "char_bad", "شخصية معطوبة", {2: "نص فيه كلمة BOOM لإفشال التضمين."})
    _write_jsonl_dataset(data_dir / "testset3", "char_ok2", "شخصية سليمة ٢", {3: "نص سليم ثانٍ."})
    embedder = _FlakyEmbedder()

    stats = ingest_missing_characters(db_session, embedder, settings=_settings(), data_dir=data_dir)
    db_session.commit()

    assert stats.characters_checked == 3
    assert stats.characters_added == 2
    assert stats.characters_failed == 1
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_ok1").count() == 1
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_ok2").count() == 1
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_bad").count() == 0


def test_discovers_multiple_dataset_directories(tmp_path, db_session):
    data_dir = tmp_path / "data_root"
    _write_jsonl_dataset(data_dir / "setone", "char_d", "الشخصية د", {1: "نص أول."})
    _write_jsonl_dataset(data_dir / "settwo", "char_e", "الشخصية هـ", {2: "نص ثانٍ."})
    embedder = FakeEmbeddingProvider()

    stats = ingest_missing_characters(db_session, embedder, settings=_settings(), data_dir=data_dir)
    db_session.commit()

    assert stats.pages_new == 2
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_d").count() == 1
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_e").count() == 1


def test_unrecognized_subdirectory_is_skipped_not_fatal(tmp_path, db_session):
    data_dir = tmp_path / "data_root"
    (data_dir / "not_a_dataset").mkdir(parents=True)
    (data_dir / "not_a_dataset" / "readme.txt").write_text("hello", encoding="utf-8")
    _write_jsonl_dataset(data_dir / "realset", "char_f", "الشخصية و", {1: "نص."})
    embedder = FakeEmbeddingProvider()

    stats = ingest_missing_characters(db_session, embedder, settings=_settings(), data_dir=data_dir)
    db_session.commit()

    assert stats.pages_new == 1
    assert db_session.query(Document).filter_by(book_id=FIXTURE_BOOK_ID, caliph_id="char_f").count() == 1


def test_missing_data_dir_is_a_noop(tmp_path, db_session):
    embedder = FakeEmbeddingProvider()
    stats = ingest_missing_characters(
        db_session, embedder, settings=_settings(), data_dir=tmp_path / "does_not_exist"
    )
    assert stats.pages_seen == 0
    assert embedder.call_count == 0
