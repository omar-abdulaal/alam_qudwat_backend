import json

import pytest

from rag.ingestion.loader import load_source_pages

JSONL_RECORD = {
    "book_id": 10906,
    "book_title": "كتاب تجريبي",
    "author": "مؤلف",
    "collection": "مجموعة",
    "caliph_id": "abu_bakr",
    "caliph_name": "أبو بكر الصديق",
    "page_id": 1,
    "printed_page": "1",
    "url": "https://example.com/1",
    "text": "نص تجريبي.",
}


def _companions_record(**overrides) -> dict:
    record = {
        "book_id": 10906,
        "book_title": "سير أعلام النبلاء - ط الرسالة",
        "author": "شمس الدين الذهبي",
        "dataset_id": "siyar_companions_tier1",
        "dataset_name": "الطبقة الأولى - قسم الصحابة",
        "layer": 1,
        "source_group": "الصحابة",
        "person_category": None,
        "character_id": "siyar_deadbeef",
        "character_name": "فلان بن فلان",
        "page_id": 2000,
        "url": "https://read.shamela.ws/book/10906/2000",
        "printed_volume": 1,
        "printed_page": 5,
        "entry_name": "فلان بن فلان",
        "text": "نص السيرة.",
        "content_hash": "deadbeef" * 8,
    }
    record.update(overrides)
    return record


def test_detects_jsonl_format(tmp_path):
    (tmp_path / "abu_bakr_pages.jsonl").write_text(
        json.dumps(JSONL_RECORD, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pages = list(load_source_pages(tmp_path))

    assert len(pages) == 1
    p = pages[0]
    assert p.character_id == "abu_bakr"
    assert p.character_name == "أبو بكر الصديق"
    assert p.printed_volume is None
    assert p.dataset_id is None
    assert p.source_content_hash is None


def test_detects_companions_style_format(tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "2000.json").write_text(json.dumps(_companions_record(), ensure_ascii=False), encoding="utf-8")

    pages = list(load_source_pages(tmp_path))

    assert len(pages) == 1
    p = pages[0]
    assert p.character_id == "siyar_deadbeef"
    assert p.character_name == "فلان بن فلان"
    assert p.collection == "الصحابة"
    assert p.printed_page == "5"  # coerced from int to str, matching the jsonl format's type
    assert p.printed_volume == 1
    assert p.dataset_id == "siyar_companions_tier1"
    assert p.source_content_hash == "deadbeef" * 8


def test_skips_pages_with_no_character_identity(tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "2000.json").write_text(json.dumps(_companions_record(), ensure_ascii=False), encoding="utf-8")
    (pages_dir / "2001.json").write_text(
        json.dumps(
            _companions_record(page_id=2001, character_id=None, character_name=None, entry_name=None),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    skipped = []
    pages = list(load_source_pages(tmp_path, on_skip=lambda pid, reason: skipped.append((pid, reason))))

    assert len(pages) == 1
    assert pages[0].page_id == 2000
    assert skipped == [(2001, "no character_id/character_name in source record")]


def test_prefers_flat_pages_dir_over_a_coexisting_aggregate_jsonl(tmp_path):
    """Regression test: a real dataset directory can have BOTH a flat
    pages/*.json cache AND a *_pages.jsonl aggregate export sitting next
    to it (companions_tier1 does, once its scraper's write_outputs() has
    run) — and the aggregate can use a totally different record schema
    than the Rashidun-style JSONL this loader also supports. Picking the
    wrong one here used to raise KeyError('collection') instead of
    reading the pages/ cache; this must not regress."""
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "2000.json").write_text(json.dumps(_companions_record(), ensure_ascii=False), encoding="utf-8")

    # An aggregate JSONL with a schema the companions-style loader doesn't
    # use (no "collection" key) sitting right next to the pages/ cache.
    (tmp_path / "companions_tier1_pages.jsonl").write_text(
        json.dumps(_companions_record(), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pages = list(load_source_pages(tmp_path))

    assert len(pages) == 1
    assert pages[0].character_id == "siyar_deadbeef"


def test_unrecognized_directory_raises(tmp_path):
    (tmp_path / "not_a_dataset.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="no recognized dataset format"):
        list(load_source_pages(tmp_path))
