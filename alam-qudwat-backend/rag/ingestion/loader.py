"""Loads scraper output into a normalized in-memory record, regardless of
which on-disk shape a given dataset's scraper happened to produce.

Two formats are currently supported, auto-detected from ``input_dir``:

- **Rashidun-style**: flat ``{character_id}_pages.jsonl`` files directly
  in ``input_dir`` (produced by scraper/shamela_scraper.py). The combined
  ``rashidun_pages.jsonl`` is intentionally NOT read — as of this writing
  it only contains one caliph's records (a stale/incomplete artifact of
  how the scraper was last run), so per-character files are the source
  of truth.
- **Per-page-JSON-style** (e.g. companions_tier1): one JSON file per page
  under ``input_dir/pages/*.json`` (filename == page_id), each a single
  JSON object with a richer, dataset-specific schema.

Both formats are normalized into the same dataset-agnostic ``SourcePage``.
Adding a third format later means adding a third ``_load_*`` function and
one more branch in ``load_source_pages`` — the rest of the ingestion
pipeline (cleaning, chunking, hashing, diffing, embedding) never needs to
know which format a page came from.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

logger = logging.getLogger("rag.ingestion.loader")

# Filenames written by the combined-output pass of the Rashidun scraper;
# never treated as an ingestion source (see module docstring).
_EXCLUDED_FILENAMES = {"rashidun_pages.jsonl"}

# Called with (page_id, reason) for every source record that exists but
# can't be turned into a SourcePage (e.g. no character identity at all).
# Defaults to just logging; ingest.py passes one that also increments a
# stats counter.
SkipCallback = Callable[[int, str], None]


def _default_on_skip(page_id: int, reason: str) -> None:
    logger.warning("skipping page_id=%s: %s", page_id, reason)


@dataclass(frozen=True)
class SourcePage:
    book_id: int
    book_title: str
    author: str
    collection: str
    character_id: str
    character_name: str
    page_id: int
    printed_page: str | None
    source_url: str
    raw_text: str
    printed_volume: int | None = None
    dataset_id: str | None = None
    # The hash the *source* itself provided, if any — preserved verbatim.
    # Never used for idempotency (rag.ingestion.hashing.content_hash(),
    # computed fresh from raw_text, is what ingestion diffs against).
    source_content_hash: str | None = None


def _iter_jsonl_files(input_dir: Path) -> Iterator[Path]:
    for path in sorted(input_dir.glob("*_pages.jsonl")):
        if path.name in _EXCLUDED_FILENAMES:
            continue
        yield path


def _load_jsonl_style_pages(input_dir: Path) -> Iterator[SourcePage]:
    for path in _iter_jsonl_files(input_dir):
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON — {exc}") from exc

                yield SourcePage(
                    book_id=record["book_id"],
                    book_title=record["book_title"],
                    author=record["author"],
                    collection=record["collection"],
                    character_id=record["caliph_id"],
                    character_name=record["caliph_name"],
                    page_id=record["page_id"],
                    printed_page=record.get("printed_page"),
                    source_url=record["url"],
                    raw_text=record["text"],
                )


def _load_companions_style_pages(input_dir: Path, on_skip: SkipCallback) -> Iterator[SourcePage]:
    pages_dir = input_dir / "pages"
    files = sorted(pages_dir.glob("*.json"), key=lambda p: int(p.stem))

    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            try:
                record = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON — {exc}") from exc

        page_id = record["page_id"]
        character_id = record.get("character_id")
        character_name = record.get("character_name")

        # Some pages are section/transition headers spanning multiple
        # people (e.g. "السابقون الأولون") rather than one person's
        # biography — the source itself withholds a character identity
        # for these (character_id/character_name are null). Respect that:
        # skip rather than guessing or forcing a NOT NULL character column.
        if not character_id or not character_name:
            on_skip(page_id, "no character_id/character_name in source record")
            continue

        printed_page = record.get("printed_page")

        yield SourcePage(
            book_id=record["book_id"],
            book_title=record["book_title"],
            author=record["author"],
            collection=record.get("source_group") or "",
            character_id=character_id,
            character_name=character_name,
            page_id=page_id,
            printed_page=str(printed_page) if printed_page is not None else None,
            source_url=record["url"],
            raw_text=record["text"],
            printed_volume=record.get("printed_volume"),
            dataset_id=record.get("dataset_id"),
            source_content_hash=record.get("content_hash"),
        )


def load_source_pages(input_dir: Path, on_skip: SkipCallback = _default_on_skip) -> Iterator[SourcePage]:
    """Yield every valid page record from ``input_dir``, auto-detecting
    which on-disk format it uses (see module docstring).

    Checked in this order — a flat ``pages/*.json`` cache first, then
    ``*_pages.jsonl`` — because a dataset directory can legitimately have
    both at once (e.g. a scraper's resumable per-page cache *and* a
    convenience aggregate JSONL export it writes at the end of a run,
    which can use a different record schema than the Rashidun-style JSONL
    this loader also supports). The per-page cache is the primary,
    always-current source; an aggregate export is derived from it, so it
    loses ties. This doesn't affect Rashidun, whose own pages/ directory
    is nested one level deeper by character (pages/{character_id}/*.json)
    and so never matches the flat pages/*.json check.
    """
    has_pages_dir = (input_dir / "pages").is_dir() and any((input_dir / "pages").glob("*.json"))
    has_jsonl = any(_iter_jsonl_files(input_dir))

    if has_pages_dir:
        yield from _load_companions_style_pages(input_dir, on_skip)
    elif has_jsonl:
        yield from _load_jsonl_style_pages(input_dir)
    else:
        raise ValueError(
            f"{input_dir}: no recognized dataset format found "
            f"(expected either *_pages.jsonl files, or a pages/*.json subdirectory)"
        )
