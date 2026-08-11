"""Ingestion CLI: load -> clean -> chunk -> diff against DB -> embed -> upsert.

Deliberately a standalone script, run manually or via cron/CI — NOT
imported or triggered by any backend request path. The backend only ever
reads the resulting Postgres tables at query time.

Works against any dataset directory rag.ingestion.loader knows how to
read (auto-detected — see its docstring), e.g.:
    python -m rag.ingestion.ingest --input data/raw/rashidun
    python -m rag.ingestion.ingest --input data/raw/companions_tier1

Re-running with unchanged source files is a no-op beyond a handful of
cheap SELECTs (see rag/ingestion/hashing.py): unchanged pages are
skipped before chunking even runs, and unchanged chunks are skipped
before the embedding API is called.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session, selectinload

from rag.config import Settings, get_settings
from rag.db.models import Chunk, Document
from rag.db.session import session_scope
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.ingestion.chunker import chunk_text
from rag.ingestion.cleaning import clean_text
from rag.ingestion.hashing import content_hash
from rag.ingestion.loader import SourcePage, load_source_pages

logger = logging.getLogger("rag.ingestion")


@dataclass
class IngestionStats:
    pages_seen: int = 0
    pages_skipped: int = 0
    pages_new: int = 0
    pages_updated: int = 0
    pages_unchanged: int = 0
    chunks_new: int = 0
    chunks_updated: int = 0
    chunks_unchanged: int = 0
    chunks_deleted: int = 0
    embedding_calls: int = 0
    embedded_chunks: int = 0
    # Only populated by ingest_missing_characters() — run_ingestion() works
    # page-by-page, not grouped by character, so these stay 0 there.
    characters_checked: int = 0
    characters_added: int = 0
    characters_failed: int = 0

    def summary(self) -> str:
        return (
            f"pages: seen={self.pages_seen} skipped={self.pages_skipped} new={self.pages_new} "
            f"updated={self.pages_updated} unchanged={self.pages_unchanged} | "
            f"chunks: new={self.chunks_new} updated={self.chunks_updated} "
            f"unchanged={self.chunks_unchanged} deleted={self.chunks_deleted} | "
            f"embedded={self.embedded_chunks} in {self.embedding_calls} API call(s)"
        )


def _upsert_document(session: Session, page: SourcePage, era: str, stats: IngestionStats) -> tuple[Document, bool]:
    """Return (document, changed). changed=False means the page's raw text
    is byte-identical to what's already stored — caller should skip
    re-chunking entirely."""
    existing = (
        session.query(Document)
        .options(selectinload(Document.chunks))
        .filter_by(book_id=page.book_id, page_id=page.page_id)
        .one_or_none()
    )
    doc_hash = content_hash(page.raw_text)

    if existing is not None and existing.content_hash == doc_hash:
        stats.pages_unchanged += 1
        return existing, False

    if existing is None:
        doc = Document(
            id=uuid.uuid4(),
            book_id=page.book_id,
            book_title=page.book_title,
            author=page.author,
            collection=page.collection,
            caliph_id=page.character_id,
            caliph_name=page.character_name,
            era=era,
            page_id=page.page_id,
            printed_page=page.printed_page,
            printed_volume=page.printed_volume,
            dataset_id=page.dataset_id,
            source_url=page.source_url,
            raw_text=page.raw_text,
            content_hash=doc_hash,
            source_content_hash=page.source_content_hash,
        )
        session.add(doc)
        stats.pages_new += 1
    else:
        doc = existing
        doc.book_title = page.book_title
        doc.author = page.author
        doc.collection = page.collection
        doc.caliph_name = page.character_name
        doc.era = era
        doc.printed_page = page.printed_page
        doc.printed_volume = page.printed_volume
        doc.dataset_id = page.dataset_id
        doc.source_url = page.source_url
        doc.raw_text = page.raw_text
        doc.content_hash = doc_hash
        doc.source_content_hash = page.source_content_hash
        stats.pages_updated += 1

    session.flush()
    return doc, True


def _sync_chunks(
    session: Session,
    doc: Document,
    page: SourcePage,
    settings,
    stats: IngestionStats,
) -> list[Chunk]:
    """Upsert this document's chunk rows (text only, no embedding yet).
    Returns the list of chunk rows that need a fresh embedding."""
    cleaned = clean_text(page.raw_text)
    new_chunks = chunk_text(
        cleaned,
        max_tokens=settings.chunk_token_size,
        overlap_tokens=settings.chunk_token_overlap,
        min_tokens=settings.chunk_min_token_size,
    )

    existing_by_index = {c.chunk_index: c for c in doc.chunks}
    kept_indices: set[int] = set()
    needs_embedding: list[Chunk] = []

    for c in new_chunks:
        kept_indices.add(c.index)
        c_hash = content_hash(c.text)
        existing_chunk = existing_by_index.get(c.index)

        if existing_chunk is not None and existing_chunk.content_hash == c_hash:
            stats.chunks_unchanged += 1
            continue

        if existing_chunk is not None:
            existing_chunk.text = c.text
            existing_chunk.token_count = c.token_count
            existing_chunk.content_hash = c_hash
            existing_chunk.embedding = None
            row = existing_chunk
            stats.chunks_updated += 1
        else:
            row = Chunk(
                id=uuid.uuid4(),
                document_id=doc.id,
                chunk_index=c.index,
                text=c.text,
                token_count=c.token_count,
                content_hash=c_hash,
                character=page.character_id,
                caliph_name=page.character_name,
                book_title=page.book_title,
                author=page.author,
                era=doc.era,
                page_id=page.page_id,
                printed_page=page.printed_page,
                printed_volume=page.printed_volume,
                dataset_id=page.dataset_id,
                source_url=page.source_url,
            )
            session.add(row)
            stats.chunks_new += 1

        needs_embedding.append(row)

    for idx, old_chunk in existing_by_index.items():
        if idx not in kept_indices:
            session.delete(old_chunk)
            stats.chunks_deleted += 1

    session.flush()
    return needs_embedding


def _embed_pending(
    session: Session,
    pending: list[Chunk],
    embedder: EmbeddingProvider,
    batch_size: int,
    stats: IngestionStats,
) -> None:
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed([row.text for row in batch])
        stats.embedding_calls += 1
        for row, vector in zip(batch, vectors):
            row.embedding = vector
            stats.embedded_chunks += 1
        session.flush()


def run_ingestion(
    input_dir: Path,
    session: Session,
    embedder: EmbeddingProvider,
    settings=None,
) -> IngestionStats:
    settings = settings or get_settings()
    stats = IngestionStats()
    pending: list[Chunk] = []

    def _on_skip(page_id: int, reason: str) -> None:
        stats.pages_skipped += 1
        logger.warning("skipping page_id=%s: %s", page_id, reason)

    for page in load_source_pages(input_dir, on_skip=_on_skip):
        stats.pages_seen += 1
        era = settings.era_for_page(page.character_id, page.collection)
        doc, changed = _upsert_document(session, page, era, stats)
        if not changed:
            continue
        pending.extend(_sync_chunks(session, doc, page, settings, stats))

    _embed_pending(session, pending, embedder, settings.embedding_batch_size, stats)
    return stats


def ingest_missing_characters(
    session: Session,
    embedder: EmbeddingProvider,
    settings: Settings | None = None,
    data_dir: Path | None = None,
) -> IngestionStats:
    """Ingest every character present in any dataset under `data_dir` that
    has no rows in `documents` yet — characters already present are
    skipped entirely, with no content-hash comparison and no embedding
    calls for them at all. This is deliberately coarser than
    run_ingestion()'s per-page diffing: it answers "is this character in
    the DB at all?", not "did this page's text change?". To force a
    character to be re-ingested (e.g. the source text was corrected),
    delete its rows first (see rag.ingestion.delete_character) — the next
    call will then see it as missing and ingest it fresh.

    Intended for the backend's startup sync (app/services/rag_sync.py),
    where resilience matters more than throughput: each character is
    processed and committed as its own small unit, wrapped in its own
    try/except, so one failure (a transient API error, a race with
    another instance hitting the same unique constraint) never aborts
    the sync for every other missing character.
    """
    settings = settings or get_settings()
    data_dir = data_dir or settings.data_dir
    stats = IngestionStats()
    started_at = time.monotonic()

    if not data_dir.exists():
        logger.info("data_dir does not exist, nothing to sync: %s", data_dir)
        return stats

    known_character_ids = {row[0] for row in session.query(Document.caliph_id).distinct().all()}
    logger.info(
        "checking for missing characters under %s (%d character(s) already in the DB)",
        data_dir,
        len(known_character_ids),
    )

    for dataset_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        pages_by_character: dict[str, list[SourcePage]] = defaultdict(list)

        def _on_skip(page_id: int, reason: str, _dataset_name: str = dataset_dir.name) -> None:
            stats.pages_skipped += 1
            logger.warning("[%s] skipping page_id=%s: %s", _dataset_name, page_id, reason)

        try:
            for page in load_source_pages(dataset_dir, on_skip=_on_skip):
                if page.character_id not in known_character_ids:
                    pages_by_character[page.character_id].append(page)
        except ValueError:
            logger.info("skipping %s: not a recognized dataset directory", dataset_dir)
            continue

        if not pages_by_character:
            continue

        logger.info(
            "[%s] %d character(s) missing, ingesting now", dataset_dir.name, len(pages_by_character)
        )

        for character_id, pages in pages_by_character.items():
            stats.characters_checked += 1
            try:
                pending: list[Chunk] = []
                for page in pages:
                    stats.pages_seen += 1
                    era = settings.era_for_page(page.character_id, page.collection)
                    doc, _changed = _upsert_document(session, page, era, stats)
                    pending.extend(_sync_chunks(session, doc, page, settings, stats))
                _embed_pending(session, pending, embedder, settings.embedding_batch_size, stats)
                session.commit()
                known_character_ids.add(character_id)
                stats.characters_added += 1
                logger.info(
                    "[%s] added character_id=%s (%d page(s))", dataset_dir.name, character_id, len(pages)
                )
            except Exception:
                session.rollback()
                stats.characters_failed += 1
                logger.exception(
                    "failed to ingest character_id=%s from %s; skipping", character_id, dataset_dir.name
                )

    elapsed = time.monotonic() - started_at
    logger.info(
        "finished checking for missing characters in %.1fs: %d checked, %d added, %d failed (%s)",
        elapsed,
        stats.characters_checked,
        stats.characters_added,
        stats.characters_failed,
        stats.summary(),
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/rashidun"),
        help="Dataset directory — either *_pages.jsonl files or a pages/*.json "
        "subdirectory (default: data/raw/rashidun)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    if not args.input.exists():
        print(f"Input directory does not exist: {args.input}", file=sys.stderr)
        return 1

    settings = get_settings()
    embedder = OpenAIEmbeddingProvider()

    with session_scope() as session:
        stats = run_ingestion(args.input, session, embedder, settings)

    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
