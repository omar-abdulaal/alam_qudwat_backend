"""SQLAlchemy models for the RAG store.

Two tables:

``documents``
    One row per scraped source page (the unit the scraper produces).
    Kept mainly so re-ingestion can detect whether a page's raw text has
    changed without re-reading the chunk table, and so every chunk can be
    traced back to exactly one page.

``chunks``
    One row per retrievable chunk. Metadata is denormalized onto the chunk
    (rather than requiring a join) so retrieval queries stay a single,
    index-friendly SELECT. This is the table embeddings live in.

Despite the names, ``caliph_id``/``caliph_name``/``character`` are generic
"who this text is about" fields, not caliph-specific — the same columns
hold e.g. Companions' hash-based ``character_id`` values from the
companions_tier1 dataset. Renamed columns weren't worth the churn on a
live schema; see rag/ingestion/loader.py's SourcePage for the
dataset-agnostic field names used everywhere upstream of these models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1536  # must match Settings.embedding_dim / the migration


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """A single scraped page (source of truth: data/raw/rashidun/*_pages.jsonl)."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("book_id", "page_id", name="uq_documents_book_page"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    book_id: Mapped[int] = mapped_column(Integer, nullable=False)
    book_title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str] = mapped_column(Text, nullable=False)
    collection: Mapped[str] = mapped_column(Text, nullable=False)

    caliph_id: Mapped[str] = mapped_column(String(64), nullable=False)
    caliph_name: Mapped[str] = mapped_column(Text, nullable=False)
    era: Mapped[str] = mapped_column(Text, nullable=False)

    page_id: Mapped[int] = mapped_column(Integer, nullable=False)
    printed_page: Mapped[str | None] = mapped_column(String(32), nullable=True)
    printed_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    # Which ingestion source this page came from, e.g. "siyar_companions_tier1"
    # (also encodes the dataset's "layer"/tier — a future tier2 dataset gets
    # its own dataset_id, no schema change needed). NULL for datasets that
    # predate this concept (Rashidun).
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # The content hash the *source* provided (when it does), preserved
    # verbatim for provenance/cross-checking — distinct from content_hash
    # above, which this project always computes itself and is what
    # idempotency actually keys off.
    source_content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """A retrievable slice of a Document's raw_text, with its embedding."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("ix_chunks_character", "character"),
        Index("ix_chunks_era", "era"),
        Index("ix_chunks_book_title", "book_title"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    # Denormalized source metadata — kept identical to the parent Document's
    # values so every retrieval hit is self-describing / directly citable.
    character: Mapped[str] = mapped_column(String(64), nullable=False)
    caliph_name: Mapped[str] = mapped_column(Text, nullable=False)
    book_title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str] = mapped_column(Text, nullable=False)
    era: Mapped[str] = mapped_column(Text, nullable=False)
    page_id: Mapped[int] = mapped_column(Integer, nullable=False)
    printed_page: Mapped[str | None] = mapped_column(String(32), nullable=True)
    printed_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")
