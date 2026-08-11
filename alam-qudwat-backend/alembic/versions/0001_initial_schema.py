"""initial schema: documents, chunks, pgvector extension

Revision ID: 0001
Revises:
Create Date: 2026-08-09

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("book_id", sa.Integer(), nullable=False),
        sa.Column("book_title", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=False),
        sa.Column("collection", sa.Text(), nullable=False),
        sa.Column("caliph_id", sa.String(length=64), nullable=False),
        sa.Column("caliph_name", sa.Text(), nullable=False),
        sa.Column("era", sa.Text(), nullable=False),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("printed_page", sa.String(length=32), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("book_id", "page_id", name="uq_documents_book_page"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("character", sa.String(length=64), nullable=False),
        sa.Column("caliph_name", sa.Text(), nullable=False),
        sa.Column("book_title", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=False),
        sa.Column("era", sa.Text(), nullable=False),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("printed_page", sa.String(length=32), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )

    op.create_index("ix_chunks_character", "chunks", ["character"])
    op.create_index("ix_chunks_era", "chunks", ["era"])
    op.create_index("ix_chunks_book_title", "chunks", ["book_title"])

    # HNSW index for cosine-distance ANN search. Requires pgvector >= 0.5.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_book_title", table_name="chunks")
    op.drop_index("ix_chunks_era", table_name="chunks")
    op.drop_index("ix_chunks_character", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
