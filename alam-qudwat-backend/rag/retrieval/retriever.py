"""Semantic + metadata-filtered retrieval over the chunks table.

Every result is a fully self-describing, directly citable hit — no
separate lookup is needed to reconstruct the source (character, book,
author, page, URL).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.db.models import Chunk
from rag.embeddings.base import EmbeddingProvider


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float  # cosine similarity, higher is more relevant (1.0 = identical)
    character: str
    caliph_name: str
    book_title: str
    author: str
    era: str
    page_id: int
    printed_page: str | None
    source_url: str

    def citation(self) -> str:
        page = self.printed_page or str(self.page_id)
        return f"{self.book_title}، {self.author}، صفحة {page} — {self.source_url}"


def retrieve(
    session: Session,
    query: str,
    embedder: EmbeddingProvider,
    *,
    top_k: int = 5,
    character: str | None = None,
    era: str | None = None,
    book_title: str | None = None,
) -> list[RetrievedChunk]:
    """Embed `query` and return the top_k most similar chunks, optionally
    restricted by character / era / book_title."""
    (query_vector,) = embedder.embed([query])

    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = select(Chunk, distance.label("distance")).where(Chunk.embedding.is_not(None))

    if character is not None:
        stmt = stmt.where(Chunk.character == character)
    if era is not None:
        stmt = stmt.where(Chunk.era == era)
    if book_title is not None:
        stmt = stmt.where(Chunk.book_title == book_title)

    stmt = stmt.order_by(distance).limit(top_k)

    results: list[RetrievedChunk] = []
    for chunk, dist in session.execute(stmt).all():
        results.append(
            RetrievedChunk(
                chunk_id=str(chunk.id),
                text=chunk.text,
                score=1.0 - float(dist),
                character=chunk.character,
                caliph_name=chunk.caliph_name,
                book_title=chunk.book_title,
                author=chunk.author,
                era=chunk.era,
                page_id=chunk.page_id,
                printed_page=chunk.printed_page,
                source_url=chunk.source_url,
            )
        )
    return results
