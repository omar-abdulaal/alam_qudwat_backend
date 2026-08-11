"""SQLAlchemy models for the backend/chat domain.

Deliberately a separate DeclarativeBase from rag.db.models.Base — this
package (app/) is a *consumer* of rag/, not the other way around, and
keeping the RAG-domain tables (documents, chunks) and the app/chat-domain
tables (characters, conversations, messages, story_of_day) on separate
metadata objects keeps rag/ reusable/standalone. Both sets of tables live
in the same physical Postgres database and share one Alembic history.

Linkage between the two domains is by matching string value, not a
cross-base foreign key: ``Character.slug`` is always identical to the
``chunks.character`` / ``documents.caliph_id`` value it corresponds to
(e.g. "abu_bakr").
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AppBase(DeclarativeBase):
    pass


class Character(AppBase):
    """A browsable character (currently the 4 Rashidun caliphs). `slug`
    matches rag chunks/documents' `character`/`caliph_id` values exactly,
    so retrieval filtering and citation linking need no lookup/join."""

    __tablename__ = "characters"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name_ar: Mapped[str] = mapped_column(Text, nullable=False)
    era: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    short_description: Mapped[str] = mapped_column(Text, nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Conversation(AppBase):
    """A chat session scoped to one character. No user-account system yet
    (documented MVP limitation) — the client holds the opaque `id` as its
    only handle, and any request bearing the shared API token plus a valid
    conversation id can read/continue it."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    character_slug: Mapped[str] = mapped_column(
        String(64), ForeignKey("characters.slug"), nullable=False
    )
    narrator_mode: Mapped[str] = mapped_column(String(16), nullable=False)  # "kids" | "adults"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(AppBase):
    """One chat turn. `citations` holds the structured source list for
    assistant turns (mirrors RetrievedChunk fields, JSON-serialized).
    `extra` is unused today — reserved so future features (e.g. branching
    story choices) don't require a schema migration to add."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class StoryOfDay(AppBase):
    """A curated override for a given calendar date's "story of the day".
    Absence of a row for a date is not an error — story_service.py falls
    back to a deterministic rotation over active characters."""

    __tablename__ = "story_of_day"
    __table_args__ = (UniqueConstraint("story_date", name="uq_story_of_day_date"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_date: Mapped[date] = mapped_column(Date, nullable=False)
    character_slug: Mapped[str] = mapped_column(
        String(64), ForeignKey("characters.slug"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    teaser: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
