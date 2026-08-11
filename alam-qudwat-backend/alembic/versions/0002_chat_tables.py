"""chat backend: characters, conversations, messages, story_of_day

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-09

"""
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Slugs match rag chunks.character / documents.caliph_id exactly.
CHARACTER_SEED = [
    {
        "slug": "abu_bakr",
        "name_ar": "أبو بكر الصديق",
        "era": "الخلافة الراشدة",
        "category": "الخلفاء الراشدون",
        "short_description": "أول الخلفاء الراشدين، وأقرب صحابة النبي صلى الله عليه وسلم وصاحبه في الهجرة.",
        "sort_order": 1,
    },
    {
        "slug": "umar",
        "name_ar": "عمر بن الخطاب",
        "era": "الخلافة الراشدة",
        "category": "الخلفاء الراشدون",
        "short_description": "ثاني الخلفاء الراشدين، عُرف بالعدل وبتوسّع الدولة الإسلامية في عهده.",
        "sort_order": 2,
    },
    {
        "slug": "uthman",
        "name_ar": "عثمان بن عفان",
        "era": "الخلافة الراشدة",
        "category": "الخلفاء الراشدون",
        "short_description": "ثالث الخلفاء الراشدين، جمع القرآن في مصحف واحد وواصل الفتوحات الإسلامية.",
        "sort_order": 3,
    },
    {
        "slug": "ali",
        "name_ar": "علي بن أبي طالب",
        "era": "الخلافة الراشدة",
        "category": "الخلفاء الراشدون",
        "short_description": "رابع الخلفاء الراشدين وابن عم النبي صلى الله عليه وسلم وصهره.",
        "sort_order": 4,
    },
]


def upgrade() -> None:
    op.create_table(
        "characters",
        sa.Column("slug", sa.String(length=64), primary_key=True),
        sa.Column("name_ar", sa.Text(), nullable=False),
        sa.Column("era", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("short_description", sa.Text(), nullable=False),
        sa.Column("avatar_url", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("character_slug", sa.String(length=64), sa.ForeignKey("characters.slug"), nullable=False),
        sa.Column("narrator_mode", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_conversations_character_slug", "conversations", ["character_slug"])

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=True),
        sa.Column("extra", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    op.create_table(
        "story_of_day",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("story_date", sa.Date(), nullable=False),
        sa.Column("character_slug", sa.String(length=64), sa.ForeignKey("characters.slug"), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("teaser", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("story_date", name="uq_story_of_day_date"),
    )

    characters_table = sa.table(
        "characters",
        sa.column("slug", sa.String),
        sa.column("name_ar", sa.Text),
        sa.column("era", sa.Text),
        sa.column("category", sa.Text),
        sa.column("short_description", sa.Text),
        sa.column("sort_order", sa.Integer),
        sa.column("created_at", sa.DateTime),
    )
    seeded_at = datetime.now(timezone.utc)
    op.bulk_insert(
        characters_table,
        [{**row, "created_at": seeded_at} for row in CHARACTER_SEED],
    )


def downgrade() -> None:
    op.drop_table("story_of_day")
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_character_slug", table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("characters")
