"""split character categories into a controlled, multi-valued taxonomy

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-12

`characters.category` (singular) has, since 0006, held exactly the same
value as `characters.group` for every row -- 0006 introduced `group`
specifically to take over the broad-classification role (era-adjacent
grouping like "الخلفاء الراشدون" / "الصحابة"), freeing `category` to
eventually hold a specific per-person role. Rather than reuse that single
scalar column for per-person roles, this migration gives roles their own
controlled, multi-valued home, so a character can hold several roles at
once and only roles from a fixed vocabulary are ever stored:

- `categories` -- the fixed, controlled taxonomy of role labels (e.g.
  "خليفة", "فقيه", "طبيب") that scripts/generate_character_classification.py
  is allowed to assign. Seeded here from that script's CATEGORY_TAXONOMY
  so an invalid label is rejected at the DB level (FK), not only in
  application code. Keep this list in sync with that script's tuple.
- `character_categories` -- a character <-> category join table. The
  composite primary key prevents a character from ever having the same
  category twice; the FK to `categories` prevents any value outside the
  controlled taxonomy. `position` preserves the order roles were
  assigned in (e.g. the model's stated primary role first).

`characters.category` becomes fully redundant once this lands -- every
byte of its meaning already lives in `group` -- so it is dropped rather
than left behind as unused, confusingly-duplicate cruft. `era`/`group`
are untouched; role data lives entirely in the new tables.
"""
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match scripts/generate_character_classification.py's CATEGORY_TAXONOMY.
CATEGORY_TAXONOMY: tuple[str, ...] = (
    "خليفة",
    "أمير",
    "والي",
    "قائد عسكري",
    "فارس",
    "فقيه",
    "محدث",
    "مفسر",
    "مقرئ",
    "قاضٍ",
    "عالم",
    "داعية",
    "معلّم",
    "كاتب",
    "شاعر",
    "أديب",
    "نسابة",
    "طبيب",
    "تاجر",
    "راوية",
)


def upgrade() -> None:
    # Defensive resync in case `group` and `category` ever drifted after
    # 0006 backfilled `group` from `category` (e.g. a manual edit on some
    # deployed environment) -- guarantees no data is lost when `category`
    # is dropped below.
    op.execute('UPDATE characters SET "group" = category WHERE "group" IS DISTINCT FROM category')

    op.create_table(
        "categories",
        sa.Column("code", sa.Text(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    categories_table = sa.table(
        "categories",
        sa.column("code", sa.Text),
        sa.column("created_at", sa.DateTime),
    )
    seeded_at = datetime.now(timezone.utc)
    op.bulk_insert(
        categories_table,
        [{"code": code, "created_at": seeded_at} for code in CATEGORY_TAXONOMY],
    )

    op.create_table(
        "character_categories",
        sa.Column(
            "character_slug",
            sa.String(length=64),
            sa.ForeignKey("characters.slug", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "category_code",
            sa.Text(),
            sa.ForeignKey("categories.code"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_character_categories_category_code", "character_categories", ["category_code"]
    )

    op.drop_column("characters", "category")


def downgrade() -> None:
    op.add_column("characters", sa.Column("category", sa.Text(), nullable=True))
    op.execute('UPDATE characters SET category = "group"')
    op.alter_column("characters", "category", nullable=False)

    op.drop_index("ix_character_categories_category_code", table_name="character_categories")
    op.drop_table("character_categories")
    op.drop_table("categories")
