"""seed companions_tier1 characters into the characters table

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-11

"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Matches rag.config.COLLECTION_ERA_MAP["الصحابة"] — companions have no
# individual per-person era in the source, only this collection-level label.
_ERA = "الصحابة"
_CATEGORY = "الصحابة"

# alembic/versions/0004_....py -> parents[2] == repo root
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "companions_tier1" / "characters"


def _load_seed_rows() -> list[dict]:
    """One row per distinct character_id found under companions_tier1/characters/,
    matching exactly the caliph_id values rag.ingestion assigns these pages
    (see rag/ingestion/loader.py's _load_companions_style_pages), so
    Character.slug lines up with chunks.character / documents.caliph_id with
    no lookup/join needed (same invariant as the Rashidun seed in 0002).

    A handful of character_ids have more than one biography folder (the
    source splits some entries across the book); the earliest one
    (lowest page_id_start) is kept as canonical.
    """
    by_character_id: dict[str, dict] = {}
    for metadata_path in sorted(_DATA_DIR.glob("*/metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        character_id = meta["character_id"]
        existing = by_character_id.get(character_id)
        if existing is not None and existing["page_id_start"] <= meta["page_id_start"]:
            continue
        by_character_id[character_id] = meta

    rows = []
    for sort_order, meta in enumerate(
        sorted(by_character_id.values(), key=lambda m: m["page_id_start"]),
        start=5,  # 1-4 are the Rashidun seed from 0002
    ):
        rows.append(
            {
                "slug": meta["character_id"],
                "name_ar": meta["character_name"].strip(),
                "era": _ERA,
                "category": _CATEGORY,
                "short_description": "من الصحابة، وردت ترجمته في كتاب سير أعلام النبلاء للذهبي.",
                "sort_order": sort_order,
            }
        )
    return rows


def upgrade() -> None:
    if not _DATA_DIR.is_dir():
        # Source data not present in this checkout (shouldn't happen, it's
        # tracked in git) -- skip rather than fail a migration that has
        # nothing to seed from.
        return

    rows = _load_seed_rows()
    if not rows:
        return

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
        [{**row, "created_at": seeded_at} for row in rows],
    )


def downgrade() -> None:
    # character_id values for this dataset are always "siyar_<hash>" (see
    # metadata.json's biography_id/character_id fields) -- distinguishes
    # these rows from the Rashidun seed without relying on sort_order.
    op.execute(sa.text("DELETE FROM characters WHERE slug LIKE 'siyar_%'"))
