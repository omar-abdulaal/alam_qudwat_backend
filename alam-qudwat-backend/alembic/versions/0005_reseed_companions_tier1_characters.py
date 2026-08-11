"""reseed missing companions_tier1 characters

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-12

"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ERA = "الصحابة"
_CATEGORY = "الصحابة"

# alembic/versions/0005_....py -> repo root
_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "companions_tier1"
    / "characters"
)


def _load_seed_rows() -> list[dict]:
    by_character_id: dict[str, dict] = {}

    for metadata_path in sorted(_DATA_DIR.glob("*/metadata.json")):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))

        character_id = meta["character_id"]

        existing = by_character_id.get(character_id)
        if (
            existing is not None
            and existing["page_id_start"] <= meta["page_id_start"]
        ):
            continue

        by_character_id[character_id] = meta

    rows = []

    for sort_order, meta in enumerate(
        sorted(
            by_character_id.values(),
            key=lambda m: m["page_id_start"],
        ),
        start=5,
    ):
        rows.append(
            {
                "slug": meta["character_id"],
                "name_ar": meta["character_name"].strip(),
                "era": _ERA,
                "category": _CATEGORY,
                "short_description": (
                    "من الصحابة، وردت ترجمته في كتاب "
                    "سير أعلام النبلاء للذهبي."
                ),
                "sort_order": sort_order,
            }
        )

    return rows


def upgrade() -> None:
    # Important: fail instead of silently marking the migration as applied.
    if not _DATA_DIR.is_dir():
        raise RuntimeError(
            f"Companions seed directory not found: {_DATA_DIR}"
        )

    rows = _load_seed_rows()

    if not rows:
        raise RuntimeError(
            f"No companion metadata files found under: {_DATA_DIR}"
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

    connection = op.get_bind()

    existing_slugs = set(
        connection.execute(
            sa.select(characters_table.c.slug)
        ).scalars()
    )

    missing_rows = [
        row
        for row in rows
        if row["slug"] not in existing_slugs
    ]

    if not missing_rows:
        return

    seeded_at = datetime.now(timezone.utc)

    op.bulk_insert(
        characters_table,
        [
            {
                **row,
                "created_at": seeded_at,
            }
            for row in missing_rows
        ],
    )


def downgrade() -> None:
    # Corrective data migration.
    # Intentionally left non-destructive because some environments
    # may already have received these rows from migration 0004.
    pass