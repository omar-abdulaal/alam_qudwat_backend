"""Startup sync of AI-classified character data (categories + short
description) from a committed JSON snapshot into the DB.

scripts/generate_character_classification.py calls OpenAI's Batch API and
is meant to be run manually, locally, by a developer who reviews the
results — never on a server (no OPENAI_API_KEY needed there, no per-
deploy AI cost). Once satisfied with a local run,
scripts/export_character_classifications.py snapshots the DB's current
`categories`/`short_description` for every character into
data/generated/character_classifications.json, which gets committed and
shipped with the rest of the code. This module applies that snapshot to
whatever DB the backend is actually running against, on every startup —
so a server only ever needs the normal deploy (pull code, `alembic
upgrade head`, restart) to pick up newly-classified characters; it never
talks to OpenAI for this.

Unlike app/services/rag_sync.py, this makes no network calls — the file
is already on disk and the target rows are a few hundred small text/list
updates — so it's cheap enough to run synchronously before the app starts
serving, rather than backgrounded on its own thread. Never raises: any
failure here (missing file, bad JSON, DB unreachable) is logged and
swallowed so it can never crash the API process.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.db.models import Character
from rag.db.session import session_scope

logger = logging.getLogger("app.character_classification_sync")

_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "generated" / "character_classifications.json"
)


def sync_character_classifications(snapshot_path: Path | None = None) -> None:
    path = snapshot_path or _SNAPSHOT_PATH
    if not path.is_file():
        logger.info("no classification snapshot found at %s, nothing to sync", path)
        return

    try:
        snapshot: dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("could not read/parse classification snapshot %s; skipping sync", path)
        return

    updated = 0
    unknown = 0
    try:
        with session_scope() as session:
            for slug, fields in snapshot.items():
                character = session.get(Character, slug)
                if character is None:
                    unknown += 1
                    continue

                changed = False
                categories = fields.get("categories")
                if isinstance(categories, list) and character.categories != categories:
                    character.categories = categories
                    changed = True

                short_description = fields.get("short_description")
                if isinstance(short_description, str) and short_description.strip():
                    if character.short_description != short_description:
                        character.short_description = short_description
                        changed = True

                if changed:
                    updated += 1
    except Exception:
        logger.exception("classification sync failed; API continues serving without it")
        return

    logger.info(
        "classification sync finished: %d character(s) updated, %d unknown slug(s) skipped (snapshot had %d entries)",
        updated,
        unknown,
        len(snapshot),
    )
