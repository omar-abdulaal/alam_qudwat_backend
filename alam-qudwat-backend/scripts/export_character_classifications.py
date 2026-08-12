"""Snapshot every character's current `categories`/`short_description`
from the DB into a committed JSON file.

scripts/generate_character_classification.py calls OpenAI's Batch API and
is meant to be run manually against a local/dev database — never on a
server (no OPENAI_API_KEY needed there, no per-deploy AI cost). Once
you're satisfied with a local `apply` run (and any manual touch-ups),
run this to snapshot the results:

    python -m scripts.export_character_classifications

Commit the resulting data/generated/character_classifications.json.
app/services/character_classification_sync.py reads that file and
applies it to the DB automatically on every backend startup, so a server
only ever needs the normal deploy (pull code, `alembic upgrade head`,
restart) to pick up newly-classified characters — it never re-runs the
classification script or talks to OpenAI for this.

Exports every character (not just ones with a non-empty `categories`),
so an unclassified character's current short_description is preserved in
the snapshot too — the sync applies both fields, and a character with no
assigned roles yet is still represented correctly (empty list).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.db.models import Character
from rag.db.session import session_scope

_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "generated" / "character_classifications.json"


def export_classifications(output_path: Path = _OUTPUT_PATH) -> int:
    with session_scope() as session:
        characters = session.query(Character).order_by(Character.slug).all()
        snapshot = {
            character.slug: {
                "categories": character.categories,
                "short_description": character.short_description,
            }
            for character in characters
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(snapshot)


def main() -> int:
    count = export_classifications()
    print(f"Exported {count} character(s) to {_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
