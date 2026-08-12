"""Clear short_description so the next classification `submit` actually
regenerates it, instead of skipping characters that already look
"complete".

scripts/generate_character_classification.py decides whether a character
needs an AI-written description purely by "is short_description
non-empty" -- so retrying after a prompt change (e.g. the "don't mention
the character's name" rule added to _SYSTEM_PROMPT) requires clearing the
old text first, or `submit` will just skip everyone again.

Two selection modes:

  (default) --placeholder   Only rows that still hold the generic filler
      sentence alembic/versions/0004_seed_companions_tier1_characters.py
      seeded every companions_tier1 character with at first
      ("من الصحابة، وردت ترجمته في كتاب سير أعلام النبلاء للذهبي.").
      Use this the first time, before any real AI description exists yet.

  --all-companions   Every companions_tier1 character (any siyar_* slug),
      regardless of current text -- use this to force a full re-run after
      changing the prompt/taxonomy, even though every companion already
      has a real (but now-outdated) AI-written description.

Neither mode ever touches the 4 Rashidun (abu_bakr/umar/uthman/ali) or
`categories` -- only short_description on companions_tier1 characters.

IMPORTANT -- do this immediately after running this script, before
starting the backend (uvicorn, pytest, or anything else that boots the
FastAPI app) again: re-run `python -m scripts.export_character_
classifications` to refresh data/generated/character_classifications.json.
That file is what app/services/character_classification_sync.py applies
to the DB on every startup -- if it's left stale (still holding the old
text), the very next backend boot will silently overwrite the fields
this script just cleared, undoing the reset.

Usage:
    python -m scripts.reset_placeholder_descriptions [--dry-run]
    python -m scripts.reset_placeholder_descriptions --all-companions [--dry-run]
"""
from __future__ import annotations

import argparse

from app.db.models import Character
from rag.db.session import session_scope

PLACEHOLDER_SHORT_DESCRIPTION = "من الصحابة، وردت ترجمته في كتاب سير أعلام النبلاء للذهبي."

# Everyone else in the DB is a companions_tier1 character (slug starts
# with "siyar_") -- see alembic/versions/0004_seed_companions_tier1_characters.py.
RASHIDUN_SLUGS = frozenset({"abu_bakr", "umar", "uthman", "ali"})


def reset_descriptions(all_companions: bool = False, dry_run: bool = False) -> int:
    with session_scope() as session:
        if all_companions:
            matches = [c for c in session.query(Character).all() if c.slug not in RASHIDUN_SLUGS]
        else:
            matches = session.query(Character).filter_by(short_description=PLACEHOLDER_SHORT_DESCRIPTION).all()
        count = len(matches)
        if not dry_run:
            for character in matches:
                character.short_description = ""
        else:
            session.rollback()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report how many rows would change, without writing")
    parser.add_argument(
        "--all-companions",
        action="store_true",
        help="Clear every companions_tier1 character's short_description, regardless of current text",
    )
    args = parser.parse_args(argv)

    count = reset_descriptions(all_companions=args.all_companions, dry_run=args.dry_run)
    verb = "Would clear" if args.dry_run else "Cleared"
    print(f"{verb} short_description on {count} character(s).")
    if not args.dry_run and count:
        print(
            "IMPORTANT: run `python -m scripts.export_character_classifications` now, before starting "
            "the backend again -- otherwise the next startup sync will silently undo this."
        )
        print("Then: `python -m scripts.generate_character_classification submit` (no --force needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
