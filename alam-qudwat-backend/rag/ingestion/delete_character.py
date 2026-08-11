"""Delete a character's data from the RAG store.

The other half of the "update = delete then re-ingest" workflow used by
rag.ingestion.ingest.ingest_missing_characters(): that function only ever
adds characters that aren't in the DB yet, so forcing a re-ingest (e.g.
after the source text was corrected) means removing the character's rows
first — the next ingestion run (manual or the backend's startup sync)
will then see it as missing and ingest it fresh from the source files.

Usage:
    python -m rag.ingestion.delete_character <character_id>
    python -m rag.ingestion.delete_character <character_id> --yes

Deleting a Document cascades to its Chunks automatically via the existing
ON DELETE CASCADE foreign key (rag/db/models.py) — no separate chunk
deletion logic needed.
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import func, select

from rag.db.models import Chunk, Document
from rag.db.session import session_scope

# Arabic character names must print correctly regardless of the terminal's
# default codepage (e.g. Windows consoles default to cp1252, which can't
# encode Arabic and would otherwise crash this script before it even gets
# to the confirmation prompt).
for _stream in (sys.stdout, sys.stdin):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("character_id", help="The character/caliph_id to delete, e.g. abu_bakr or siyar_...")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    args = parser.parse_args(argv)

    with session_scope() as session:
        documents = session.scalars(select(Document).filter_by(caliph_id=args.character_id)).all()

        if not documents:
            print(f"No documents found for character_id={args.character_id!r}. Nothing to do.")
            return 0

        chunk_count = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id.in_([d.id for d in documents]))
        )
        character_name = documents[0].caliph_name
        book_title = documents[0].book_title

        print(f"character_id : {args.character_id}")
        print(f"character    : {character_name}")
        print(f"book         : {book_title}")
        print(f"documents    : {len(documents)}")
        print(f"chunks       : {chunk_count}")

        if not args.yes:
            answer = input("Delete all of the above? Type 'yes' to confirm: ").strip().lower()
            if answer != "yes":
                print("Aborted — nothing was deleted.")
                return 1

        for doc in documents:
            session.delete(doc)

    print(f"Deleted {len(documents)} document(s) and {chunk_count} chunk(s) for character_id={args.character_id!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
