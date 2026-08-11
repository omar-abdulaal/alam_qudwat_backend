#!/usr/bin/env python3
"""
Split companions_tier1_pages.jsonl into one persistent biography dataset per
Shamela entry/character, without re-scraping the website.

Default input:
    data/raw/companions_tier1/companions_tier1_pages.jsonl

Default output:
    data/raw/companions_tier1/characters/

Each detected biography gets its own folder:
    <first_page>_<readable_name>/
        metadata.json
        pages.jsonl
        raw.txt

Additionally creates:
    companions_biographies.jsonl   # one aggregated JSON record per biography
    characters_manifest.json       # summary/index
    unassigned_pages.jsonl         # only if some pages cannot be assigned safely

Run:
    python split_companions.py

Custom input/output:
    python split_companions.py path/to/companions_tier1_pages.jsonl \
        --output path/to/characters
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/raw/companions_tier1/companions_tier1_pages.jsonl")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_filename(value: str, max_length: int = 72) -> str:
    """Keep Arabic readable while removing path-hostile punctuation."""
    value = value.strip()
    value = re.sub(r"[\\/:*?\"<>|]+", " ", value)
    value = re.sub(r"[^\w\-\u0600-\u06FF ]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value).strip("_.-")
    if not value:
        value = "unknown_character"
    return value[:max_length].rstrip("_.-")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc

            if "page_id" not in record or "text" not in record:
                raise ValueError(
                    f"Line {line_number} is missing page_id or text in {path}"
                )
            records.append(record)

    records.sort(key=lambda item: int(item["page_id"]))
    return records


def nearest_known_character(
    records: list[dict[str, Any]], index: int, direction: int
) -> tuple[str | None, str | None]:
    i = index + direction
    while 0 <= i < len(records):
        char_id = records[i].get("character_id")
        char_name = records[i].get("character_name") or records[i].get("entry_name")
        if char_id and char_name:
            return str(char_id), str(char_name)
        i += direction
    return None, None


def resolve_character_context(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Use explicit character metadata when present.

    For a page missing character metadata, inherit only when the nearest known
    character before AND after the page are the same. This avoids silently
    attaching section/title pages to the wrong biography.
    """
    resolved: list[dict[str, Any]] = []

    for index, original in enumerate(records):
        record = dict(original)
        char_id = record.get("character_id")
        char_name = record.get("character_name") or record.get("entry_name")

        if char_id and char_name:
            record["character_name"] = char_name
            record["character_assignment"] = "source"
            resolved.append(record)
            continue

        prev_id, prev_name = nearest_known_character(records, index, -1)
        next_id, next_name = nearest_known_character(records, index, +1)

        if prev_id and next_id and prev_id == next_id:
            record["character_id"] = prev_id
            record["character_name"] = prev_name or next_name
            record["character_assignment"] = "inferred_between_same_character"
        else:
            record["character_assignment"] = "unassigned"

        resolved.append(record)

    return resolved


def split_contiguous_biographies(
    records: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Split by contiguous character_id, not merely by name globally.

    This prevents two separate entries with the same display name from being
    accidentally merged if that ever occurs in the source.
    """
    biographies: list[list[dict[str, Any]]] = []
    unassigned: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_char_id: str | None = None

    def flush() -> None:
        nonlocal current, current_char_id
        if current:
            biographies.append(current)
        current = []
        current_char_id = None

    for record in records:
        char_id = record.get("character_id")
        char_name = record.get("character_name")

        if not char_id or not char_name:
            flush()
            unassigned.append(record)
            continue

        char_id = str(char_id)
        if current and char_id != current_char_id:
            flush()

        if not current:
            current_char_id = char_id

        current.append(record)

    flush()
    return biographies, unassigned


def aggregate_biography(records: list[dict[str, Any]], sequence: int) -> dict[str, Any]:
    first = records[0]
    last = records[-1]
    character_id = str(first["character_id"])
    character_name = str(first["character_name"])
    first_page_id = int(first["page_id"])
    last_page_id = int(last["page_id"])

    texts = [str(record.get("text", "")).strip() for record in records]
    combined_text = "\n\n".join(text for text in texts if text)

    source_page_ids = [int(record["page_id"]) for record in records]
    source_urls = [record.get("url") for record in records if record.get("url")]
    printed_pages = [
        record.get("printed_page")
        for record in records
        if record.get("printed_page") is not None
    ]
    printed_volumes = [
        record.get("printed_volume")
        for record in records
        if record.get("printed_volume") is not None
    ]

    biography_id = f"{character_id}_{first_page_id}"

    return {
        "biography_id": biography_id,
        "sequence": sequence,
        "character_id": character_id,
        "character_name": character_name,
        "book_id": first.get("book_id"),
        "book_title": first.get("book_title"),
        "author": first.get("author"),
        "dataset_id": first.get("dataset_id"),
        "dataset_name": first.get("dataset_name"),
        "layer": first.get("layer"),
        "source_group": first.get("source_group"),
        "person_category": first.get("person_category"),
        "source_section": first.get("source_section"),
        "page_count": len(records),
        "page_id_start": first_page_id,
        "page_id_end": last_page_id,
        "source_page_ids": source_page_ids,
        "source_urls": source_urls,
        "printed_volume_start": printed_volumes[0] if printed_volumes else None,
        "printed_volume_end": printed_volumes[-1] if printed_volumes else None,
        "printed_page_start": printed_pages[0] if printed_pages else None,
        "printed_page_end": printed_pages[-1] if printed_pages else None,
        "text": combined_text,
        "content_hash": stable_hash(combined_text),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_biography_folder(
    output_root: Path,
    biography: dict[str, Any],
    page_records: list[dict[str, Any]],
) -> Path:
    folder_name = (
        f"{int(biography['page_id_start']):04d}_"
        f"{safe_filename(str(biography['character_name']))}"
    )
    folder = output_root / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    enriched_pages: list[dict[str, Any]] = []
    for record in page_records:
        item = dict(record)
        item["biography_id"] = biography["biography_id"]
        enriched_pages.append(item)

    write_jsonl(folder / "pages.jsonl", enriched_pages)

    raw_parts: list[str] = []
    for record in enriched_pages:
        header = (
            f"===== page_id={record['page_id']}"
            f" | volume={record.get('printed_volume')}"
            f" | page={record.get('printed_page')} ====="
        )
        raw_parts.append(header + "\n\n" + str(record["text"]).strip())
    (folder / "raw.txt").write_text("\n\n".join(raw_parts) + "\n", encoding="utf-8")

    metadata = {key: value for key, value in biography.items() if key != "text"}
    metadata["folder"] = folder.name
    write_json(folder / "metadata.json", metadata)
    return folder


def split_file(input_path: Path, output_root: Path) -> dict[str, Any]:
    records = read_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    resolved = resolve_character_context(records)
    biography_groups, unassigned = split_contiguous_biographies(resolved)

    output_root.mkdir(parents=True, exist_ok=True)

    # Remove stale generated biography folders from earlier runs, while keeping
    # top-level files. This makes repeated execution deterministic.
    for child in output_root.iterdir():
        if child.is_dir() and re.match(r"^\d{4,}_", child.name):
            for nested in child.iterdir():
                if nested.is_file():
                    nested.unlink()
            try:
                child.rmdir()
            except OSError:
                pass

    biographies: list[dict[str, Any]] = []
    folder_index: list[dict[str, Any]] = []

    for sequence, group in enumerate(biography_groups, start=1):
        biography = aggregate_biography(group, sequence)
        folder = write_biography_folder(output_root, biography, group)
        biographies.append(biography)
        folder_index.append(
            {
                "sequence": sequence,
                "biography_id": biography["biography_id"],
                "character_id": biography["character_id"],
                "character_name": biography["character_name"],
                "folder": folder.name,
                "page_id_start": biography["page_id_start"],
                "page_id_end": biography["page_id_end"],
                "page_count": biography["page_count"],
            }
        )

    write_jsonl(output_root / "companions_biographies.jsonl", biographies)

    if unassigned:
        write_jsonl(output_root / "unassigned_pages.jsonl", unassigned)
    else:
        unassigned_path = output_root / "unassigned_pages.jsonl"
        if unassigned_path.exists():
            unassigned_path.unlink()

    assignment_counts = Counter(record.get("character_assignment") for record in resolved)
    manifest = {
        "source_file": str(input_path),
        "total_source_pages": len(records),
        "total_biographies": len(biographies),
        "assigned_pages": len(records) - len(unassigned),
        "unassigned_pages": len(unassigned),
        "assignment_counts": dict(assignment_counts),
        "biographies": folder_index,
    }
    write_json(output_root / "characters_manifest.json", manifest)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split the companions page-level JSONL into one folder/file set per biography."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"Input JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        help="Output directory (default: <input-folder>/characters)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output) if args.output else input_path.parent / "characters"

    try:
        manifest = split_file(input_path, output_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.")
    print(f"Source pages      : {manifest['total_source_pages']}")
    print(f"Biographies       : {manifest['total_biographies']}")
    print(f"Assigned pages    : {manifest['assigned_pages']}")
    print(f"Unassigned pages  : {manifest['unassigned_pages']}")
    print(f"Output            : {output_root}")
    print(f"Aggregated JSONL  : {output_root / 'companions_biographies.jsonl'}")
    if manifest["unassigned_pages"]:
        print(
            "WARNING: Some pages could not be safely assigned. Review: "
            f"{output_root / 'unassigned_pages.jsonl'}"
        )


if __name__ == "__main__":
    main()
