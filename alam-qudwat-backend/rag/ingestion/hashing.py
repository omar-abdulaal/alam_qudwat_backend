"""Content hashing used to make ingestion idempotent.

A stable sha256 of the exact text is stored alongside every document and
chunk. Re-running ingestion recomputes these hashes and only touches rows
whose hash changed (or that are new) — unchanged pages/chunks are skipped
entirely, including the embedding API call.
"""
from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
