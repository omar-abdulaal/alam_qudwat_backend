"""Lossless cleanup of scraper output.

The scraper extracts raw HTML text from Shamela, which leaves behind
structural bracket markers around section headings (e.g. ``[[heading]``,
``]``, ``[-``) that are artifacts of the page's HTML structure, not part
of the historical source text. This module strips *only* those markers
and collapses incidental whitespace — it never touches, reorders, or
rewrites a single Arabic word of the actual content.
"""
from __future__ import annotations

import re

# A line that consists *only* of bracket/dash artifacts, e.g. "]", "[-", "[[".
_BRACKET_ONLY_LINE = re.compile(r"^\[+-?$|^\]+$")
_LEADING_BRACKETS = re.compile(r"^\[+-?\s*")
_TRAILING_BRACKETS = re.compile(r"\s*\]+$")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip HTML-extraction bracket artifacts; preserve all real content."""
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _BRACKET_ONLY_LINE.match(stripped):
            continue
        stripped = _LEADING_BRACKETS.sub("", stripped)
        stripped = _TRAILING_BRACKETS.sub("", stripped)
        if stripped:
            cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines)
    result = _MULTI_BLANK_LINES.sub("\n\n", result)
    return result.strip()
