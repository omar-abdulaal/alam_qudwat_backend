"""Semantic-ish, token-bounded chunking for Arabic historical text.

Strategy: split cleaned page text into sentence/paragraph-level segments,
then greedily pack consecutive segments into chunks up to
``chunk_token_size`` tokens (measured with tiktoken's cl100k_base
encoding, used purely as a length proxy — no OpenAI call involved), with
a sliding overlap of ``chunk_token_overlap`` tokens carried into the next
chunk so retrieval doesn't lose context at a chunk boundary.

This only ever *splits* the cleaned text at existing boundaries (sentence
terminators, newlines) — no word is added, removed, or reordered.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import tiktoken

_SENTENCE_SPLIT = re.compile(r"(?<=[.!؟])\s+")
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int


def _segments(text: str) -> list[str]:
    segments: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                segments.append(sentence)
    return segments


def _token_len(s: str) -> int:
    return len(_ENCODING.encode(s))


def _hard_split_long_segment(segment: str, max_tokens: int) -> list[str]:
    """Fallback for a single sentence longer than max_tokens: split on raw
    token boundaries so no chunk ever exceeds the configured size."""
    tokens = _ENCODING.encode(segment)
    pieces = []
    for start in range(0, len(tokens), max_tokens):
        pieces.append(_ENCODING.decode(tokens[start : start + max_tokens]))
    return pieces


def chunk_text(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[Chunk]:
    """Chunk cleaned text into token-bounded pieces with sliding overlap."""
    if not text.strip():
        return []

    raw_segments = _segments(text)
    segments: list[str] = []
    for seg in raw_segments:
        if _token_len(seg) > max_tokens:
            segments.extend(_hard_split_long_segment(seg, max_tokens))
        else:
            segments.append(seg)

    if not segments:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> list[str]:
        """Flush `current` into a chunk; return the tail segments to carry
        forward as overlap for the next chunk."""
        nonlocal current, current_tokens
        chunk_text_value = " ".join(current)
        chunks.append(Chunk(index=len(chunks), text=chunk_text_value, token_count=current_tokens))

        # Build overlap tail from the end of `current` totaling ~overlap_tokens.
        tail: list[str] = []
        tail_tokens = 0
        for seg in reversed(current):
            seg_tokens = _token_len(seg)
            if tail_tokens + seg_tokens > overlap_tokens and tail:
                break
            tail.insert(0, seg)
            tail_tokens += seg_tokens
        current = tail
        current_tokens = tail_tokens
        return tail

    for seg in segments:
        seg_tokens = _token_len(seg)
        if current and current_tokens + seg_tokens > max_tokens:
            flush()
        current.append(seg)
        current_tokens += seg_tokens

    if current:
        # Avoid a final orphan chunk that's too small — merge it back into
        # the previous chunk when possible instead of dropping content.
        if chunks and current_tokens < min_tokens:
            merged_text = chunks[-1].text + " " + " ".join(current)
            merged_tokens = _token_len(merged_text)
            chunks[-1] = Chunk(index=chunks[-1].index, text=merged_text, token_count=merged_tokens)
        else:
            chunks.append(Chunk(index=len(chunks), text=" ".join(current), token_count=current_tokens))

    return chunks
