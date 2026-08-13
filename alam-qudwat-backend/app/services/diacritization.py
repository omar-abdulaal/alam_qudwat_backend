"""Arabic diacritics (tashkeel) stripping — TTS pronunciation support.

The main chat-generation call writes its answer already fully diacritized
(rag.generation.prompt.DIACRITIZATION_RULE) — that's the only source of
diacritics in this system. There is deliberately no LLM-based diacritizer
here: text that has no diacritics is simply spoken without them rather
than being sent through a model to add them, even as a fallback. This
keeps TTS latency-predictable (no extra LLM round-trip ever) and avoids a
second model call reproducing content the first one already generated.

app/api/routes/chat.py strips diacritics back out before anything is
shown to/stored for the user (see strip_diacritics() below) but keeps the
raw diacritized text in `messages.extra["diacritized_content"]` for
app/api/routes/tts.py to use verbatim.
"""
from __future__ import annotations

import re

# Standard Arabic combining-diacritic codepoint ranges: U+0610-U+061A
# (Quranic annotation/honorific signs), U+064B-U+065F (tanween, harakat,
# shadda, sukun and related marks), U+0670 (superscript alef), U+06D6-
# U+06ED (further Quranic annotation marks).
#
# Built from explicit (start, end) integer codepoints via chr() rather
# than a literal-character regex string — a prior literal-character
# version silently paired range boundaries in the wrong order (a
# character class pairs "X-Y" by *string position*, not by whichever
# ranges the author had in mind) and ended up stripping real base letters
# like ه/ذ/ا along with the actual diacritics. This form can't have that
# failure mode: each tuple is one unambiguous range.
_DIACRITIC_RANGES = [
    (0x0610, 0x061A),
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06ED),
]
_ARABIC_DIACRITICS_RE = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _DIACRITIC_RANGES) + "]")


def strip_diacritics(text: str) -> str:
    return _ARABIC_DIACRITICS_RE.sub("", text)
