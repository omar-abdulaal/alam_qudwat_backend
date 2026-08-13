"""Arabic diacritization (tashkeel) — TTS pronunciation aid only.

SILMA's pronunciation quality depends heavily on tashkeel: undiacritized
Arabic is genuinely ambiguous to a TTS model in ways it usually isn't to
a human reader (the same bare letters can be multiple different words).

The main chat-generation call itself now writes its answer already fully
diacritized (rag.generation.prompt.DIACRITIZATION_RULE) — one LLM call
does double duty instead of a second one re-diacritizing the same answer
afterwards (that used to add a full extra round-trip of latency to every
"play audio" request). app/api/routes/chat.py keeps that raw diacritized
text (never shown to the user or stored as `messages.content` — see
strip_diacritics() below) alongside the plain text in `messages.extra`,
and app/api/routes/tts.py uses it directly, unmodified, when synthesizing
an existing assistant message.

OpenAITextDiacritizer below still exists for the cases with no
pre-diacritized text available: synthesizing arbitrary `text` (not an
existing message — nothing to have pre-diacritized), and the ungrounded-
fallback answer (a fixed string, not LLM-generated, so never diacritized
up front). If diacritization fails there, the caller falls back to the
plain text rather than blocking voice output — this is a pronunciation
enhancement, not a correctness requirement.
"""
from __future__ import annotations

import re
from typing import Protocol

from openai import AsyncOpenAI

from rag.config import get_settings

# Standard Arabic combining-diacritic codepoint ranges: U+0610-U+061A
# (Quranic annotation/honorific signs), U+064B-U+065F (tanween, harakat,
# shadda, sukun and related marks), U+0670 (superscript alef), U+06D6-
# U+06ED (further Quranic annotation marks). Removing these turns
# diacritized text back into what the user is shown/what gets stored —
# see rag.generation.prompt.DIACRITIZATION_RULE, the only source of
# diacritics this needs to undo (CLOSING_QUESTION_DIACRITIZED is never
# round-tripped through this; it and CLOSING_QUESTION are independent
# fixed strings, not derived from one another).
_ARABIC_DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")


def strip_diacritics(text: str) -> str:
    return _ARABIC_DIACRITICS_RE.sub("", text)


_SYSTEM_PROMPT = (
    "أنت أداة تشكيل نصوص عربية دقيقة. ستتلقى نصًا عربيًا غير مشكَّل، ومهمتك "
    "إعادة كتابته حرفيًا مع إضافة علامات التشكيل الكاملة (الفتحة، الضمة، "
    "الكسرة، السكون، الشدة، التنوين) لتحسين نطقه بواسطة نظام تحويل نص إلى "
    "كلام.\n"
    "قواعد صارمة:\n"
    "1. لا تُغيّر أي كلمة، ولا تحذف أو تضف أو تعيد ترتيب أي كلمة أو حرف أو "
    "علامة ترقيم — أضف التشكيل فقط.\n"
    "2. لا تُترجم ولا تُفسّر ولا تُضف أي نص جديد؛ أعد النص نفسه حرفيًا مع "
    "التشكيل فقط.\n"
    "3. الأرقام والرموز وأي نص غير عربي (إن وُجد) تُعاد كما هي دون أي تغيير.\n"
    "4. أعد النص المُشكَّل فقط، بدون أي مقدمة أو تعليق إضافي."
)


class TextDiacritizer(Protocol):
    async def diacritize(self, text: str) -> str:
        """Return `text` rewritten with full Arabic tashkeel, for TTS use
        only — the wording/word order must be unchanged."""
        ...


class OpenAITextDiacritizer:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.diacritization_model
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file — see .env.example."
            )
        self._client = AsyncOpenAI(api_key=key)

    async def diacritize(self, text: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        diacritized = response.choices[0].message.content
        return diacritized.strip() if diacritized else text
