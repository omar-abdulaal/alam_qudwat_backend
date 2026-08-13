"""Predefined, deterministic follow-up suggestions, plus the "already
used" filter shared by the chat flow (app/services/chat_service.py).

Suggested questions are capped at MAX_SUGGESTIONS (3): up to 2 predefined
here — chosen by NarratorMode alone (never raw age — the Flutter app
already converts age to "kids"/"adults", which is the only signal used)
and, for adults, by the character's role categories — plus, only when
slots remain, LLM-generated ones (rag.generation.prompt.
build_suggestions_prompt, called from chat_service.generate_suggestions).

Predefined suggestions cost no LLM call, so they can be shown even on the
ungrounded-fallback path in principle; chat_service currently only calls
this for grounded turns — see that module's docstring.
"""
from __future__ import annotations

MAX_SUGGESTIONS = 3

KIDS_PREDEFINED_SUGGESTIONS: tuple[str, ...] = (
    "كيف يمكنني أن أقلد أخلاق هذه الشخصية في حياتي اليومية؟",
    "ما هو أفضل موقف أو حكمة نتعلمها من بطل قصتنا اليوم؟",
)

ADULTS_FIRST_SUGGESTION = "ما هي أبرز الدروس المستفادة من سيرة هذه الشخصية لتطبيقها في واقعنا اليوم؟"

# One fixed, deterministic question per role — configuration, never
# LLM-generated. Must cover every code in the controlled taxonomy seeded
# by alembic/versions/0007_character_role_categories.py and used by
# scripts/generate_character_classification.py's CATEGORY_TAXONOMY; keep
# both lists in sync if the taxonomy ever changes.
#
# Dict insertion order doubles as the priority order used below to pick
# one question for a character with multiple roles: the first key (in
# this order) present in the character's categories wins. This keeps
# selection deterministic and the whole mapping maintainable in one place
# — reordering these lines is the only thing needed to change priority.
ADULTS_ROLE_SUGGESTIONS: dict[str, str] = {
    "خليفة": "كيف تعامل مع تحديات الحكم وإدارة شؤون الدولة؟",
    "أمير": "كيف تعامل مع تحديات الإمارة والمسؤولية عن قومه؟",
    "والي": "كيف تعامل مع تحديات إدارة الولاية وشؤون أهلها؟",
    "قائد عسكري": "كيف تعامل مع تحديات المعارك والقيادة؟",
    "فارس": "كيف واجه تحديات القتال والفروسية في معاركه؟",
    "فقيه": "كيف تعامل مع التحديات في اجتهاده الفقهي وفتاواه؟",
    "محدث": "كيف واجه التحديات في حفظ الحديث وروايته؟",
    "مفسر": "كيف واجه التحديات في تفسيره لكتاب الله؟",
    "مقرئ": "كيف واجه التحديات في تعلّم القرآن وتعليمه؟",
    "قاضٍ": "كيف تعامل مع تحديات القضاء وإصدار الأحكام العادلة؟",
    "عالم": "كيف واجه الصعوبات في رحلته لطلب العلم؟",
    "داعية": "كيف واجه التحديات في دعوته إلى الله؟",
    "معلّم": "كيف تعامل مع تحديات التعليم ونقل العلم لطلابه؟",
    "كاتب": "كيف تعامل مع تحديات الكتابة والتدوين في عصره؟",
    "شاعر": "كيف تعامل مع تحديات الإبداع الشعري في حياته؟",
    "أديب": "كيف تعامل مع تحديات الأدب والبيان في عصره؟",
    "نسابة": "كيف واجه التحديات في توثيق الأنساب ودراستها؟",
    "طبيب": "كيف تعامل مع تحديات مهنة الطب في عصره؟",
    "تاجر": "كيف أدار تحدياته المالية والاجتماعية؟",
    "راوية": "كيف واجه التحديات في رواية الأخبار والأشعار ونقلها؟",
}


def _role_specific_adult_suggestion(categories: list[str]) -> str | None:
    """Deterministic pick for a character with multiple categories: the
    first role in ADULTS_ROLE_SUGGESTIONS' (priority) order that the
    character actually has. None if the character has no matching role
    (e.g. not yet classified) — the caller then falls back to just the
    one universal adult suggestion, leaving more slots for the LLM."""
    character_roles = set(categories)
    for role, suggestion in ADULTS_ROLE_SUGGESTIONS.items():
        if role in character_roles:
            return suggestion
    return None


def predefined_suggestions(mode: str, categories: list[str]) -> list[str]:
    """Up to 2 deterministic suggestions for this turn, no LLM call.
    `mode` must be "kids" or "adults" (NarratorMode) — never derived from
    age directly."""
    if mode == "kids":
        return list(KIDS_PREDEFINED_SUGGESTIONS)

    suggestions = [ADULTS_FIRST_SUGGESTION]
    role_suggestion = _role_specific_adult_suggestion(categories)
    if role_suggestion is not None:
        suggestions.append(role_suggestion)
    return suggestions


def filter_unused(suggestions: list[str], already_asked: set[str]) -> list[str]:
    """Drop any suggestion whose exact text the user has already sent as
    a message in this conversation — it must never be offered again,
    whether predefined or previously LLM-generated."""
    return [s for s in suggestions if s not in already_asked]
