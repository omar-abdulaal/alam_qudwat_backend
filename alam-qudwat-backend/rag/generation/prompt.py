"""Prompt construction for grounded LLM answers.

This module does not call any LLM itself — it only builds prompts and
citation lists from retrieved chunks, so it can be wired into whichever
chat/completions call needs it (see app/services/llm.py for the actual
call) without duplicating the grounding rules in multiple places.

Two prompt shapes live here:

- ``build_prompt`` / ``SYSTEM_PROMPT`` — the original single-turn Q&A
  shape (system + user string pair). Kept as-is, still usable standalone.
- ``build_chat_messages`` / ``narrator_system_prompt`` — the multi-turn,
  narrator-mode-aware shape used by the chat API, reusing the same
  grounding rules plus persona/tone instructions per audience.
"""
from __future__ import annotations

from typing import Literal

from rag.retrieval.retriever import RetrievedChunk

# The non-negotiable grounding rules, shared by every prompt shape below.
GROUNDING_RULES = (
    "القواعد الملزمة:\n"
    "1. لا تستخدم أي معلومة لا توجد نصًا في المصادر المرفقة، ولا تخترع أسماء "
    "أو تواريخ أو أحداثًا أو إسنادات غير موجودة في النص المرفق.\n"
    "2. إذا كانت المصادر المرفقة لا تحتوي على إجابة كافية للسؤال، صرّح بذلك "
    "بوضوح بدلاً من التخمين.\n"
    "3. اذكر مصدر كل معلومة تستخدمها بالإشارة إلى رقم المقطع [١]، [٢]... "
    "كما وردت في قائمة المصادر.\n"
    "4. لا تُعدّل النص التاريخي المقتبس ولا تُعيد صياغته؛ استشهد به كما ورد."
)

# Applies specifically to the character-narrator chat persona (not the
# generic single-turn Q&A shape) — the model must always speak *about* the
# historical figure, in third person, never impersonate them.
NEVER_IMPERSONATE_RULE = (
    "5. لا تتحدث أبدًا بصيغة المتكلم بصفتك الشخصية التاريخية نفسها، ولا تقل "
    "\"أنا فعلت كذا\" أو ما شابه منسوبًا للشخصية. تحدث دائمًا كراوٍ ومؤرّخ يروي "
    "ويحلل سيرة هذه الشخصية بضمير الغائب."
)

# Response length must track the question, not default to maximal detail —
# a short factual question deserves a short answer.
RESPONSE_LENGTH_RULE = (
    "6. اجعل طول إجابتك مناسبًا لطبيعة السؤال، وليس طولًا ثابتًا افتراضيًا: "
    "إذا كان السؤال بسيطًا أو مباشرًا (كسؤال عن اسم أو تاريخ أو حدث محدد)، "
    "أجب في بضع جمل مختصرة فقط دون إطالة. أما إذا طلب المستخدم صراحةً "
    "قصة أو شرحًا مفصلاً أو سياقًا واسعًا، فيمكنك عندها أن تسرد بإسهاب أكبر. "
    "لا تضف تفاصيل أو مصادر إضافية لمجرد إطالة الإجابة."
)

SYSTEM_PROMPT = (
    "أنت مساعد بحثي متخصص في التاريخ الإسلامي، ومهمتك الإجابة عن الأسئلة "
    "اعتمادًا حصريًا على المقاطع المصدرية المرفقة أدناه.\n" + GROUNDING_RULES
)

_KIDS_PERSONA = (
    "أنت راوٍ قصص ودود يروي للأطفال سيرة {character_name} من التاريخ الإسلامي، "
    "اعتمادًا حصريًا على المصادر التاريخية المرفقة.\n"
    "أسلوبك: لغة عربية فصحى مبسطة وواضحة، جمل قصيرة، نبرة دافئة ومشجعة، "
    "تركّز على القيم والعبر (كالصدق والشجاعة والعدل والوفاء) أكثر من التفاصيل "
    "السياسية أو الخلافات المعقدة أو المشاهد العنيفة. اجعل القصة ملموسة وسهلة "
    "الفهم لطفل، واختم عند المناسب بعبرة بسيطة يمكن للطفل تذكّرها.\n"
    + GROUNDING_RULES
    + "\n"
    + NEVER_IMPERSONATE_RULE
    + "\n"
    + RESPONSE_LENGTH_RULE
)

_ADULTS_PERSONA = (
    "أنت راوٍ ومؤرخ متخصص يقدّم سيرة {character_name} من التاريخ الإسلامي "
    "لجمهور من البالغين، اعتمادًا حصريًا على المصادر التاريخية المرفقة.\n"
    "أسلوبك: لغة عربية فصحى راقية، مع سياق تاريخي وسياسي واجتماعي عند الحاجة، "
    "وتحليل متوازن للأحداث والخلافات كما وردت في المصادر دون تحيّز أو تبسيط "
    "مخل، مع الإشارة إلى تعدد الروايات إن وُجد.\n"
    + GROUNDING_RULES
    + "\n"
    + NEVER_IMPERSONATE_RULE
    + "\n"
    + RESPONSE_LENGTH_RULE
)

NarratorMode = Literal["kids", "adults"]


def narrator_system_prompt(mode: NarratorMode, character_name: str) -> str:
    """System prompt for the character-narrator chat persona, tailored to
    the audience. Both modes share the same grounding + never-impersonate
    rules; only tone/vocabulary/focus differ."""
    template = _KIDS_PERSONA if mode == "kids" else _ADULTS_PERSONA
    return template.format(character_name=character_name)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered source list for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] الكتاب: {c.book_title} | المؤلف: {c.author} | الشخصية: {c.caliph_name} "
            f"| الصفحة: {c.printed_page or c.page_id} | الرابط: {c.source_url}\n{c.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> dict[str, str]:
    """Return {"system": ..., "user": ...} ready to pass to an LLM's
    chat/completions call. Grounding is enforced by SYSTEM_PROMPT plus
    restricting the "user" content to only the retrieved sources."""
    context = format_context(chunks)
    user_prompt = (
        f"السؤال: {question}\n\nالمصادر المتاحة:\n{context}\n\n"
        "أجب عن السؤال أعلاه معتمدًا فقط على المصادر المذكورة، مع ذكر أرقام "
        "المصادر [رقم] بجانب كل معلومة."
    )
    return {"system": SYSTEM_PROMPT, "user": user_prompt}


def citation_list(chunks: list[RetrievedChunk]) -> list[str]:
    return [f"[{i}] {c.citation()}" for i, c in enumerate(chunks, start=1)]


def build_chat_messages(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    mode: NarratorMode,
    character_name: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build a full OpenAI-style ``messages`` list for the character-narrator
    chat persona: system prompt (mode + character specific) + prior turns
    (already-stored conversation history, each ``{"role": "user"|"assistant",
    "content": ...}``) + the current, grounded user turn.

    Reuses ``format_context``/``narrator_system_prompt`` rather than
    re-implementing source formatting or grounding rules. Prior turns are
    passed through verbatim — only the *current* question is attached to
    the retrieved-sources block, keeping token usage bounded regardless of
    conversation length.
    """
    system = narrator_system_prompt(mode, character_name)
    context = format_context(chunks) if chunks else "(لا توجد مصادر مسترجعة تخص هذا السؤال)"
    user_content = (
        f"سؤال المستخدم: {question}\n\nالمصادر المتاحة للإجابة عن هذا السؤال تحديدًا:\n{context}\n\n"
        "أجب بالاعتماد فقط على هذه المصادر، مع ذكر أرقام المصادر [رقم] بجانب كل معلومة تستخدمها."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


_SUGGESTIONS_SYSTEM_PROMPT = (
    "أنت مساعد يقترح أسئلة متابعة طبيعية بعد إجابة سردية عن شخصية تاريخية "
    "إسلامية، بالاعتماد فقط على المصادر المرفقة نفسها التي استُخدمت في "
    "الإجابة.\n"
    "قواعد:\n"
    "1. اقترح من صفر إلى ثلاثة أسئلة متابعة قصيرة بالعربية فقط عندما تكون "
    "هناك بالفعل جوانب أخرى مثيرة للاهتمام تغطيها المصادر المرفقة ولم تتم "
    "تغطيتها في الإجابة.\n"
    "2. إذا لم تكن هناك أسئلة متابعة طبيعية أو كافية (مثل أن الإجابة كانت "
    "شاملة، أو أن المصادر لا تغطي شيئًا إضافيًا مفيدًا)، أعد قائمة فارغة. لا "
    "تخترع أسئلة لمجرد ملء القائمة.\n"
    "3. لا تقترح سؤالًا تمت الإجابة عنه بالفعل ضمن الإجابة السردية.\n"
    "4. أعد الناتج حصريًا بصيغة JSON على الشكل التالي، بدون أي نص إضافي:\n"
    '{"suggestions": ["سؤال أول؟", "سؤال ثانٍ؟"]}'
)


def build_suggestions_prompt(
    question: str,
    answer_text: str,
    chunks: list[RetrievedChunk],
    *,
    mode: NarratorMode,
    character_name: str,
) -> list[dict[str, str]]:
    """Build messages for a small, non-streamed, JSON-mode LLM call that
    decides whether follow-up question suggestions are natural here — the
    model itself decides, per _SUGGESTIONS_SYSTEM_PROMPT's rule 2, rather
    than suggestions being heuristically always-on or always-off."""
    context = format_context(chunks) if chunks else "(لا توجد مصادر)"
    audience = "طفل" if mode == "kids" else "بالغ"
    user_content = (
        f"الشخصية: {character_name}\nالجمهور: {audience}\n\n"
        f"سؤال المستخدم: {question}\n\nالإجابة السردية التي قُدّمت له:\n{answer_text}\n\n"
        f"المصادر المستخدمة في هذه الإجابة:\n{context}\n\n"
        "اقترح أسئلة متابعة وفق القواعد أعلاه."
    )
    return [
        {"role": "system", "content": _SUGGESTIONS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
