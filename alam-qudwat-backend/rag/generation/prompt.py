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
# a short factual question deserves a short answer. Explicitly calls out
# generic "tell me about this person" requests (e.g. "حدثني عن هذه
# الشخصية") as belonging in the brief bucket, not the detailed-story one —
# "حدّثني" reads like a story request on its own, which was causing long
# answers even when the user hadn't actually asked for depth/detail yet.
RESPONSE_LENGTH_RULE = (
    "6. اجعل طول إجابتك مناسبًا لطبيعة السؤال، وليس طولًا ثابتًا افتراضيًا: "
    "إذا كان السؤال بسيطًا أو مباشرًا (كسؤال عن اسم أو تاريخ أو حدث محدد)، أو "
    "كان طلبًا عامًا فضفاضًا للتعريف بالشخصية دون تحديد جانب أو حدث معيّن "
    "(مثل \"حدثني عن هذه الشخصية\" أو \"من هو/من هي؟\" أو \"أخبرني عنها\")، "
    "أجب بإيجاز — فقرة أو فقرتان مختصرتان تقدّمان لمحة عامة فقط — دون "
    "إطالة. لا تنتقل إلى سرد مطوّل إلا إذا طلب المستخدم صراحةً وبوضوح مزيدًا "
    "من التفاصيل أو قصة كاملة أو سياقًا واسعًا (مثل \"أعطني تفاصيل أكثر\" أو "
    "\"احكِ لي القصة كاملة\")، أو سأل عن جانب أو حدث محدد يستدعي شرحًا أوسع. "
    "لا تضف تفاصيل أو مصادر إضافية لمجرد إطالة الإجابة."
)

# The exact sentence every assistant answer must end with. Appended
# programmatically (app/api/routes/chat.py), not left to the LLM to
# reproduce verbatim — "exactly" is safer guaranteed in code than via
# instruction-following. NO_OWN_CLOSING_RULE below tells the model not to
# invent its own version, so the final text never ends with two.
CLOSING_QUESTION = "ماذا تريد أن تعرف أيضاً عن هذه الشخصية العظيمة؟"

# Hand-diacritized once, independently of CLOSING_QUESTION above (never
# derived from it via strip_diacritics, and never stripped back down to
# it either — the two are just two fixed spellings of the same fixed
# sentence, one for TTS, one for display). Used only when appending the
# closing question to the diacritized accumulator app/api/routes/chat.py
# keeps for TTS; app/services/diacritization.strip_diacritics() is what
# turns CLOSING_QUESTION_DIACRITIZED-style text back into plain text for
# anything the LLM itself generates with diacritics.
CLOSING_QUESTION_DIACRITIZED = "مَاذَا تُرِيدُ أَنْ تَعْرِفَ أَيْضاً عَنْ هَذِهِ الشَّخْصِيَّةِ الْعَظِيمَةِ؟"

NO_OWN_CLOSING_RULE = (
    "7. لا تختم إجابتك بسؤال أو عبارة ختامية من عندك تدعو المستخدم للمتابعة "
    "(مثل \"هل تريد معرفة المزيد؟\") — سيُضاف سؤال ختامي موحّد تلقائيًا بعد "
    "إجابتك، فلا داعي لتكرار الفكرة أو تمهيد لها."
)

# Asks the SAME generation call to write fully-diacritized Arabic, so a
# later TTS request for this same answer never needs a second LLM call
# just to add tashkeel (app/api/routes/chat.py strips it right back out
# for what's actually shown to/stored for the user — see
# app/services/diacritization.strip_diacritics() — but keeps the raw
# diacritized text for TTS use). Applies to the main narrated answer
# only; suggestions are never spoken, so they're never diacritized.
DIACRITIZATION_RULE = (
    "8. اكتب كل إجابتك بالتشكيل العربي الكامل على كل كلمة (الفتحة والضمة "
    "والكسرة والسكون والشدة والتنوين)، لتحسين نطقها لاحقًا عند تحويلها إلى "
    "صوت. هذا مطلب صارم يشمل النص كله وليس اختياريًا لبعض الكلمات فقط، لكنه "
    "لا يغيّر طول الإجابة أو مضمونها أو الالتزام بالقواعد الأخرى أعلاه."
)

# Only appended for a conversation's very first assistant answer (see
# build_chat_messages' is_first_message) — later turns use RESPONSE_LENGTH_RULE's
# normal question-driven length instead.
FIRST_MESSAGE_LENGTH_RULE = (
    "ملاحظة مهمة: هذه أول إجابة لك في هذه المحادثة، لذا اجعلها موجزة قدر "
    "الإمكان — في حدود فقرتين على الأكثر — دون إخلال بالدقة التاريخية أو "
    "حذف الاستشهاد بالمصادر."
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
    + "\n"
    + NO_OWN_CLOSING_RULE
    + "\n"
    + DIACRITIZATION_RULE
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
    + "\n"
    + NO_OWN_CLOSING_RULE
    + "\n"
    + DIACRITIZATION_RULE
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
    is_first_message: bool = False,
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

    ``is_first_message`` appends FIRST_MESSAGE_LENGTH_RULE — true only for
    a conversation's very first assistant answer (the caller determines
    this from whether ``history`` contains any prior assistant turn).
    """
    system = narrator_system_prompt(mode, character_name)
    if is_first_message:
        system = system + "\n" + FIRST_MESSAGE_LENGTH_RULE
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


def _suggestions_system_prompt(max_suggestions: int) -> str:
    """max_suggestions is the number of *remaining* slots after predefined
    suggestions (app/services/suggestions.py) already filled some of the
    product's 3-suggestion cap — this call only ever needs to fill what's
    left, never the full cap itself."""
    return (
        "أنت مساعد يقترح أسئلة متابعة طبيعية بعد إجابة سردية عن شخصية تاريخية "
        "إسلامية، بالاعتماد فقط على المصادر المرفقة نفسها التي استُخدمت في "
        "الإجابة.\n"
        "قواعد:\n"
        f"1. اقترح من صفر إلى {max_suggestions} سؤال/أسئلة متابعة قصيرة "
        "بالعربية فقط عندما تكون هناك بالفعل جوانب أخرى مثيرة للاهتمام تغطيها "
        "المصادر المرفقة ولم تتم تغطيتها في الإجابة.\n"
        "2. إذا لم تكن هناك أسئلة متابعة طبيعية أو كافية (مثل أن الإجابة كانت "
        "شاملة، أو أن المصادر لا تغطي شيئًا إضافيًا مفيدًا)، أعد قائمة فارغة. لا "
        "تخترع أسئلة لمجرد ملء القائمة.\n"
        "3. لا تقترح سؤالًا تمت الإجابة عنه بالفعل ضمن الإجابة السردية.\n"
        "4. لا تقترح أي سؤال مذكور في قسم \"أسئلة سبق طرحها\" أدناه، ولا سؤالاً "
        "آخر مشابهًا له في المعنى — هذه الأسئلة استُخدمت بالفعل في هذه المحادثة "
        "ويجب ألا تظهر مجددًا بأي صياغة.\n"
        "5. أعد الناتج حصريًا بصيغة JSON على الشكل التالي، بدون أي نص إضافي:\n"
        '{"suggestions": ["سؤال أول؟", "سؤال ثانٍ؟"]}'
    )


def build_suggestions_prompt(
    question: str,
    answer_text: str,
    chunks: list[RetrievedChunk],
    *,
    mode: NarratorMode,
    character_name: str,
    already_asked: list[str] | None = None,
    max_suggestions: int = 3,
) -> list[dict[str, str]]:
    """Build messages for a small, non-streamed, JSON-mode LLM call that
    decides whether follow-up question suggestions are natural here — the
    model itself decides, per rule 2, rather than suggestions being
    heuristically always-on or always-off.

    ``already_asked`` — every question the user has already sent as a
    message in this conversation (predefined or not) — so the model never
    repeats something the user already asked (app/services/suggestions.py
    additionally deduplicates deterministically for the predefined/exact
    cases; this covers paraphrase-level repeats an exact-string check
    would miss). ``max_suggestions`` is the number of slots actually left
    to fill (see _suggestions_system_prompt).
    """
    context = format_context(chunks) if chunks else "(لا توجد مصادر)"
    audience = "طفل" if mode == "kids" else "بالغ"
    asked_block = "\n".join(f"- {q}" for q in already_asked) if already_asked else "(لا يوجد)"
    user_content = (
        f"الشخصية: {character_name}\nالجمهور: {audience}\n\n"
        f"سؤال المستخدم: {question}\n\nالإجابة السردية التي قُدّمت له:\n{answer_text}\n\n"
        f"المصادر المستخدمة في هذه الإجابة:\n{context}\n\n"
        f"أسئلة سبق طرحها في هذه المحادثة (لا تقترح أيًا منها ولا ما يماثلها):\n{asked_block}\n\n"
        "اقترح أسئلة متابعة وفق القواعد أعلاه."
    )
    return [
        {"role": "system", "content": _suggestions_system_prompt(max(max_suggestions, 0))},
        {"role": "user", "content": user_content},
    ]
