from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import AppSettingsDep, DbSession, DiacritizerDep, TTSDep
from app.db.models import Message
from app.schemas.tts import TTSRequest
from app.services.diacritization import strip_diacritics
from app.services.tts import TTSUpstreamAudioError, strip_citation_markers

logger = logging.getLogger("app.tts")

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])
import re

def strip_reference_numbers(text: str) -> str:
    return re.sub(r"\[[0-9٠-٩]+\]", "", text)

@router.post("/speak")
async def speak_endpoint(
    req: TTSRequest,
    session: DbSession,
    tts: TTSDep,
    diacritizer: DiacritizerDep,
    app_settings: AppSettingsDep,
) -> StreamingResponse:
    precomputed_diacritized: str | None = None
    if req.message_id is not None:
        message = session.get(Message, req.message_id)
        if message is None or message.role != "assistant":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown assistant message_id")
        text = message.content
        # Set for messages saved after this feature shipped whose answer
        # was grounded (app/api/routes/chat.py) — the exact text the chat
        # LLM generated, already fully diacritized, used verbatim ("text
        # without edits") rather than re-diacritizing the plain text with
        # a second LLM call. Absent for older messages and the
        # ungrounded-fallback answer (a fixed string, never diacritized
        # up front) — those fall through to the on-demand diacritizer
        # below exactly as before.
    #     if message.extra:
    #         precomputed_diacritized = message.extra.get("diacritized_content")
    # else:
    #     text = req.text or ""
    #
    # text = strip_citation_markers(text)
    # if not text:
    #     raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to synthesize")
    # if len(text) > app_settings.tts_max_text_length:
    #     raise HTTPException(
    #         status.HTTP_400_BAD_REQUEST, f"text too long (max {app_settings.tts_max_text_length} chars)"
    #     )
    #
    # if precomputed_diacritized:
    #     speech_text = strip_citation_markers(precomputed_diacritized)
    # else:
    #     # Diacritized purely to help SILMA's pronunciation — never what's
    #     # stored in `messages` or returned to the client (that's `text`
    #     # above, untouched). A failure here is a quality regression, not
    #     # a correctness one: fall back to the plain text rather than
    #     # blocking voice output over an enhancement (same posture as
    #     # chat's follow-up suggestions, which silently no-op on failure).
    #     try:
    #         speech_text = await diacritizer.diacritize(text)
    #     except Exception:
    #         logger.exception("diacritization failed; speaking undiacritized text")
    #         speech_text = text

    speech_text = strip_diacritics(req.text)
    speech_text = strip_reference_numbers(speech_text)

    # Synthesis happens entirely inside speak() before it returns (the
    # SageMaker call isn't truly incremental — see app/services/tts.py) —
    # so awaiting it here already fails fast on any config/upstream error,
    # with the real error mapped to the right status before a 200 starts.
    try:
        audio = await tts.speak(speech_text, voice_id=req.voice_id)
    except TTSUpstreamAudioError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    headers = {
        "X-Audio-Sample-Rate": str(audio.sample_rate),
        "X-Audio-Channels": str(audio.channels),
        "X-Audio-Sample-Format": audio.sample_format,
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(audio.chunks, media_type="application/octet-stream", headers=headers)
