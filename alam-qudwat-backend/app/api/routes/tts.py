from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import AppSettingsDep, DbSession, LiveGenerationRegistryDep, TTSDep
from app.db.models import Message
from app.schemas.tts import TTSLiveRequest, TTSRequest
from app.services.tts import TTSAudio, TTSUpstreamAudioError, strip_citation_markers

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


def _audio_headers(audio: TTSAudio) -> dict[str, str]:
    return {
        "X-Audio-Sample-Rate": str(audio.sample_rate),
        "X-Audio-Channels": str(audio.channels),
        "X-Audio-Sample-Format": audio.sample_format,
        "Cache-Control": "no-cache",
    }


async def _single_chunk_stream(text: str) -> AsyncIterator[str]:
    """Wraps an already-complete string as the one-shot text stream
    TextToSpeech.speak() expects — SilmaSageMakerTTS still splits it into
    sentence-sized segments and starts streaming the first one's audio
    before later segments are even synthesized (see app/services/tts.py).
    """
    yield text


@router.post("/speak")
async def speak_endpoint(
    req: TTSRequest,
    session: DbSession,
    tts: TTSDep,
    app_settings: AppSettingsDep,
) -> StreamingResponse:
    if req.message_id is not None:
        message = session.get(Message, req.message_id)
        if message is None or message.role != "assistant":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown assistant message_id")
        # The exact text the chat LLM generated, already fully diacritized
        # (rag/generation/prompt.py DIACRITIZATION_RULE), stored verbatim
        # in `extra` by app/api/routes/chat.py. Used as-is when present —
        # there is no LLM-based diacritizer here, not even as a fallback:
        # a message with no stored diacritized version (older messages,
        # the ungrounded-fallback answer) is simply spoken without
        # diacritics, which is fine.
        speech_text = (message.extra or {}).get("diacritized_content") or message.content
    else:
        speech_text = req.text or ""

    speech_text = strip_citation_markers(speech_text)
    if not speech_text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to synthesize")
    if len(speech_text) > app_settings.tts_max_text_length:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"text too long (max {app_settings.tts_max_text_length} chars)"
        )

    # Synthesis of the first segment happens inside speak() before it
    # returns (see app/services/tts.py) — so awaiting it here already
    # fails fast on any config/upstream error, with the real error mapped
    # to the right status before a 200 starts. Later segments (if any)
    # synthesize lazily as `audio.chunks` is consumed below.
    try:
        audio = await tts.speak(_single_chunk_stream(speech_text), voice_id=req.voice_id)
    except TTSUpstreamAudioError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return StreamingResponse(audio.chunks, media_type="application/octet-stream", headers=_audio_headers(audio))


@router.post("/speak/live")
async def speak_live_endpoint(
    req: TTSLiveRequest,
    tts: TTSDep,
    live_generations: LiveGenerationRegistryDep,
) -> StreamingResponse:
    """Synthesize speech for a chat turn that may still be generating.
    `generation_id` comes from POST /api/v1/chat/stream's "conversation"
    SSE event — call this right after receiving that event (in parallel
    with continuing to read the text stream) rather than waiting for
    `done`, so audio for the first sentence can start well before the
    full answer has finished generating. There's no `text`/`message_id`
    equivalent here: this endpoint only ever speaks a specific in-flight
    (or very recently finished) generation.
    """
    broadcast = live_generations.get(req.generation_id)
    if broadcast is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown or expired generation_id")

    try:
        audio = await tts.speak(broadcast.subscribe(), voice_id=req.voice_id)
    except TTSUpstreamAudioError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return StreamingResponse(audio.chunks, media_type="application/octet-stream", headers=_audio_headers(audio))
