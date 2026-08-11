from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import AppSettingsDep, DbSession, TTSDep
from app.db.models import Message
from app.schemas.tts import TTSRequest
from app.services.tts import TTS_CHANNELS, TTS_SAMPLE_FORMAT, TTS_SAMPLE_RATE, strip_citation_markers

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


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
        text = message.content
    else:
        text = req.text or ""

    text = strip_citation_markers(text)
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to synthesize")
    if len(text) > app_settings.tts_max_text_length:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"text too long (max {app_settings.tts_max_text_length} chars)"
        )

    try:
        # Fail fast on a config/endpoint error rather than starting a 200
        # response and erroring mid-stream.
        stream = tts.speak(text, voice_id=req.voice_id)
        first_chunk = await stream.__anext__()
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except StopAsyncIteration:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "TTS produced no audio") from None

    async def audio_stream():
        yield first_chunk
        async for chunk in stream:
            yield chunk

    headers = {
        "X-Audio-Sample-Rate": str(TTS_SAMPLE_RATE),
        "X-Audio-Channels": str(TTS_CHANNELS),
        "X-Audio-Sample-Format": TTS_SAMPLE_FORMAT,
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(audio_stream(), media_type="application/octet-stream", headers=headers)
