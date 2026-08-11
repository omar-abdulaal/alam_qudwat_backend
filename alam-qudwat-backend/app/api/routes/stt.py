from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.deps import AppSettingsDep, STTDep
from app.schemas.stt import TranscriptionOut

router = APIRouter(prefix="/api/v1/stt", tags=["stt"])

_ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}


@router.post("/transcribe", response_model=TranscriptionOut)
async def transcribe_endpoint(
    stt: STTDep,
    app_settings: AppSettingsDep,
    audio: UploadFile = File(..., description="Audio file to transcribe."),
) -> TranscriptionOut:
    filename = audio.filename or "audio"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported audio file type: {ext}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty audio file")
    if len(audio_bytes) > app_settings.stt_max_audio_bytes:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Audio file too large (max {app_settings.stt_max_audio_bytes} bytes)",
        )

    text = await stt.transcribe(audio_bytes=audio_bytes, filename=filename)
    return TranscriptionOut(text=text)
