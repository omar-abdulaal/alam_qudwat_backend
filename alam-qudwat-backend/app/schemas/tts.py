from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class TTSRequest(BaseModel):
    text: str | None = Field(default=None, description="Text to synthesize.")
    message_id: UUID | None = Field(
        default=None, description="Synthesize an existing assistant message's content instead of raw text."
    )
    voice_id: str | None = Field(default=None, description="Overrides the default SILMA voice.")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "TTSRequest":
        if bool(self.text) == bool(self.message_id):
            raise ValueError("Provide exactly one of `text` or `message_id`.")
        return self


class TTSLiveRequest(BaseModel):
    """Synthesize speech for a chat turn that may still be streaming —
    see POST /api/v1/tts/speak/live. `generation_id` comes from the
    "conversation" SSE event of the matching POST /api/v1/chat/stream
    call."""

    generation_id: UUID
    voice_id: str | None = Field(default=None, description="Overrides the default SILMA voice.")
