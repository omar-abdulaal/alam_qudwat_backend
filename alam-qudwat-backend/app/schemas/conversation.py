from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.schemas.chat import CitationOut


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationOut] | None
    extra: dict | None  # e.g. {"suggestions": [...]} for assistant turns
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    character_slug: str
    narrator_mode: Literal["kids", "adults"]
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
