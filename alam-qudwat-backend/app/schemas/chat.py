from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

NarratorMode = Literal["kids", "adults"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's chat message.")
    conversation_id: UUID | None = Field(
        default=None, description="Omit to start a new conversation."
    )
    character_slug: str | None = Field(
        default=None, description="Required when conversation_id is omitted."
    )
    mode: NarratorMode | None = Field(
        default=None,
        description="Narrator mode. Required for a new conversation; optional override "
        "for an existing one (updates the conversation's mode going forward).",
    )


class CitationOut(BaseModel):
    index: int
    book_title: str
    author: str
    character: str
    character_name: str
    page: str
    source_url: str
    score: float
