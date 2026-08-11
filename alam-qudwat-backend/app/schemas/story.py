from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.schemas.character import CharacterOut


class StoryOfDayOut(BaseModel):
    story_date: date
    character: CharacterOut
    title: str | None
    teaser: str | None
    is_curated: bool  # False when this came from the deterministic fallback rotation
