from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CharacterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    slug: str
    name_ar: str
    era: str
    group: str
    categories: list[str]
    short_description: str
    avatar_url: str | None
    sort_order: int
    created_at: datetime
