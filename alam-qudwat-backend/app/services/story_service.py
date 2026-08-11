"""'Story of the day' resolution — entirely server-computed, no coupling
to the Flutter client and no admin UI required for the default path.
"""
from __future__ import annotations

import hashlib
from datetime import date

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Character, StoryOfDay
from app.schemas.character import CharacterOut
from app.schemas.story import StoryOfDayOut


def get_story_of_day(session: Session, *, for_date: date | None = None) -> StoryOfDayOut:
    for_date = for_date or date.today()

    override = session.query(StoryOfDay).filter_by(story_date=for_date).one_or_none()
    if override is not None:
        character = session.get(Character, override.character_slug)
        if character is not None and character.is_active:
            return StoryOfDayOut(
                story_date=for_date,
                character=CharacterOut.model_validate(character),
                title=override.title,
                teaser=override.teaser,
                is_curated=True,
            )
        # Curated row points at an inactive/missing character — fall through
        # to the deterministic rotation rather than erroring for the day.

    active = session.query(Character).filter_by(is_active=True).order_by(Character.sort_order).all()
    if not active:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No active characters configured")

    # Deterministic, stateless rotation keyed by the date — same result for
    # every request on a given day, no row needs to be written.
    digest = hashlib.sha256(for_date.isoformat().encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(active)
    character = active[index]

    return StoryOfDayOut(
        story_date=for_date,
        character=CharacterOut.model_validate(character),
        title=None,
        teaser=None,
        is_curated=False,
    )
