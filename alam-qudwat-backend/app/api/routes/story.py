from datetime import date

from fastapi import APIRouter, Query

from app.api.deps import DbSession
from app.schemas.story import StoryOfDayOut
from app.services.story_service import get_story_of_day

router = APIRouter(prefix="/api/v1/story-of-day", tags=["story"])


@router.get("", response_model=StoryOfDayOut)
def story_of_day_endpoint(
    session: DbSession,
    for_date: date | None = Query(default=None, alias="date", description="Defaults to today (server time)"),
):
    return get_story_of_day(session, for_date=for_date)
