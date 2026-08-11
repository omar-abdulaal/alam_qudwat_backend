from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DbSession
from app.schemas.character import CharacterOut
from app.services import character_service

router = APIRouter(prefix="/api/v1/characters", tags=["characters"])


@router.get("", response_model=list[CharacterOut])
def list_characters_endpoint(
    session: DbSession,
    era: str | None = Query(default=None, description="Filter by era, e.g. الخلافة الراشدة"),
    category: str | None = Query(default=None, description="Filter by category"),
):
    return character_service.list_characters(session, era=era, category=category)


@router.get("/{slug}", response_model=CharacterOut)
def get_character_endpoint(slug: str, session: DbSession):
    character = character_service.get_character(session, slug)
    if character is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown character: {slug}")
    return character
