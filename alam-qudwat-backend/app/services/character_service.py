from __future__ import annotations

from sqlalchemy.orm import Session, selectinload

from app.db.models import Character, CharacterCategory


def list_characters(
    session: Session,
    *,
    era: str | None = None,
    group: str | None = None,
    category: str | None = None,
) -> list[Character]:
    query = (
        session.query(Character)
        .options(selectinload(Character.category_links))
        .filter_by(is_active=True)
    )
    if era is not None:
        query = query.filter_by(era=era)
    if group is not None:
        query = query.filter_by(group=group)
    if category is not None:
        query = query.filter(Character.category_links.any(CharacterCategory.category_code == category))
    return query.order_by(Character.sort_order).all()


def get_character(session: Session, slug: str) -> Character | None:
    character = session.get(Character, slug)
    if character is None or not character.is_active:
        return None
    return character
