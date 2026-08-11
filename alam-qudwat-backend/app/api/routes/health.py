from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbSession

router = APIRouter(tags=["health"])


@router.get("/health")
def health(session: DbSession):
    session.execute(text("SELECT 1"))
    return {"status": "ok"}
