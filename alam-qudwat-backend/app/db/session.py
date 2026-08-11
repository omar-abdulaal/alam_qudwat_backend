"""FastAPI DB-session dependency.

Reuses rag.db.session's engine/session factory (same DATABASE_URL, same
connection pool) rather than standing up a second one — the app/ and
rag/ layers share one physical database and one pool.
"""
from __future__ import annotations

from typing import Iterator

from sqlalchemy.orm import Session

from rag.db.session import get_session_factory


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
