"""Shared fixtures for integration tests.

Integration tests need a real Postgres instance, with the schema already
migrated (`alembic upgrade head`) and the pgvector extension enabled.
They are skipped automatically if that isn't available — unit tests
(chunker/cleaning/hashing) never depend on this.

SAFETY: two earlier versions of this fixture tried to isolate tests via
a "rag_test" Postgres *schema* (search_path) and then via a separate
"<db>_test" *database*, both inside/alongside the real DATABASE_URL. The
schema approach silently failed to isolate an ORM session's queries and
wiped real `documents`/`chunks` rows; the separate-database approach hit
an environment-specific auth restriction. Neither is used anymore.

Instead, every test runs inside a single outer database transaction that
is opened before the test and unconditionally rolled back after — using
SQLAlchemy's savepoint join mode, so even if test/application code calls
`session.commit()`, that only commits a SAVEPOINT nested inside the outer
transaction. The final `rollback()` on the outer transaction discards
everything no matter what happened inside. This is standard SQLAlchemy
test-isolation practice and does not depend on search_path, a second
database, or remembering to clean up rows manually.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    os.environ.get("DATABASE_URL", "postgresql+psycopg://postgres:1234@localhost:5432/alam_qudwat"),
)


@pytest.fixture(scope="session")
def _engine():
    engine = create_engine(DATABASE_URL, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM documents LIMIT 0"))
            conn.execute(text("SELECT 1 FROM chunks LIMIT 0"))
            conn.execute(text("SELECT '[1]'::vector(1)"))
    except (OperationalError, ProgrammingError) as exc:
        pytest.skip(
            f"Postgres at {DATABASE_URL} isn't ready for tests (schema migrated? "
            f"pgvector extension enabled?): {exc}"
        )
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(_engine):
    """A Session bound to a connection whose outer transaction is always
    rolled back at teardown — nothing a test does is ever persisted."""
    connection = _engine.connect()
    outer_transaction = connection.begin()
    Session = sessionmaker(bind=connection, future=True, join_transaction_mode="create_savepoint")
    session = Session()
    try:
        yield session
    finally:
        session.close()
        outer_transaction.rollback()
        connection.close()


@pytest.fixture()
def client(db_session):
    """A FastAPI TestClient wired to the same rolled-back-transaction
    session as `db_session` (via dependency override) and with auth
    bypassed by default — API tests inherit the same "never touches real
    data" guarantee already proven for the RAG tests. Auth itself is
    tested separately in test_api_auth.py without this override."""
    from fastapi.testclient import TestClient

    from app.core.security import require_api_token
    from app.db.session import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[require_api_token] = lambda: None
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
