"""Exercises the real auth dependency (the shared `client` fixture bypasses
it on purpose for every other test's convenience)."""
from fastapi.testclient import TestClient

from app.core.config import get_app_settings
from app.db.session import get_db
from app.main import app


def test_missing_auth_header_returns_401(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as c:
            resp = c.get("/api/v1/characters")
            assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_wrong_token_returns_401(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as c:
            resp = c.get("/api/v1/characters", headers={"Authorization": "Bearer wrong-token"})
            assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_correct_bearer_token_succeeds(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    token = get_app_settings().api_auth_token
    try:
        with TestClient(app) as c:
            resp = c.get("/api/v1/characters", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_does_not_require_auth(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
    finally:
        app.dependency_overrides.clear()
