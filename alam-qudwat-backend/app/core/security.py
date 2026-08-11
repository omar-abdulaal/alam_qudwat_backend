"""Shared-secret bearer-token auth dependency.

MVP-level access control: every /api/v1/* route requires
"Authorization: Bearer <API_AUTH_TOKEN>". This is what stands between the
OpenAI-backed endpoints and open abuse in the absence of a real
per-user account system (documented limitation — see api_documentation.txt).
"""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.core.config import get_app_settings


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>",
        )

    token = authorization.removeprefix("Bearer ").strip()
    expected = get_app_settings().api_auth_token

    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")
