"""Logging setup: structured-ish console logging + a request-timing
middleware. Never logs message content or secrets — only method, path,
status, and duration.
"""
from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.config import get_app_settings

logger = logging.getLogger("app")


def configure_logging() -> None:
    logging.basicConfig(
        level=get_app_settings().log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("%s %s failed after %.1fms", request.method, request.url.path, duration_ms)
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms
        )
        return response
