"""Optional API-key / Bearer auth middleware for BuildML serving.

Default serving remains open on localhost. When ``api_keys`` is provided,
``/predict`` (and optionally all non-docs routes) require
``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.

This is a thin library middleware — **not** a managed identity / cloud IAM
product. Prefer TLS + auth at a reverse proxy for internet exposure.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Sequence
from typing import Any

from buildml.core.errors import ValidationError

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
except ImportError:  # pragma: no cover - serve extra missing
    BaseHTTPMiddleware = object  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    Response = Any  # type: ignore[assignment]


def normalize_api_keys(api_keys: str | Sequence[str] | None) -> frozenset[str]:
    """Normalize caller-provided API keys to a frozenset of non-empty strings."""
    if api_keys is None:
        return frozenset()
    if isinstance(api_keys, str):
        items = [api_keys]
    else:
        items = list(api_keys)
    cleaned = {str(k).strip() for k in items if str(k).strip()}
    if not cleaned:
        raise ValidationError("api_keys must contain at least one non-empty key")
    return frozenset(cleaned)


def extract_presented_key(authorization: str | None, api_key_header: str | None) -> str | None:
    """Extract a bearer or X-API-Key credential from request headers."""
    if api_key_header and api_key_header.strip():
        return api_key_header.strip()
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token or None
    return None


def key_is_authorized(presented: str | None, allowed: Iterable[str]) -> bool:
    """Constant-time-ish membership check for presented API keys."""
    if presented is None:
        return False
    allowed_list = list(allowed)
    for candidate in allowed_list:
        if secrets.compare_digest(presented, candidate):
            return True
    return False


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing API-key / Bearer auth on protected paths."""

    def __init__(
        self,
        app: Any,
        *,
        api_keys: frozenset[str],
        open_paths: frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.api_keys = api_keys
        self.open_paths = open_paths or frozenset({"/health", "/docs", "/openapi.json"})

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        path = request.url.path
        if path in self.open_paths:
            return await call_next(request)
        presented = extract_presented_key(
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
        )
        if not key_is_authorized(presented, self.api_keys):
            return JSONResponse(
                {
                    "detail": (
                        "Unauthorized. Provide Authorization: Bearer <key> "
                        "or X-API-Key: <key>."
                    )
                },
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
