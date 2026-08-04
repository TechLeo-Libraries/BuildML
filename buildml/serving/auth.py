"""Optional API-key / Bearer and HTTP Basic auth for BuildML serving.

Default serving remains open on localhost. When API keys and/or Basic
credentials are configured, protected routes require either:

* ``Authorization: Bearer <key>`` / ``X-API-Key: <key>``, or
* ``Authorization: Basic <base64(user:pass)>``

Either mechanism may authorize when both are configured. This is thin
library middleware: **not** managed identity / cloud IAM. Prefer TLS + auth
at a reverse proxy for internet exposure.
"""

from __future__ import annotations

import base64
import binascii
import secrets
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from buildml.core.errors import ValidationError

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
except ImportError:  # pragma: no cover - serve extra missing
    BaseHTTPMiddleware = object  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    Response = Any  # type: ignore[assignment,misc]


DEFAULT_OPEN_PATHS_WITH_DOCS: frozenset[str] = frozenset(
    {"/health", "/docs", "/openapi.json", "/redoc"}
)
DEFAULT_OPEN_PATHS_HEALTH_ONLY: frozenset[str] = frozenset({"/health"})


def normalize_api_keys(api_keys: str | Sequence[str] | None) -> frozenset[str]:
    """Turn one key or a list of them into a validated, immutable set.

    Accepting a single string is the common case and accepting a list allows key
    rotation: issue a new key, keep both accepted for a period, then drop the
    old one. Blank entries are dropped, since a whitespace-only key would
    otherwise be silently accepted and match nothing.

    Parameters
    ----------
    api_keys:
        One key, several keys, or ``None`` for no API-key authentication.
        Surrounding whitespace is stripped.

    Returns
    -------
    frozenset of str
        The cleaned keys, or an empty set when ``api_keys`` is ``None``.

    Raises
    ------
    ValidationError
        If keys were supplied but all of them were blank.
    """
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


def normalize_basic_credentials(
    basic_auth: str
    | tuple[str, str]
    | Sequence[tuple[str, str]]
    | Mapping[str, str]
    | None,
) -> frozenset[tuple[str, str]]:
    """Normalize Basic-auth credential pairs into an immutable set.

    Parameters
    ----------
    basic_auth:
        ``'user:pass'``, ``(user, pass)``, a sequence of pairs, a
        ``{username, password}`` mapping, or ``None``.

    Returns
    -------
    frozenset of tuple[str, str]
        Accepted ``(username, password)`` pairs.

    Raises
    ------
    ValidationError
        If credentials were supplied but none were usable.
    """
    if basic_auth is None:
        return frozenset()
    pairs: list[tuple[str, str]] = []
    if isinstance(basic_auth, str):
        text = basic_auth.strip()
        if not text:
            raise ValidationError("basic_auth must contain username:password")
        if ":" not in text:
            raise ValidationError(
                "basic_auth string must be 'username:password' (colon-separated)"
            )
        user, _, password = text.partition(":")
        if not user.strip():
            raise ValidationError("basic_auth username must be non-empty")
        pairs.append((user.strip(), password))
    elif isinstance(basic_auth, Mapping):
        mapping_user: str | None = basic_auth.get("username")
        if mapping_user is None:
            mapping_user = basic_auth.get("user")
        mapping_password: str | None = basic_auth.get("password")
        if mapping_password is None:
            mapping_password = basic_auth.get("pass")
        if mapping_user is None or mapping_password is None:
            raise ValidationError(
                "basic_auth mapping requires both username and password"
            )
        user_s = str(mapping_user).strip()
        if not user_s:
            raise ValidationError("basic_auth username must be non-empty")
        pairs.append((user_s, str(mapping_password)))
    elif isinstance(basic_auth, Sequence) and not isinstance(basic_auth, (str, bytes)):
        # Distinguish a single (user, pass) pair from a list of pairs.
        items = list(basic_auth)
        if len(items) == 2 and all(not isinstance(x, (tuple, list)) for x in items):
            user_s = str(items[0]).strip()
            if not user_s:
                raise ValidationError("basic_auth username must be non-empty")
            pairs.append((user_s, str(items[1])))
        else:
            for item in items:
                if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
                    raise ValidationError(
                        "basic_auth sequence entries must be (username, password)"
                    )
                entry = list(item)
                if len(entry) != 2:
                    raise ValidationError(
                        "basic_auth sequence entries must be (username, password)"
                    )
                user_s = str(entry[0]).strip()
                if not user_s:
                    raise ValidationError("basic_auth username must be non-empty")
                pairs.append((user_s, str(entry[1])))
    else:
        raise ValidationError(
            "basic_auth must be 'user:pass', (user, pass), or {username, password}"
        )
    if not pairs:
        raise ValidationError("basic_auth must contain at least one credential pair")
    return frozenset(pairs)


def extract_presented_key(authorization: str | None, api_key_header: str | None) -> str | None:
    """Pull an API-key credential from Bearer or ``X-API-Key``.

    ``X-API-Key`` wins when both are present. ``Basic`` and other schemes are
    ignored here (handled by :func:`extract_basic_credentials`).
    """
    if api_key_header and api_key_header.strip():
        return api_key_header.strip()
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token or None
    return None


def extract_basic_credentials(authorization: str | None) -> tuple[str, str] | None:
    """Decode ``Authorization: Basic ...`` into ``(username, password)``.

    Returns ``None`` for missing, non-Basic, or malformed headers so callers
    treat them the same as no credential.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return None
    token = parts[1].strip()
    if not token:
        return None
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if ":" not in decoded:
        return None
    user, _, password = decoded.partition(":")
    if not user:
        return None
    return user, password


def key_is_authorized(presented: str | None, allowed: Iterable[str]) -> bool:
    """Check a presented API key against the allowed set without leaking timing."""
    if presented is None:
        return False
    allowed_list = list(allowed)
    for candidate in allowed_list:
        if secrets.compare_digest(presented, candidate):
            return True
    return False


def basic_is_authorized(
    presented: tuple[str, str] | None,
    allowed: Iterable[tuple[str, str]],
) -> bool:
    """Check Basic credentials with constant-time-ish comparisons."""
    if presented is None:
        return False
    user, password = presented
    for cand_user, cand_password in allowed:
        user_ok = secrets.compare_digest(user, cand_user)
        pass_ok = secrets.compare_digest(password, cand_password)
        if user_ok and pass_ok:
            return True
    return False


def request_is_authorized(
    *,
    authorization: str | None,
    api_key_header: str | None,
    api_keys: Iterable[str],
    basic_credentials: Iterable[tuple[str, str]],
) -> bool:
    """Return True when API-key **or** Basic credentials authorize the request."""
    keys = list(api_keys)
    basics = list(basic_credentials)
    if keys:
        presented_key = extract_presented_key(authorization, api_key_header)
        if key_is_authorized(presented_key, keys):
            return True
    if basics:
        presented_basic = extract_basic_credentials(authorization)
        if basic_is_authorized(presented_basic, basics):
            return True
    return False


def _www_authenticate_value(*, api_keys: bool, basic: bool) -> str:
    parts: list[str] = []
    if api_keys:
        parts.append("Bearer")
    if basic:
        parts.append('Basic realm="buildml-serve"')
    return ", ".join(parts) if parts else "Bearer"


def _unauthorized_detail(*, api_keys: bool, basic: bool) -> str:
    options: list[str] = []
    if api_keys:
        options.append("Authorization: Bearer <key> or X-API-Key: <key>")
    if basic:
        options.append("Authorization: Basic <base64(user:pass)>")
    joined = " or ".join(options) if options else "a valid credential"
    return f"Unauthorized. Provide {joined}."


class ServingAuthMiddleware(BaseHTTPMiddleware):
    """Require API-key and/or HTTP Basic credentials on protected routes.

    Either configured mechanism may authorize. ``/health`` stays open for
    probes. Docs/OpenAPI paths are open only when included in ``open_paths``
    (closed by default when auth is enabled and docs are not opted in).
    """

    def __init__(
        self,
        app: Any,
        *,
        api_keys: frozenset[str] | None = None,
        basic_credentials: frozenset[tuple[str, str]] | None = None,
        open_paths: frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.api_keys = api_keys or frozenset()
        self.basic_credentials = basic_credentials or frozenset()
        if not self.api_keys and not self.basic_credentials:
            raise ValidationError(
                "ServingAuthMiddleware requires api_keys and/or basic_credentials"
            )
        self.open_paths = (
            open_paths
            if open_paths is not None
            else DEFAULT_OPEN_PATHS_HEALTH_ONLY
        )

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        path = request.url.path
        if path in self.open_paths:
            return cast(Response, await call_next(request))
        authorized = request_is_authorized(
            authorization=request.headers.get("authorization"),
            api_key_header=request.headers.get("x-api-key"),
            api_keys=self.api_keys,
            basic_credentials=self.basic_credentials,
        )
        if not authorized:
            has_keys = bool(self.api_keys)
            has_basic = bool(self.basic_credentials)
            return cast(
                Response,
                JSONResponse(
                    {
                        "detail": _unauthorized_detail(
                            api_keys=has_keys, basic=has_basic
                        )
                    },
                    status_code=401,
                    headers={
                        "WWW-Authenticate": _www_authenticate_value(
                            api_keys=has_keys, basic=has_basic
                        )
                    },
                ),
            )
        return cast(Response, await call_next(request))


class APIKeyAuthMiddleware(ServingAuthMiddleware):
    """Backward-compatible API-key-only middleware.

    Prefer :class:`ServingAuthMiddleware` when configuring Basic auth as well.
    Default ``open_paths`` historically included docs; callers that want closed
    docs should pass ``open_paths`` explicitly (``create_serving_app`` does).
    """

    def __init__(
        self,
        app: Any,
        *,
        api_keys: frozenset[str],
        open_paths: frozenset[str] | None = None,
    ) -> None:
        super().__init__(
            app,
            api_keys=api_keys,
            basic_credentials=frozenset(),
            open_paths=(
                open_paths
                if open_paths is not None
                else DEFAULT_OPEN_PATHS_WITH_DOCS
            ),
        )
