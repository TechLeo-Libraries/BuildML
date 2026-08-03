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
    """Turn one key or a list of them into a validated, immutable set.

    Accepting a single string is the common case and accepting a list allows key
    rotation — issue a new key, keep both accepted for a period, then drop the
    old one. Blank entries are dropped, since a whitespace-only key would
    otherwise be silently accepted and match nothing.

    Parameters
    ----------
    api_keys:
        One key, several keys, or ``None`` for no authentication. Surrounding
        whitespace is stripped, which matters when keys come from environment
        variables or files with trailing newlines.

    Returns
    -------
    frozenset of str
        The cleaned keys, or an empty set when ``api_keys`` is ``None``.
        Immutable so the middleware's allowed set cannot change under it.

    Raises
    ------
    ValidationError
        If keys were supplied but all of them were blank. An empty set is only
        valid as an explicit ``None``, because silently disabling authentication
        for a caller who asked for it is the worse outcome.

    Examples
    --------
    >>> sorted(normalize_api_keys(["  alpha  ", "beta", "   "]))
    ['alpha', 'beta']
    >>> normalize_api_keys(None)
    frozenset()
    >>> normalize_api_keys("solo")
    frozenset({'solo'})

    See Also
    --------
    key_is_authorized : Checking a presented key against this set.
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


def extract_presented_key(authorization: str | None, api_key_header: str | None) -> str | None:
    """Pull the credential out of whichever of the two headers carries it.

    Both spellings are accepted because both are in common use: ``Authorization:
    Bearer <key>`` is the HTTP convention and works with most clients unchanged,
    while ``X-API-Key: <key>`` is simpler to set by hand and in scripts.
    ``X-API-Key`` wins when both are present, on the grounds that it is the more
    deliberate of the two.

    Parameters
    ----------
    authorization:
        The ``Authorization`` header value, or ``None``. Only the ``Bearer``
        scheme is recognised; ``Basic`` and others yield ``None``.
    api_key_header:
        The ``X-API-Key`` header value, or ``None``.

    Returns
    -------
    str or None
        The credential with whitespace stripped, or ``None`` when neither header
        carries a usable one.

    Notes
    -----
    **A malformed header returns ``None`` rather than raising**, so it is
    treated the same as no credential at all and produces a 401. There is no
    useful distinction to draw for a caller who is not authorised either way.

    Examples
    --------
    >>> extract_presented_key("Bearer abc123", None)
    'abc123'
    >>> extract_presented_key(None, "  abc123 ")
    'abc123'
    >>> extract_presented_key("Basic dXNlcjpwYXNz", None) is None
    True
    >>> extract_presented_key(None, None) is None
    True
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


def key_is_authorized(presented: str | None, allowed: Iterable[str]) -> bool:
    """Check a presented key against the allowed set without leaking timing.

    The obvious implementation, ``presented in allowed``, compares strings with
    an early exit on the first differing character. The time it takes therefore
    depends on how many leading characters matched, and an attacker who can
    measure that can recover a key one character at a time. Comparing every
    candidate with :func:`secrets.compare_digest` removes the signal.

    Parameters
    ----------
    presented:
        The credential from the request, or ``None`` when none was supplied.
    allowed:
        The configured keys.

    Returns
    -------
    bool
        Whether the presented key matches one of the allowed keys. ``None``
        always returns ``False``.

    Notes
    -----
    **"Constant-time-ish" is the honest description.** Each comparison is
    constant-time, but the loop still exits on the first match, so the total
    time reveals a little about position within the set. With a handful of keys
    that is not a practical concern; the per-comparison guarantee is what
    matters.

    Examples
    --------
    >>> key_is_authorized("abc123", {"abc123", "def456"})
    True
    >>> key_is_authorized("wrong", {"abc123"})
    False
    >>> key_is_authorized(None, {"abc123"})
    False

    See Also
    --------
    secrets.compare_digest : The comparison being used.
    """
    if presented is None:
        return False
    allowed_list = list(allowed)
    for candidate in allowed_list:
        if secrets.compare_digest(presented, candidate):
            return True
    return False


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid key on every route except the ones left deliberately open.

    Installed by :func:`~buildml.serving.app.create_serving_app` when
    ``api_keys`` is given. Requests without a valid credential get a 401 with a
    ``WWW-Authenticate`` header naming the scheme, so a client knows what to
    send rather than guessing.

    Attributes
    ----------
    api_keys:
        The accepted credentials.
    open_paths:
        Routes that skip the check.

    Notes
    -----
    **``/health`` is open by design.** Liveness probes and load balancers cannot
    usually be configured with credentials, and a probe that fails
    authentication looks identical to a dead process. The health response
    deliberately contains no data about the model beyond what it is serving.

    **``/docs`` and ``/openapi.json`` are also open**, which means the request
    schema is public. Pass an ``open_paths`` set containing only ``/health`` to
    close them.

    **Path matching is exact.** A path with a trailing slash, or any route not
    listed verbatim, is protected.

    See Also
    --------
    buildml.serving.app.create_serving_app : Where this gets installed.
    """

    def __init__(
        self,
        app: Any,
        *,
        api_keys: frozenset[str],
        open_paths: frozenset[str] | None = None,
    ) -> None:
        """Wrap an ASGI app, protecting every path outside ``open_paths``.

        The allowed keys and open paths are captured once here rather than read
        per request, so the set a request is checked against cannot change while
        the server is running.

        Parameters
        ----------
        app:
            The ASGI application to wrap. Supplied by Starlette when the
            middleware is added, not by user code.
        api_keys:
            The accepted keys, already normalised by
            :func:`normalize_api_keys`. Passing an empty set makes every
            protected request fail, which is a configuration error rather than
            an open server.
        open_paths:
            Exact paths that skip the check. Defaults to ``/health``, ``/docs``,
            and ``/openapi.json``. Pass a narrower set to close the docs.
        """
        super().__init__(app)
        self.api_keys = api_keys
        self.open_paths = open_paths or frozenset({"/health", "/docs", "/openapi.json"})

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Let the request through, or answer it with a 401.

        Open paths pass straight to the next handler. Everything else must carry
        a credential in either accepted header that matches a configured key.

        Parameters
        ----------
        request:
            The incoming request, inspected for its path and auth headers.
        call_next:
            The rest of the middleware chain, awaited only for authorised
            requests.

        Returns
        -------
        Response
            The downstream response, or a 401 carrying ``WWW-Authenticate:
            Bearer`` and a message naming both accepted headers.

        Notes
        -----
        **The rejection is deliberately uninformative about why.** A missing
        credential, a malformed header, and a wrong key all produce the same
        response, since distinguishing them tells an attacker which part to
        vary.
        """
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
