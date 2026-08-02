"""BuildML managed model serving (local FastAPI alpha).

Install ``buildml[serve]``. Bind defaults to localhost. Optional API-key/Bearer
middleware is available; still not a managed IAM / cloud product — put a
reverse proxy (TLS) in front for any non-local exposure.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "APIKeyAuthMiddleware",
    "ServeHandle",
    "ServingLaunchError",
    "create_serving_app",
    "normalize_api_keys",
    "serve_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"create_serving_app"}:
        from buildml.serving.app import create_serving_app

        return create_serving_app
    if name in {"APIKeyAuthMiddleware", "normalize_api_keys"}:
        from buildml.serving import auth as auth_mod

        return getattr(auth_mod, name)
    if name in {"ServeHandle", "ServingLaunchError", "serve_bundle"}:
        from buildml.serving import launch as launch_mod

        return getattr(launch_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
