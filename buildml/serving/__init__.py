"""BuildML managed model serving (local FastAPI alpha).

Install ``buildml[serve]``. Bind defaults to localhost; no auth product claim —
put a reverse proxy in front for any non-local exposure.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ServeHandle",
    "ServingLaunchError",
    "create_serving_app",
    "serve_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"create_serving_app"}:
        from buildml.serving.app import create_serving_app

        return create_serving_app
    if name in {"ServeHandle", "ServingLaunchError", "serve_bundle"}:
        from buildml.serving import launch as launch_mod

        return getattr(launch_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
