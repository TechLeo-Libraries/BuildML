"""Optional dependency gates for causal industry backends (DoWhy / EconML)."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_dowhy(*, feature: str = "DoWhy causal backend") -> Any:
    """Import and return ``dowhy``, or raise :class:`MissingExtraError`."""
    try:
        import dowhy
    except ImportError as exc:
        raise MissingExtraError("causal-industry", feature) from exc
    return dowhy


def require_econml(*, feature: str = "EconML causal backend") -> Any:
    """Import and return ``econml``, or raise :class:`MissingExtraError`."""
    try:
        import econml
    except ImportError as exc:
        raise MissingExtraError("causal-industry", feature) from exc
    return econml


def dowhy_available() -> bool:
    return importlib.util.find_spec("dowhy") is not None


def econml_available() -> bool:
    return importlib.util.find_spec("econml") is not None


def causal_industry_available() -> bool:
    """True when DoWhy or EconML is importable."""
    return dowhy_available() or econml_available()
