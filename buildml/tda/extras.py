"""Optional TDA dependency gate for ``buildml[tda]`` (ripser + persim)."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_ripser(*, feature: str = "Persistent homology (ripser)") -> Any:
    """Import and return ``ripser``, or raise :class:`MissingExtraError`."""
    try:
        import ripser
    except ImportError as exc:
        raise MissingExtraError("tda", feature) from exc
    except OSError as exc:
        raise MissingExtraError("tda", feature) from exc
    return ripser


def require_persim(*, feature: str = "Persistence vectorization (persim)") -> Any:
    """Import and return ``persim``, or raise :class:`MissingExtraError`."""
    try:
        import persim
    except ImportError as exc:
        raise MissingExtraError("tda", feature) from exc
    except OSError as exc:
        raise MissingExtraError("tda", feature) from exc
    return persim


def require_tda_stack(*, feature: str = "Topological Data Analysis") -> tuple[Any, Any]:
    """Import ``ripser`` and ``persim``, or raise :class:`MissingExtraError`."""
    return require_ripser(feature=feature), require_persim(feature=feature)


def tda_available() -> bool:
    """Return True when both ``ripser`` and ``persim`` can be imported."""
    if importlib.util.find_spec("ripser") is None:
        return False
    if importlib.util.find_spec("persim") is None:
        return False
    try:
        import persim  # noqa: F401
        import ripser  # noqa: F401
    except Exception:
        return False
    return True
