"""Optional dependency gates for probabilistic industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def mapie_available() -> bool:
    return importlib.util.find_spec("mapie") is not None


def ngboost_available() -> bool:
    return importlib.util.find_spec("ngboost") is not None


def probabilistic_industry_available() -> bool:
    """True when MAPIE or NGBoost is importable."""
    return mapie_available() or ngboost_available()


def require_mapie(*, feature: str = "MAPIE conformal prediction") -> Any:
    try:
        import mapie
    except ImportError as exc:
        raise MissingExtraError("probabilistic-industry", feature) from exc
    return mapie


def require_ngboost(*, feature: str = "NGBoost probabilistic boosting") -> Any:
    try:
        import ngboost
    except ImportError as exc:
        raise MissingExtraError("probabilistic-industry", feature) from exc
    return ngboost
