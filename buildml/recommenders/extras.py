"""Optional dependency gates for recommender industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def implicit_available() -> bool:
    return importlib.util.find_spec("implicit") is not None


def lightfm_available() -> bool:
    return importlib.util.find_spec("lightfm") is not None


def recommenders_industry_available() -> bool:
    """True when implicit or LightFM is importable."""
    return implicit_available() or lightfm_available()


def require_implicit(*, feature: str = "implicit ALS/BPR recommenders") -> Any:
    try:
        import implicit
    except ImportError as exc:
        raise MissingExtraError("recommenders-industry", feature) from exc
    return implicit


def require_lightfm(*, feature: str = "LightFM hybrid recommender") -> Any:
    try:
        import lightfm
    except ImportError as exc:
        raise MissingExtraError("recommenders-industry", feature) from exc
    return lightfm
