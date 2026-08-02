"""Optional dependency gates for synthetic industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def sdv_available() -> bool:
    return importlib.util.find_spec("sdv") is not None


def sdmetrics_available() -> bool:
    return importlib.util.find_spec("sdmetrics") is not None


def great_expectations_available() -> bool:
    return importlib.util.find_spec("great_expectations") is not None


def synthetic_industry_available() -> bool:
    """True when SDV (CTGAN/TVAE/CopulaGAN) is importable."""
    return sdv_available()


def require_sdv(*, feature: str = "SDV tabular synthesizers") -> Any:
    try:
        import sdv
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdv


def require_sdmetrics(*, feature: str = "SDMetrics synthetic quality reports") -> Any:
    try:
        import sdmetrics
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdmetrics
