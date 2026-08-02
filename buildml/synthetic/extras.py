"""Optional dependency gates for synthetic industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def sdv_available() -> bool:
    """True when SDV imports cleanly (may pull torch — catch broken wheels)."""
    if importlib.util.find_spec("sdv") is None:
        return False
    try:
        import sdv  # noqa: F401
    except Exception:
        return False
    return True


def sdmetrics_available() -> bool:
    """True when sdmetrics imports cleanly.

    find_spec alone is insufficient — sdmetrics may import torch and raise
    OSError on broken Windows wheels.
    """
    if importlib.util.find_spec("sdmetrics") is None:
        return False
    try:
        import sdmetrics  # noqa: F401
    except Exception:
        return False
    return True


def great_expectations_available() -> bool:
    if importlib.util.find_spec("great_expectations") is None:
        return False
    try:
        import great_expectations  # noqa: F401
    except Exception:
        return False
    return True


def synthetic_industry_available() -> bool:
    """True when SDV (CTGAN/TVAE/CopulaGAN) is importable."""
    return sdv_available()


def require_sdv(*, feature: str = "SDV tabular synthesizers") -> Any:
    try:
        import sdv
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdv


def require_sdmetrics(*, feature: str = "SDMetrics synthetic quality reports") -> Any:
    try:
        import sdmetrics
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdmetrics
