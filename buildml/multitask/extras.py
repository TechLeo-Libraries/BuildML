"""Optional dependency gates for multi-task industry backends.

Native sklearn MultiOutput/Chain paths are always available. Industry GBDT
multi-target and torch shared-trunk paths require optional extras.

Industry ``*_available`` predicates use subprocess import probes so broken
native wheels are never reported as ready. Use ``*_spec_present`` for cheap
discovery disclosure in capability matrices.

See Also
--------
buildml.multitask.catalog.multitask_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def lightgbm_spec_present() -> bool:
    """Cheap find_spec discovery for LightGBM."""
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_spec_present() -> bool:
    """Cheap find_spec discovery for XGBoost."""
    return importlib.util.find_spec("xgboost") is not None


def catboost_spec_present() -> bool:
    """Cheap find_spec discovery for CatBoost."""
    return importlib.util.find_spec("catboost") is not None


def lightgbm_available() -> bool:
    """Return whether lightgbm imports cleanly for multi_output_lgbm."""
    if not lightgbm_spec_present():
        return False
    return _runtime_ok("lightgbm")


def xgboost_available() -> bool:
    """Return whether xgboost imports cleanly for multi_output_xgb."""
    if not xgboost_spec_present():
        return False
    return _runtime_ok("xgboost")


def catboost_available() -> bool:
    """Return whether catboost imports cleanly for multi_output_catboost."""
    if not catboost_spec_present():
        return False
    return _runtime_ok("catboost")


def gradient_boosting_extras_available() -> bool:
    """Return whether at least one industry GBDT library imports cleanly."""
    return lightgbm_available() or xgboost_available() or catboost_available()


def multitask_industry_available() -> bool:
    """Return whether industry GBDT multi-target libraries import cleanly."""
    return gradient_boosting_extras_available()


def require_xgboost(*, feature: str = "XGBoost multi-target multi-task") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`."""
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return xgboost


def require_lightgbm(*, feature: str = "LightGBM multi-target multi-task") -> Any:
    """Import and return ``lightgbm``, or raise :class:`MissingExtraError`."""
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return lightgbm


def require_catboost(*, feature: str = "CatBoost multi-target multi-task") -> Any:
    """Import and return ``catboost``, or raise :class:`MissingExtraError`."""
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return catboost


def require_torch_multitask(*, feature: str = "Torch shared-trunk multi-head multi-task") -> Any:
    """Import and return ``torch`` for the shared-trunk multi-head backend."""
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "catboost_available",
    "catboost_spec_present",
    "gradient_boosting_extras_available",
    "lightgbm_available",
    "lightgbm_spec_present",
    "multitask_industry_available",
    "require_catboost",
    "require_lightgbm",
    "require_torch_multitask",
    "require_xgboost",
    "torch_available",
    "torch_spec_available",
    "xgboost_available",
    "xgboost_spec_present",
]
