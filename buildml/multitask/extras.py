"""Optional dependency gates for multi-task industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def catboost_available() -> bool:
    return importlib.util.find_spec("catboost") is not None


def gradient_boosting_extras_available() -> bool:
    return lightgbm_available() or xgboost_available() or catboost_available()


def multitask_industry_available() -> bool:
    """True when industry GBDT multi-target libraries are importable."""
    return gradient_boosting_extras_available()


def require_xgboost(*, feature: str = "XGBoost multi-target multi-task") -> Any:
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return xgboost


def require_lightgbm(*, feature: str = "LightGBM multi-target multi-task") -> Any:
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return lightgbm


def require_catboost(*, feature: str = "CatBoost multi-target multi-task") -> Any:
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("multitask-industry", feature) from exc
    return catboost


def require_torch_multitask(*, feature: str = "Torch shared-trunk multi-head multi-task") -> Any:
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "catboost_available",
    "gradient_boosting_extras_available",
    "lightgbm_available",
    "multitask_industry_available",
    "require_catboost",
    "require_lightgbm",
    "require_torch_multitask",
    "require_xgboost",
    "torch_available",
    "torch_spec_available",
    "xgboost_available",
]
