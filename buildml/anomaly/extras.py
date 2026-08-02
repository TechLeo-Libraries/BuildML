"""Optional dependency gates for anomaly industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available


def require_pyod(*, feature: str = "PyOD anomaly detectors") -> Any:
    """Import and return ``pyod``, or raise :class:`MissingExtraError`."""
    try:
        import pyod
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return pyod


def pyod_available() -> bool:
    return importlib.util.find_spec("pyod") is not None


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def gradient_boosting_extras_available() -> bool:
    return lightgbm_available() or xgboost_available()


def anomaly_industry_available() -> bool:
    """True when PyOD or industry supervised GBDT libraries are importable."""
    return pyod_available() or gradient_boosting_extras_available()


def require_lightgbm(*, feature: str = "LightGBM supervised anomaly scorer") -> Any:
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost supervised anomaly scorer") -> Any:
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return xgboost


def require_torch_anomaly(*, feature: str = "Torch autoencoder anomaly detector") -> Any:
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)
