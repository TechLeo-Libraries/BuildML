"""Optional dependency gates for AutoML industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_optuna(*, feature: str = "Optuna AutoML search") -> Any:
    """Import and return ``optuna``, or raise :class:`MissingExtraError`."""
    try:
        import optuna
    except ImportError as exc:
        raise MissingExtraError("automl", feature) from exc
    return optuna


def require_flaml(*, feature: str = "FLAML tabular AutoML adapter") -> Any:
    """Import and return ``flaml``, or raise :class:`MissingExtraError`."""
    try:
        import flaml
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return flaml


def require_autogluon(*, feature: str = "AutoGluon tabular AutoML adapter") -> Any:
    """Import TabularPredictor, or raise :class:`MissingExtraError`."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return TabularPredictor


def optuna_available() -> bool:
    return importlib.util.find_spec("optuna") is not None


def flaml_available() -> bool:
    return importlib.util.find_spec("flaml") is not None


def autogluon_available() -> bool:
    return importlib.util.find_spec("autogluon") is not None


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def catboost_available() -> bool:
    return importlib.util.find_spec("catboost") is not None


def gradient_boosting_extras_available() -> bool:
    """True when at least one industry GBDT library is importable."""
    return lightgbm_available() or xgboost_available() or catboost_available()


def automl_industry_available() -> bool:
    """True when FLAML or AutoGluon industry adapters can be imported."""
    return flaml_available() or autogluon_available()


def require_lightgbm(*, feature: str = "LightGBM estimator family") -> Any:
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost estimator family") -> Any:
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return xgboost


def require_catboost(*, feature: str = "CatBoost estimator family") -> Any:
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return catboost
