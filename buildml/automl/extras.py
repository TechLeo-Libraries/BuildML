"""Optional dependency gates for AutoML industry backends.

Native grid/randomized/evolutionary search is always available. Optuna-backed
search requires ``buildml[automl]``; FLAML and AutoGluon industry adapters
require ``buildml[automl-industry]``.

See Also
--------
buildml.automl.catalog.automl_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_optuna(*, feature: str = "Optuna AutoML search") -> Any:
    """Import and return ``optuna``, or raise :class:`MissingExtraError`.

    Called by Optuna-backed AutoML paths at fit time so missing extras surface
    as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported optuna module.

    Raises
    ------
    MissingExtraError
        When optuna is not installed. Install with ``pip install 'buildml[automl]'``.
    """
    try:
        import optuna
    except ImportError as exc:
        raise MissingExtraError("automl", feature) from exc
    return optuna


def require_flaml(*, feature: str = "FLAML tabular AutoML adapter") -> Any:
    """Import and return ``flaml``, or raise :class:`MissingExtraError`.

    Called by the FLAML industry adapter when ``backend='flaml'`` is selected.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported flaml module.

    Raises
    ------
    MissingExtraError
        When flaml is not installed. Install with
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import flaml
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return flaml


def require_autogluon(*, feature: str = "AutoGluon tabular AutoML adapter") -> Any:
    """Import ``TabularPredictor``, or raise :class:`MissingExtraError`.

    Called by the AutoGluon industry adapter when ``backend='autogluon'`` is
    selected.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    type
        The ``autogluon.tabular.TabularPredictor`` class.

    Raises
    ------
    MissingExtraError
        When autogluon is not installed. Install with
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return TabularPredictor


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def optuna_available() -> bool:
    """Return whether Optuna imports cleanly (subprocess probe)."""
    if importlib.util.find_spec("optuna") is None:
        return False
    return _runtime_ok("optuna")


def flaml_spec_present() -> bool:
    """Cheap find_spec discovery for FLAML."""
    return importlib.util.find_spec("flaml") is not None


def flaml_available() -> bool:
    """Return whether FLAML imports cleanly (subprocess probe)."""
    if not flaml_spec_present():
        return False
    return _runtime_ok("flaml")


def autogluon_spec_present() -> bool:
    """Cheap find_spec discovery for AutoGluon."""
    return importlib.util.find_spec("autogluon") is not None


def autogluon_available() -> bool:
    """Return whether AutoGluon tabular imports cleanly (subprocess probe)."""
    if not autogluon_spec_present():
        return False
    return _runtime_ok("autogluon.tabular")


def lightgbm_available() -> bool:
    """Return whether LightGBM imports cleanly (subprocess probe)."""
    if importlib.util.find_spec("lightgbm") is None:
        return False
    return _runtime_ok("lightgbm")


def xgboost_available() -> bool:
    """Return whether XGBoost imports cleanly (subprocess probe)."""
    if importlib.util.find_spec("xgboost") is None:
        return False
    return _runtime_ok("xgboost")


def catboost_available() -> bool:
    """Return whether CatBoost imports cleanly (subprocess probe)."""
    if importlib.util.find_spec("catboost") is None:
        return False
    return _runtime_ok("catboost")


def gradient_boosting_extras_available() -> bool:
    """Return whether at least one industry GBDT library imports cleanly."""
    return lightgbm_available() or xgboost_available() or catboost_available()


def automl_industry_available() -> bool:
    """Return whether FLAML or AutoGluon import cleanly at runtime."""
    return flaml_available() or autogluon_available()


def require_lightgbm(*, feature: str = "LightGBM estimator family") -> Any:
    """Import and return ``lightgbm``, or raise :class:`MissingExtraError`.

    Called when the native AutoML catalog includes a LightGBM family entry.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported lightgbm module.

    Raises
    ------
    MissingExtraError
        When lightgbm is not installed. Install with
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost estimator family") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`.

    Called when the native AutoML catalog includes an XGBoost family entry.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported xgboost module.

    Raises
    ------
    MissingExtraError
        When xgboost is not installed. Install with
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return xgboost


def require_catboost(*, feature: str = "CatBoost estimator family") -> Any:
    """Import and return ``catboost``, or raise :class:`MissingExtraError`.

    Called when the native AutoML catalog includes a CatBoost family entry.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported catboost module.

    Raises
    ------
    MissingExtraError
        When catboost is not installed. Install with
        ``pip install 'buildml[automl-industry]'``.
    """
    try:
        import catboost
    except ImportError as exc:
        raise MissingExtraError("automl-industry", feature) from exc
    return catboost
