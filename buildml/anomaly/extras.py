"""Optional dependency gates for anomaly industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available


def require_pyod(*, feature: str = "PyOD anomaly detectors") -> Any:
    """Import and return ``pyod``, or raise :class:`MissingExtraError`.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    try:
        import pyod
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return pyod


def pyod_spec_present() -> bool:
    """Cheap find_spec discovery for PyOD (may still fail at import)."""
    return importlib.util.find_spec("pyod") is not None


def pyod_available() -> bool:
    """Return whether PyOD imports cleanly (subprocess probe).

    Capability matrices use this for backends.pyod.available so a broken wheel
    cannot silently report available=True.
    """
    if not pyod_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("pyod")


def lightgbm_spec_present() -> bool:
    """Cheap find_spec discovery for LightGBM."""
    return importlib.util.find_spec("lightgbm") is not None


def lightgbm_available() -> bool:
    """Return whether LightGBM imports cleanly (subprocess probe)."""
    if not lightgbm_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("lightgbm")


def xgboost_spec_present() -> bool:
    """Cheap find_spec discovery for XGBoost."""
    return importlib.util.find_spec("xgboost") is not None


def xgboost_available() -> bool:
    """Return whether XGBoost imports cleanly (subprocess probe)."""
    if not xgboost_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("xgboost")


def gradient_boosting_extras_available() -> bool:
    """True when LightGBM or XGBoost import cleanly at runtime."""
    return lightgbm_available() or xgboost_available()


def anomaly_industry_available() -> bool:
    """True when PyOD or industry supervised GBDT libraries import cleanly."""
    return pyod_available() or gradient_boosting_extras_available()


def require_lightgbm(*, feature: str = "LightGBM supervised anomaly scorer") -> Any:
    """Import optional dependency for lightgbm or raise MissingExtraError.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return lightgbm


def require_xgboost(*, feature: str = "XGBoost supervised anomaly scorer") -> Any:
    """Import optional dependency for xgboost or raise MissingExtraError.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("anomaly-industry", feature) from exc
    return xgboost


def require_torch_anomaly(*, feature: str = "Torch autoencoder anomaly detector") -> Any:
    """Import optional dependency for torch anomaly or raise MissingExtraError.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.
    """
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)
