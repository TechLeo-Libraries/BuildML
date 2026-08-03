"""Optional dependency gates for semi-supervised industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def lightgbm_available() -> bool:
    """Return whether lightgbm optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    """Return whether xgboost optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return importlib.util.find_spec("xgboost") is not None


def gradient_boosting_extras_available() -> bool:
    """Return whether gradient boosting extras optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return lightgbm_available() or xgboost_available()


def semisupervised_industry_available() -> bool:
    """True when industry GBDT pseudo-label libraries are importable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return gradient_boosting_extras_available()


def sentence_transformers_available() -> bool:
    """Return whether sentence transformers optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


def hf_text_available() -> bool:
    """HF text semi-supervised path needs sentence-transformers (ssl extra).

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return sentence_transformers_available()


def require_xgboost(*, feature: str = "XGBoost pseudo-label semi-supervised") -> Any:
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
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return xgboost


def require_lightgbm(*, feature: str = "LightGBM pseudo-label semi-supervised") -> Any:
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
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return lightgbm


def require_sentence_transformers(
    *, feature: str = "HF text pseudo-label semi-supervised"
) -> Any:
    """Import optional dependency for sentence transformers or raise MissingExtraError.

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
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("ssl", feature) from exc
    return sentence_transformers


def require_torch_semisupervised(*, feature: str = "Torch consistency semi-supervised") -> Any:
    """Import optional dependency for torch semisupervised or raise MissingExtraError.

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


__all__ = [
    "gradient_boosting_extras_available",
    "hf_text_available",
    "lightgbm_available",
    "require_lightgbm",
    "require_sentence_transformers",
    "require_torch_semisupervised",
    "require_xgboost",
    "semisupervised_industry_available",
    "sentence_transformers_available",
    "torch_available",
    "torch_spec_available",
    "xgboost_available",
]
