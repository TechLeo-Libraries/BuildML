"""Optional dependency gates for semi-supervised industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def gradient_boosting_extras_available() -> bool:
    return lightgbm_available() or xgboost_available()


def semisupervised_industry_available() -> bool:
    """True when industry GBDT pseudo-label libraries are importable."""
    return gradient_boosting_extras_available()


def sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def hf_text_available() -> bool:
    """HF text semi-supervised path needs sentence-transformers (ssl extra)."""
    return sentence_transformers_available()


def require_xgboost(*, feature: str = "XGBoost pseudo-label semi-supervised") -> Any:
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return xgboost


def require_lightgbm(*, feature: str = "LightGBM pseudo-label semi-supervised") -> Any:
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingExtraError("semisupervised-industry", feature) from exc
    return lightgbm


def require_sentence_transformers(
    *, feature: str = "HF text pseudo-label semi-supervised"
) -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("ssl", feature) from exc
    return sentence_transformers


def require_torch_semisupervised(*, feature: str = "Torch consistency semi-supervised") -> Any:
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
