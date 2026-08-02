"""Semi-supervised catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_spec_available
from buildml.semisupervised.extras import (
    gradient_boosting_extras_available,
    hf_text_available,
    lightgbm_available,
    semisupervised_industry_available,
    xgboost_available,
)

SemiSupervisedBackendName = Literal["sklearn", "industry", "torch", "hf"]


def semisupervised_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for semi-supervised backends and methods."""
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": ["label_propagation", "label_spreading", "self_training"],
                "modality": "tabular",
                "notes": "Core sklearn semi_supervised estimators; always available.",
            },
            "industry": {
                "available": semisupervised_industry_available(),
                "extra": "semisupervised-industry",
                "methods": ["pseudo_label_xgb", "pseudo_label_lgbm"],
                "modality": "tabular",
                "notes": (
                    "Iterative pseudo-labeling with XGBoost/LightGBM when installed. "
                    "Falls back to sklearn self_training when industry extra missing."
                ),
            },
            "torch": {
                "available": torch_spec_available(),
                "extra": "torch",
                "methods": ["fixmatch_tabular", "mixmatch_tabular"],
                "modality": "tabular",
                "notes": (
                    "FixMatch/MixMatch-style consistency + pseudo-label training "
                    "for numeric tabular features (buildml[torch])."
                ),
            },
            "hf": {
                "available": hf_text_available(),
                "extra": "ssl",
                "methods": ["text_pseudo_label"],
                "modality": "text",
                "notes": (
                    "Sentence-transformer embeddings + pseudo-label self-training "
                    "on a single text feature column (buildml[ssl])."
                ),
            },
        },
        "ssl_integration": {
            "documented_pipeline": [
                "fit_ssl_pretext (train-only representation learning)",
                "transform_ssl or reduce_dimensions on SSL embeddings",
                "fit_semisupervised with partial labels on embedding columns",
            ],
            "finetune_ssl_head": (
                "Labeled train rows only — use fit_semisupervised when unlabeled "
                "train rows should participate via propagation/pseudo-labels."
            ),
        },
        "label_convention": {
            "unlabeled": "target NaN/NA/None by default (unlabeled_marker optional)",
            "internal": "sklearn -1 for industry/torch/sklearn paths",
        },
        "evaluation": {
            "metrics": ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"],
            "holdout_rule": "labeled rows only; unlabeled holdout never scored as truth",
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_method_when_installed": _default_method_when_installed(),
        "install_hints": {
            "semisupervised-industry": (
                "pip install 'buildml[semisupervised-industry]'  "
                "# XGBoost/LightGBM pseudo-label paths"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# FixMatch/MixMatch tabular consistency training"
            ),
            "ssl": (
                "pip install 'buildml[ssl]'  "
                "# HF text pseudo-label via sentence-transformers"
            ),
        },
        "non_goals": [
            "Full computer-vision FixMatch on raw pixels",
            "Active-learning query loops (see buildml.activelearning)",
            "Self-supervised pretext without labels (see buildml.selfsupervised)",
            "Anomaly novelty detection (see buildml.anomaly)",
        ],
        "torch_spec_present": torch_spec_available(),
        "industry_extra_present": semisupervised_industry_available(),
        "xgboost_present": xgboost_available(),
        "lightgbm_present": lightgbm_available(),
        "hf_text_present": hf_text_available(),
    }


def _default_backend_when_installed() -> str:
    if torch_spec_available():
        return "torch"
    if semisupervised_industry_available():
        return "industry"
    return "sklearn"


def _default_method_when_installed() -> str:
    if xgboost_available():
        return "pseudo_label_xgb"
    if torch_spec_available():
        return "fixmatch_tabular"
    return "label_propagation"


def list_semisupervised_methods(
    *,
    backend: SemiSupervisedBackendName | None = None,
) -> list[str]:
    """List semi-supervised methods for a backend (or all when backend is None)."""
    matrix = semisupervised_capability_matrix()
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None:
            return []
        if not entry.get("available"):
            return []
        return list(entry.get("methods") or [])
    methods: list[str] = []
    for entry in matrix["backends"].values():
        if not entry.get("available"):
            continue
        for method in entry.get("methods") or []:
            if method not in methods:
                methods.append(method)
    return methods


def backend_available(name: SemiSupervisedBackendName) -> bool:
    matrix = semisupervised_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: SemiSupervisedBackendName | None,
    method: str,
) -> tuple[SemiSupervisedBackendName, str]:
    """Validate backend/method pairing and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_backend: SemiSupervisedBackendName
    if backend is None:
        if method in {"label_propagation", "label_spreading", "self_training"}:
            resolved_backend = "sklearn"
        elif method in {"pseudo_label_xgb", "pseudo_label_lgbm"}:
            resolved_backend = "industry"
        elif method in {"fixmatch_tabular", "mixmatch_tabular"}:
            resolved_backend = "torch"
        elif method == "text_pseudo_label":
            resolved_backend = "hf"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
            allowed_default = list_semisupervised_methods(backend=resolved_backend)
            if allowed_default:
                method = allowed_default[0]
    else:
        resolved_backend = backend

    allowed = list_semisupervised_methods(backend=resolved_backend)
    if method not in allowed:
        raise ValidationError(
            f"method='{method}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = semisupervised_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "semisupervised-industry"), f"backend='{resolved_backend}'")
    return resolved_backend, method
