"""Anomaly detector catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.anomaly.extras import (
    anomaly_industry_available,
    gradient_boosting_extras_available,
    lightgbm_available,
    pyod_available,
    xgboost_available,
)
from buildml.dl.extras import torch_available, torch_spec_available

AnomalyBackendName = Literal["sklearn", "pyod", "torch"]


def anomaly_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for anomaly backends and optional extras."""
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": ["isolation_forest", "lof", "one_class_svm"],
                "modes": ["unsupervised", "novelty"],
                "supervised_methods": ["supervised_hgb"],
                "score_calibration": "sklearn score_samples / decision_function inverted",
            },
            "pyod": {
                "available": pyod_available(),
                "extra": "anomaly-industry",
                "methods": ["hbos", "copod", "ecod", "deepsvdd"],
                "modes": ["unsupervised", "novelty"],
                "supervised_methods": [],
                "score_calibration": "PyOD decision_function (higher = more anomalous)",
                "notes": (
                    "deepsvdd may require torch inside PyOD; install buildml[torch] "
                    "when DeepSVDD training fails on CPU-only stacks."
                ),
            },
            "torch": {
                "available": torch_available(),
                "extra": "torch",
                "methods": ["autoencoder"],
                "modes": ["unsupervised", "novelty"],
                "supervised_methods": [],
                "score_calibration": "train-only MSE reconstruction error (higher = more anomalous)",
            },
        },
        "supervised_scorers": {
            "supervised_hgb": {"available": True, "extra": None},
            "supervised_xgb": {
                "available": xgboost_available(),
                "extra": "anomaly-industry",
            },
            "supervised_lgbm": {
                "available": lightgbm_available(),
                "extra": "anomaly-industry",
            },
        },
        "evaluation_metrics": [
            "average_precision",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "precision_at_k",
            "recall_at_k",
        ],
        "threshold_policies": [
            "contamination",
            "quantile",
            "score_threshold",
            "decision_zero",
            "validation_tuned",
        ],
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_supervised_when_installed": _default_supervised_when_installed(),
        "install_hints": {
            "anomaly-industry": (
                "pip install 'buildml[anomaly-industry]'  "
                "# PyOD (HBOS/COPOD/ECOD/DeepSVDD) + XGBoost/LightGBM fraud scorers"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# tabular autoencoder reconstruction-error anomaly path"
            ),
        },
        "non_goals": [
            "Graph fraud / entity networks",
            "Online streaming anomaly product",
            "Causal fraud attribution",
            "Full PyOD algorithm zoo beyond catalog methods",
        ],
        "torch_spec_present": torch_spec_available(),
        "torch_import_honesty": (
            "torch backend 'available' uses a real import probe (torch_available). "
            "torch_spec_present is the cheap find_spec signal only."
        ),
        "industry_extra_present": anomaly_industry_available(),
    }


def _default_backend_when_installed() -> str:
    if pyod_available():
        return "pyod"
    if torch_available():
        return "torch"
    return "sklearn"


def _default_supervised_when_installed() -> str:
    if xgboost_available():
        return "supervised_xgb"
    if lightgbm_available():
        return "supervised_lgbm"
    return "supervised_hgb"


def list_anomaly_methods(
    *,
    backend: AnomalyBackendName | None = None,
    mode: str | None = None,
) -> list[str]:
    """List detector methods for a backend (or all when backend is None)."""
    matrix = anomaly_capability_matrix()
    if mode == "supervised":
        out: list[str] = []
        for name, entry in matrix["supervised_scorers"].items():
            if entry.get("available"):
                out.append(name)
        return out
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None:
            return []
        return list(entry.get("methods") or [])
    methods: list[str] = []
    for entry in matrix["backends"].values():
        for method in entry.get("methods") or []:
            if method not in methods:
                methods.append(method)
    return methods


def backend_available(name: AnomalyBackendName) -> bool:
    matrix = anomaly_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: AnomalyBackendName | None,
    method: str,
    mode: str,
) -> tuple[AnomalyBackendName, str]:
    """Validate backend/method pairing and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    if mode == "supervised":
        matrix = anomaly_capability_matrix()["supervised_scorers"]
        entry = matrix.get(method)
        if entry is None:
            raise ValidationError(
                f"Unknown supervised anomaly method '{method}'. "
                f"Choose from {sorted(matrix)}."
            )
        if not entry.get("available"):
            raise MissingExtraError(
                str(entry.get("extra") or "anomaly-industry"),
                f"supervised anomaly method '{method}'",
            )
        return "sklearn", method

    resolved_backend: AnomalyBackendName
    if backend is None:
        if method in {"isolation_forest", "lof", "one_class_svm"}:
            resolved_backend = "sklearn"
        elif method in {"hbos", "copod", "ecod", "deepsvdd"}:
            resolved_backend = "pyod"
        elif method == "autoencoder":
            resolved_backend = "torch"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
    else:
        resolved_backend = backend

    allowed = list_anomaly_methods(backend=resolved_backend)
    if method not in allowed:
        raise ValidationError(
            f"method='{method}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = anomaly_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "anomaly-industry"), f"backend='{resolved_backend}'")
    return resolved_backend, method
