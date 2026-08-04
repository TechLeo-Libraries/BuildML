"""Multi-task catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_available, torch_spec_available
from buildml.multitask.extras import (
    catboost_available,
    catboost_spec_present,
    lightgbm_available,
    lightgbm_spec_present,
    multitask_industry_available,
    xgboost_available,
    xgboost_spec_present,
)

MultiTaskBackendName = Literal["sklearn", "industry", "torch"]


def multitask_capability_matrix() -> dict[str, Any]:
    """Build the honest capability matrix for multi-task backends and methods.

    Reports sklearn, industry, and torch paths, evaluation metrics, install
    hints, and explicit non-goals for teaching overlays and Session walkthroughs.

    Returns
    -------
    dict[str, Any]
        Nested backend entries, evaluation rules, and defaults.
    """
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": [
                    "multi_output",
                    "classifier_chain",
                    "regressor_chain",
                ],
                "modality": "tabular",
                "notes": (
                    "Core sklearn MultiOutput / Chain façades; always available. "
                    "Same-type targets only (all classification or all regression)."
                ),
            },
            "industry": {
                "available": multitask_industry_available(),
                "extra": "multitask-industry",
                "methods": [
                    "multi_output_xgb",
                    "multi_output_lgbm",
                    "multi_output_catboost",
                ],
                "modality": "tabular",
                "notes": (
                    "XGBoost/LightGBM/CatBoost multi-target estimators when installed. "
                    "Regression uses native multi-target where supported; classification "
                    "uses honest MultiOutputClassifier wrappers. Same-type targets only."
                ),
            },
            "torch": {
                "available": torch_available(),
                "extra": "torch",
                "methods": ["shared_trunk_multihead"],
                "modality": "tabular",
                "notes": (
                    "Shared MLP trunk with per-task heads and joint training "
                    "(buildml[torch]). Supports mixed classification+regression "
                    "targets via separate heads: not a research MTL platform."
                ),
            },
        },
        "evaluation": {
            "metrics_classification": [
                "accuracy",
                "f1_macro",
                "f1_weighted",
            ],
            "metrics_regression": ["mae", "rmse", "r2"],
            "aggregates": (
                "Unweighted means across tasks; mixed torch plans report "
                "cls/reg aggregates separately."
            ),
            "holdout_rule": "validation/test never used for fitting",
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_method_when_installed": _default_method_when_installed(),
        "install_hints": {
            "multitask-industry": (
                "pip install 'buildml[multitask-industry]'  "
                "# XGBoost/LightGBM/CatBoost multi-target paths"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# shared-trunk multi-head joint training"
            ),
        },
        "non_goals": [
            "Universal deep MTL research platform or task-affinity search",
            "Multi-label binary-relevance zoos",
            "Causal or federated multi-task",
            "ClassifierChain/RegressorChain on industry GBDT backends",
        ],
        "torch_spec_present": torch_spec_available(),
        "industry_extra_present": (
            lightgbm_spec_present()
            or xgboost_spec_present()
            or catboost_spec_present()
        ),
        "industry_runtime_present": multitask_industry_available(),
        "xgboost_present": xgboost_available(),
        "lightgbm_present": lightgbm_available(),
        "catboost_present": catboost_available(),
        "industry_import_honesty": (
            "industry backend 'available' and industry_runtime_present require "
            "successful subprocess imports of at least one GBDT library. "
            "industry_extra_present / *_spec_present are find_spec only."
        ),
    }


def _default_backend_when_installed() -> str:
    if multitask_industry_available():
        return "industry"
    if torch_available():
        return "torch"
    return "sklearn"


def _default_method_when_installed() -> str:
    if xgboost_available():
        return "multi_output_xgb"
    if lightgbm_available():
        return "multi_output_lgbm"
    if torch_available():
        return "shared_trunk_multihead"
    return "multi_output"


def list_multitask_methods(
    *,
    backend: MultiTaskBackendName | None = None,
) -> list[str]:
    """List multi-task method names available for one or all backends.

    Filters to backends that are actually installed when ``backend`` is omitted.

    Parameters
    ----------
    backend:
        Optional backend name; when set, returns methods only if that backend
        is available.

    Returns
    -------
    list[str]
        Sorted unique method identifiers (e.g. ``multi_output``, ``shared_trunk_multihead``).
    """
    matrix = multitask_capability_matrix()
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


def backend_available(name: MultiTaskBackendName) -> bool:
    """Return whether a multi-task backend is installed and usable.

    Consults :func:`multitask_capability_matrix` rather than probing imports
    directly so availability matches teaching disclosures.

    Parameters
    ----------
    name:
        Backend identifier: ``sklearn``, ``industry``, or ``torch``.

    Returns
    -------
    bool
        ``True`` when the capability matrix marks the backend as available.
    """
    matrix = multitask_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: MultiTaskBackendName | None,
    method: str,
) -> tuple[MultiTaskBackendName, str]:
    """Validate backend/method pairing and apply honest defaults.

    Normalizes method aliases, infers backend when omitted, and raises when the
    requested pair requires a missing extra.

    Parameters
    ----------
    backend:
        Optional backend override; when ``None``, inferred from ``method``.
    method:
        Multi-task method name (case-sensitive catalog identifier).

    Returns
    -------
    tuple[MultiTaskBackendName, str]
        Resolved ``(backend, method)`` pair ready for fit routing.

    Raises
    ------
    ValidationError
        When the method is unknown or incompatible with the backend.
    MissingExtraError
        When the resolved backend requires an optional extra that is not installed.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_backend: MultiTaskBackendName
    if backend is None:
        if method in {
            "multi_output",
            "classifier_chain",
            "regressor_chain",
        }:
            resolved_backend = "sklearn"
        elif method in {
            "multi_output_xgb",
            "multi_output_lgbm",
            "multi_output_catboost",
        }:
            resolved_backend = "industry"
        elif method == "shared_trunk_multihead":
            resolved_backend = "torch"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
            allowed_default = list_multitask_methods(backend=resolved_backend)
            if allowed_default:
                method = allowed_default[0]
    else:
        resolved_backend = backend

    allowed = list_multitask_methods(backend=resolved_backend)
    if method not in allowed:
        raise ValidationError(
            f"method='{method}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = multitask_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "multitask-industry"), f"backend='{resolved_backend}'")
    return resolved_backend, method
