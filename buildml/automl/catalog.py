"""AutoML method catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.automl.extras import (
    autogluon_available,
    automl_industry_available,
    catboost_available,
    flaml_available,
    gradient_boosting_extras_available,
    lightgbm_available,
    optuna_available,
    xgboost_available,
)

AutoMLBackendName = Literal["native", "optuna", "flaml", "autogluon"]
NativeMethodName = Literal["grid", "randomized", "optuna", "evolutionary"]


def automl_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for AutoML backends and optional extras.

    Returns a JSON-serializable dict suitable for docs, walkthrough, and AI tools.
    """
    gbdt = {
        "lightgbm": lightgbm_available(),
        "xgboost": xgboost_available(),
        "catboost": catboost_available(),
    }
    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "methods": ["grid", "randomized", "optuna", "evolutionary"],
                "fold_local_recipes": True,
                "industry_gbdt_families": gradient_boosting_extras_available(),
                "ensemble_voting": True,
                "ensemble_stacking": True,
                "nested_cv": True,
                "validation_selection": True,
                "time_budget": True,
                "trial_budget": True,
                "study_persistence": False,
                "multi_objective": False,
                "pruning": False,
            },
            "optuna": {
                "available": optuna_available(),
                "extra": "automl",
                "methods": ["optuna"],
                "fold_local_recipes": True,
                "industry_gbdt_families": gradient_boosting_extras_available(),
                "ensemble_voting": True,
                "ensemble_stacking": True,
                "nested_cv": True,
                "validation_selection": True,
                "time_budget": True,
                "trial_budget": True,
                "study_persistence": optuna_available(),
                "multi_objective": optuna_available(),
                "pruning": optuna_available(),
            },
            "flaml": {
                "available": flaml_available(),
                "extra": "automl-industry",
                "methods": ["flaml"],
                "fold_local_recipes": False,
                "industry_gbdt_families": True,
                "ensemble_voting": False,
                "ensemble_stacking": False,
                "nested_cv": False,
                "validation_selection": True,
                "time_budget": True,
                "trial_budget": False,
                "study_persistence": False,
                "multi_objective": False,
                "pruning": False,
                "notes": (
                    "FLAML runs internal model selection on train-only data; "
                    "fold-local PreprocessRecipe search is bypassed. "
                    "Session test never enters FLAML fit."
                ),
            },
            "autogluon": {
                "available": autogluon_available(),
                "extra": "automl-industry",
                "methods": ["autogluon"],
                "fold_local_recipes": False,
                "industry_gbdt_families": True,
                "ensemble_voting": False,
                "ensemble_stacking": False,
                "nested_cv": False,
                "validation_selection": True,
                "time_budget": True,
                "trial_budget": False,
                "study_persistence": False,
                "multi_objective": False,
                "pruning": False,
                "notes": (
                    "AutoGluon TabularPredictor runs internal stacking on train-only "
                    "data; fold-local recipe search is bypassed. "
                    "Session test never enters fit."
                ),
            },
        },
        "optional_gbdt_families": gbdt,
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "automl": "pip install 'buildml[automl]'  # Optuna-backed search",
            "automl-industry": (
                "pip install 'buildml[automl-industry]'  "
                "# FLAML and/or AutoGluon + LightGBM/XGBoost/CatBoost families"
            ),
        },
        "non_goals": [
            "Neural architecture search (NAS)",
            "Causal discovery",
            "Fully automated AI scientist",
            "Session-global preprocess as safe CV",
        ],
    }


def _default_backend_when_installed() -> str:
    """Suggest the richest installed backend (honest, not magic)."""
    if flaml_available():
        return "flaml"
    if autogluon_available():
        return "autogluon"
    if optuna_available():
        return "optuna"
    return "native"


def list_automl_methods(*, backend: AutoMLBackendName | None = None) -> list[str]:
    """List search methods available for a backend (or all when backend is None)."""
    matrix = automl_capability_matrix()["backends"]
    if backend is not None:
        entry = matrix.get(backend)
        if entry is None:
            return []
        return list(entry.get("methods") or [])
    out: list[str] = []
    for entry in matrix.values():
        for method in entry.get("methods") or []:
            if method not in out:
                out.append(method)
    return out


def backend_available(name: AutoMLBackendName) -> bool:
    matrix = automl_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))
