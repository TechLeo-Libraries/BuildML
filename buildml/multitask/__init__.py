"""Multi-task / multi-output learning domain (sklearn + industry + torch).

Industry depth (R6.4):
  - Core sklearn: MultiOutput / Chain façades (always available).
  - Industry (``buildml[multitask-industry]``): XGBoost/LightGBM/CatBoost
    multi-target when installed.
  - Torch (``buildml[torch]``): shared-trunk multi-head joint training; mixed
    classification+regression via separate heads.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Industry GBDT and
Torch multi-head paths use optional extras. Sklearn remains the honest fallback
when extras are missing.

Lazy imports: core never grows heavy MTL stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "MultiTaskBackend",
    "MultiTaskBaseEstimator",
    "MultiTaskConfig",
    "MultiTaskEvalResult",
    "MultiTaskFitResult",
    "MultiTaskMethod",
    "MultiTaskPlan",
    "MultiTaskPredictResult",
    "MultiTaskTask",
    "evaluate_multitask",
    "fit_multitask",
    "list_multitask_methods",
    "load_multitask_bundle",
    "multitask_capability_matrix",
    "multitask_status",
    "multitask_status_for_session",
    "predict_multitask",
    "save_multitask_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "MultiTaskMethod",
        "MultiTaskBackend",
        "MultiTaskTask",
        "MultiTaskBaseEstimator",
        "MultiTaskConfig",
    }:
        from buildml.multitask import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "MultiTaskPlan",
        "MultiTaskFitResult",
        "MultiTaskEvalResult",
        "MultiTaskPredictResult",
    }:
        from buildml.multitask import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_multitask":
        from buildml.multitask.fit import fit_multitask

        return fit_multitask
    if name == "predict_multitask":
        from buildml.multitask.predict import predict_multitask

        return predict_multitask
    if name == "evaluate_multitask":
        from buildml.multitask.evaluate import evaluate_multitask

        return evaluate_multitask
    if name in {"multitask_capability_matrix", "list_multitask_methods"}:
        from buildml.multitask import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_multitask_bundle",
        "load_multitask_bundle",
    }:
        from buildml.multitask import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"multitask_status", "multitask_status_for_session"}:
        from buildml.multitask import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.multitask' has no attribute {name!r}")
