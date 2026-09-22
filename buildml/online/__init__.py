"""Online / continual learning domain (sklearn ``partial_fit`` + industry backends).

Behavior and limitations:
  - Batch/stream-chunk ``partial_fit`` updates on Session train data (or
    role-aligned user frames): NOT a distributed streaming platform and NOT a
    full lifelong-learning research suite.
  - Validation/test are never used for updates.
  - Full refits during an incremental update are refused by default; optional
    ``allow_refit_fallback`` is always disclosed.
  - Classifiers require a ``classes`` vocabulary on first fit (explicit or
    discovered from the full train target column: labels only).

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Sklearn
``partial_fit`` is the default path. River streaming + drift detectors require
``buildml[online-industry]``; torch replay/EWC continual paths require
``buildml[torch]``.

Lazy imports: core never grows heavy streaming stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "OnlineConfig",
    "OnlineEstimator",
    "OnlineEvalResult",
    "OnlineFitResult",
    "OnlinePlan",
    "OnlinePredictResult",
    "OnlineTask",
    "OnlineUpdateResult",
    "evaluate_online",
    "fit_online",
    "list_online_estimators",
    "load_online_bundle",
    "online_capability_matrix",
    "online_status",
    "online_status_for_session",
    "partial_fit",
    "partial_fit_online",
    "predict_online",
    "save_online_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "OnlineEstimator",
        "OnlineTask",
        "OnlineConfig",
    }:
        from buildml.online import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "OnlinePlan",
        "OnlineFitResult",
        "OnlineUpdateResult",
        "OnlineEvalResult",
        "OnlinePredictResult",
    }:
        from buildml.online import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_online":
        from buildml.online.fit import fit_online

        return fit_online
    if name in {"partial_fit_online", "partial_fit"}:
        from buildml.online import update as update_mod

        return getattr(update_mod, name)
    if name == "evaluate_online":
        from buildml.online.evaluate import evaluate_online

        return evaluate_online
    if name == "predict_online":
        from buildml.online.predict import predict_online

        return predict_online
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_online_bundle",
        "load_online_bundle",
    }:
        from buildml.online import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"online_status", "online_status_for_session"}:
        from buildml.online import explain_hooks as hooks

        return getattr(hooks, name)
    if name == "online_capability_matrix":
        from buildml.online.catalog import online_capability_matrix

        return online_capability_matrix
    if name == "list_online_estimators":
        from buildml.online.catalog import list_online_estimators

        return list_online_estimators
    raise AttributeError(f"module 'buildml.online' has no attribute {name!r}")
