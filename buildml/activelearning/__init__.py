"""Active learning domain (human-in-the-loop train-pool querying).

Industry depth (R6.2):
  - Core sklearn: uncertainty + bagging committee query strategies.
  - Industry (``buildml[activelearning-industry]``): scikit-activeml CoreSet / QBC.
  - Torch (``buildml[torch]``): BALD / MC-dropout tabular query strategies.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Industry and Torch
query paths use optional extras. Sklearn remains the honest fallback when extras
are missing.

Lazy imports: core never grows heavy active-learning stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ActiveLearningBackend",
    "ActiveLearningConfig",
    "ActiveLearningEstimator",
    "ActiveLearningEvalResult",
    "ActiveLearningFitResult",
    "ActiveLearningLabelResult",
    "ActiveLearningPlan",
    "ActiveLearningQueryResult",
    "ActiveLearningStrategy",
    "activelearning_capability_matrix",
    "activelearning_status",
    "activelearning_status_for_session",
    "evaluate_active_learning",
    "fit_active_learner",
    "label_rows",
    "list_activelearning_strategies",
    "load_active_learning_bundle",
    "query_indices",
    "save_active_learning_bundle",
    "suggest_query",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ActiveLearningStrategy",
        "ActiveLearningBackend",
        "ActiveLearningEstimator",
        "ActiveLearningConfig",
    }:
        from buildml.activelearning import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "ActiveLearningPlan",
        "ActiveLearningFitResult",
        "ActiveLearningQueryResult",
        "ActiveLearningLabelResult",
        "ActiveLearningEvalResult",
    }:
        from buildml.activelearning import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_active_learner":
        from buildml.activelearning.fit import fit_active_learner

        return fit_active_learner
    if name in {"suggest_query", "query_indices"}:
        from buildml.activelearning import query as query_mod

        return getattr(query_mod, name)
    if name == "label_rows":
        from buildml.activelearning.label import label_rows

        return label_rows
    if name == "evaluate_active_learning":
        from buildml.activelearning.evaluate import evaluate_active_learning

        return evaluate_active_learning
    if name in {"activelearning_capability_matrix", "list_activelearning_strategies"}:
        from buildml.activelearning import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_active_learning_bundle",
        "load_active_learning_bundle",
    }:
        from buildml.activelearning import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"activelearning_status", "activelearning_status_for_session"}:
        from buildml.activelearning import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.activelearning' has no attribute {name!r}")
