"""Meta-learning domain (tabular few-shot / episodic Session protocols).

Industry depth (R6.5):
  - Core sklearn: ``prototypical`` nearest-centroid + ``warm_start`` adapt.
  - Torch (``buildml[torch]``): ``prototypical_torch`` deep tabular encoder.
  - Industry (``buildml[metalearning-industry,torch]``): ``maml`` / ``reptile``
    via learn2learn first-order tabular adapters.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Torch and industry
paths use optional extras. Sklearn remains the honest fallback when extras
are missing.

Lazy imports — core never grows heavy meta-learning stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "MetaAdaptResult",
    "MetaLearningBackend",
    "MetaLearningBaseEstimator",
    "MetaLearningConfig",
    "MetaLearningEvalResult",
    "MetaLearningFitResult",
    "MetaLearningMethod",
    "MetaLearningPlan",
    "adapt_to_task",
    "evaluate_metalearning",
    "fit_metalearning",
    "list_metalearning_methods",
    "load_metalearning_bundle",
    "metalearning_capability_matrix",
    "metalearning_status",
    "metalearning_status_for_session",
    "save_metalearning_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "MetaLearningMethod",
        "MetaLearningBackend",
        "MetaLearningBaseEstimator",
        "MetaLearningConfig",
    }:
        from buildml.metalearning import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "MetaLearningPlan",
        "MetaLearningFitResult",
        "MetaLearningEvalResult",
        "MetaAdaptResult",
    }:
        from buildml.metalearning import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_metalearning":
        from buildml.metalearning.fit import fit_metalearning

        return fit_metalearning
    if name == "adapt_to_task":
        from buildml.metalearning.adapt import adapt_to_task

        return adapt_to_task
    if name == "evaluate_metalearning":
        from buildml.metalearning.evaluate import evaluate_metalearning

        return evaluate_metalearning
    if name in {"metalearning_capability_matrix", "list_metalearning_methods"}:
        from buildml.metalearning import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_metalearning_bundle",
        "load_metalearning_bundle",
    }:
        from buildml.metalearning import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"metalearning_status", "metalearning_status_for_session"}:
        from buildml.metalearning import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.metalearning' has no attribute {name!r}")
