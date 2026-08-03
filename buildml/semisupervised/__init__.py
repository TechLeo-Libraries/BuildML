"""Semi-supervised learning domain (scarce labels + unlabeled train rows).

Industry depth (R6.1):
  - Core sklearn: LabelPropagation, LabelSpreading, SelfTrainingClassifier.
  - Industry (``buildml[semisupervised-industry]``): XGBoost/LightGBM pseudo-label.
  - Torch (``buildml[torch]``): FixMatch/MixMatch-style tabular consistency.
  - HF text (``buildml[ssl]``): sentence-transformer embeddings + pseudo-label.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Industry GBDT,
Torch consistency, and HF text paths use optional extras. Sklearn remains the
honest fallback when extras are missing.

Lazy imports: core never grows heavy semi-supervised stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "SemiSupervisedBackend",
    "SemiSupervisedConfig",
    "SemiSupervisedEvalResult",
    "SemiSupervisedFitResult",
    "SemiSupervisedMethod",
    "SemiSupervisedPlan",
    "SemiSupervisedPredictResult",
    "evaluate_semisupervised",
    "fit_semisupervised",
    "list_semisupervised_methods",
    "load_semisupervised_bundle",
    "predict_semisupervised",
    "save_semisupervised_bundle",
    "semisupervised_capability_matrix",
    "semisupervised_status",
    "semisupervised_status_for_session",
]


def __getattr__(name: str) -> Any:
    if name in {
        "SemiSupervisedMethod",
        "SemiSupervisedBackend",
        "SemiSupervisedConfig",
    }:
        from buildml.semisupervised import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "SemiSupervisedPlan",
        "SemiSupervisedFitResult",
        "SemiSupervisedPredictResult",
        "SemiSupervisedEvalResult",
    }:
        from buildml.semisupervised import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_semisupervised":
        from buildml.semisupervised.fit import fit_semisupervised

        return fit_semisupervised
    if name == "predict_semisupervised":
        from buildml.semisupervised.predict import predict_semisupervised

        return predict_semisupervised
    if name == "evaluate_semisupervised":
        from buildml.semisupervised.evaluate import evaluate_semisupervised

        return evaluate_semisupervised
    if name in {"semisupervised_capability_matrix", "list_semisupervised_methods"}:
        from buildml.semisupervised import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_semisupervised_bundle",
        "load_semisupervised_bundle",
    }:
        from buildml.semisupervised import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"semisupervised_status", "semisupervised_status_for_session"}:
        from buildml.semisupervised import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.semisupervised' has no attribute {name!r}")
