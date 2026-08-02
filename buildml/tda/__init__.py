"""Topological Data Analysis (Session-shaped persistent homology path).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly.

Phase 2 (**complete** through this module / TDA).

Phase 3 — Application systems (depth-first):
  Recommendation systems → search/LTR → knowledge graphs →
  optimisation/decision helpers → synthetic-data systems.
  Remaining deepenings: NLP/CV if still partial.

Honesty (this package):
  - Local Vietoris–Rips persistence (ripser) on kNN train neighborhoods.
  - Train-fitted vectorization: persistence images (persim) or in-tree
    landscapes / silhouettes → fixed-length features.
  - Optional sklearn classify/regress head fitted on **train** topological
    features only; holdout uses the frozen transformer (no refit).
  - **Not** a full Mapper research suite, not every TDA paper, not a
    domain-specific credit-risk product (tabular use is fine motivation).

Dependency policy: core stays numpy/pandas/sklearn.
  - ``buildml[tda]`` → ``ripser`` + ``persim`` (chosen over giotto-tda for a
    lighter, well-integrated PH → vectorization stack without pulling a
    heavier transitive tree). Silhouette vectorization is in-tree.
  - ``import buildml`` never requires ripser/persim.

Lazy imports — keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "TdaConfig",
    "TdaEvalResult",
    "TdaFitResult",
    "TdaHead",
    "TdaPlan",
    "TdaPredictResult",
    "TdaTask",
    "TdaTransformResult",
    "Vectorization",
    "evaluate_tda",
    "fit_tda",
    "load_tda_bundle",
    "predict_tda",
    "require_persim",
    "require_ripser",
    "require_tda_stack",
    "save_tda_bundle",
    "tda_available",
    "tda_status",
    "tda_status_for_session",
    "transform_tda",
]


def __getattr__(name: str) -> Any:
    if name in {"TdaConfig", "Vectorization", "TdaTask", "TdaHead"}:
        from buildml.tda import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "TdaPlan",
        "TdaFitResult",
        "TdaTransformResult",
        "TdaPredictResult",
        "TdaEvalResult",
    }:
        from buildml.tda import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_tda":
        from buildml.tda.fit import fit_tda

        return fit_tda
    if name == "transform_tda":
        from buildml.tda.transform import transform_tda

        return transform_tda
    if name == "predict_tda":
        from buildml.tda.predict import predict_tda

        return predict_tda
    if name == "evaluate_tda":
        from buildml.tda.evaluate import evaluate_tda

        return evaluate_tda
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_tda_bundle",
        "load_tda_bundle",
    }:
        from buildml.tda import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {
        "require_ripser",
        "require_persim",
        "require_tda_stack",
        "tda_available",
    }:
        from buildml.tda import extras as extras_mod

        return getattr(extras_mod, name)
    if name in {"tda_status", "tda_status_for_session"}:
        from buildml.tda import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.tda' has no attribute {name!r}")
