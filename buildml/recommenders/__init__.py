"""Recommendation systems (Session-shaped collaborative filtering path).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised → ensembles → AutoML → forecasting → anomaly …
Phase 2 (**complete** through TDA).

Phase 3 — Application systems:
  **Recommendation systems (this module)** — PASS.
  Search / LTR — **PASS**. Knowledge graphs — **PASS**.
  Optimisation / decision helpers — **PASS** (``buildml.optimize``).
  Synthetic-data systems — **PASS** (``buildml.synthetic``).
  Residual deepenings: NLP/CV vs existing Torch multimodal/speech/vision hooks.

Honesty (this package):
  - User/item/interaction tables with explicit ``user_column`` / ``item_column``.
  - Train-only fit; known-item protocol on holdout; cold-start disclosed.
  - Algorithms: item/user kNN CF, TruncatedSVD / NMF factorization, optional
    content-based scoring from numeric item features (numpy/sklearn).
  - Ranking metrics: Precision@K, Recall@K, nDCG@K, MAP@K.
  - **Not** a Netflix-scale recsys platform, **not** RAG retrieve/generate,
    **not** diagnostic EDA ``Recommendation`` Finding objects.

Dependency policy: core stays numpy/pandas/sklearn. No surprise/implicit
required for the Session path (optional extras reserved if justified later).

Lazy imports — keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ColdStartPolicy",
    "FeedbackMode",
    "RecommendResult",
    "RecommenderConfig",
    "RecommenderEvalResult",
    "RecommenderFitResult",
    "RecommenderMethod",
    "RecommenderPlan",
    "evaluate_recommender",
    "fit_recommender",
    "load_recommender_bundle",
    "recommend",
    "recommender_status",
    "recommender_status_for_session",
    "save_recommender_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "RecommenderConfig",
        "RecommenderMethod",
        "FeedbackMode",
        "ColdStartPolicy",
    }:
        from buildml.recommenders import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "RecommenderPlan",
        "RecommenderFitResult",
        "RecommendResult",
        "RecommenderEvalResult",
    }:
        from buildml.recommenders import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_recommender":
        from buildml.recommenders.fit import fit_recommender

        return fit_recommender
    if name == "recommend":
        from buildml.recommenders.recommend import recommend

        return recommend
    if name == "evaluate_recommender":
        from buildml.recommenders.evaluate import evaluate_recommender

        return evaluate_recommender
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_recommender_bundle",
        "load_recommender_bundle",
    }:
        from buildml.recommenders import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"recommender_status", "recommender_status_for_session"}:
        from buildml.recommenders import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.recommenders' has no attribute {name!r}")
