"""Recommendation systems (Session-shaped collaborative filtering path).

Behavior and limitations:
  - User/item/interaction tables with explicit ``user_column`` / ``item_column``.
  - Train-only fit; known-item protocol on holdout; cold-start disclosed.
  - Core: item/user kNN CF, TruncatedSVD / NMF, content-based (numpy/sklearn).
  - Optional ``buildml[recommenders-industry]``: implicit ALS/BPR; ALS is
    selected for implicit feedback when that backend is available.
  - Optional ``buildml[recommenders-lightfm]``: LightFM with side features.
  - Ranking metrics: Precision@K, Recall@K, nDCG@K, MAP@K.
  - **Not** a Netflix-scale recsys platform, **not** RAG retrieve/generate,
    **not** diagnostic EDA ``Recommendation`` Finding objects.

Dependency policy: core stays numpy/pandas/sklearn. Industry backends are
optional: ``recommenders-industry`` installs implicit;
``recommenders-lightfm`` installs LightFM separately.

Optional dependencies are imported only when their backends are used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ColdStartPolicy",
    "FeedbackMode",
    "RecommendResult",
    "RecommenderBackend",
    "RecommenderConfig",
    "RecommenderEvalResult",
    "RecommenderFitResult",
    "RecommenderMethod",
    "RecommenderPlan",
    "evaluate_recommender",
    "fit_recommender",
    "list_recommender_methods",
    "load_recommender_bundle",
    "recommend",
    "recommender_capability_matrix",
    "recommender_status",
    "recommender_status_for_session",
    "resolve_backend_method",
    "save_recommender_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "RecommenderConfig",
        "RecommenderMethod",
        "RecommenderBackend",
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
        "recommender_capability_matrix",
        "list_recommender_methods",
        "resolve_backend_method",
    }:
        from buildml.recommenders import catalog as catalog_mod

        return getattr(catalog_mod, name)
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
