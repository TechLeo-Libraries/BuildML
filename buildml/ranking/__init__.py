"""Tabular learning-to-rank / search ranking (Session-shaped LTR path).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1–2 complete. Phase 3 — Application systems:
  Recommendation systems (**PASS**).
  **Search / learning-to-rank (this module)** — **PASS**.
  Knowledge graphs — **PASS**. Optimisation / decision helpers — **PASS**.
  Synthetic-data systems — **PASS** (``buildml.synthetic``).

Honesty (this package):
  - Tabular query–item (or query–document) feature rows with relevance labels.
  - Train-only fit; prefer ``group_split`` on ``query_column`` so test queries'
    labels never enter training.
  - Algorithms: pointwise relevance regression (Ridge / HistGradientBoosting)
    and pairwise RankSVM-lite (LinearSVC on within-query differences).
  - Ranking metrics: graded nDCG@K, MAP@K, MRR@K (macro over queries).
  - **Not** a search-engine product, **not** RAG retrieve/generate,
    **not** recommender user–item CF.

Dependency policy: core stays numpy/pandas/sklearn. LightGBM/XGBoost LambdaMART
is intentionally not required; optional extras only if a complete path is added
later.

Lazy imports — keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "PairwiseEstimator",
    "PointwiseEstimator",
    "RankerConfig",
    "RankerEvalResult",
    "RankerFitResult",
    "RankerMethod",
    "RankerPlan",
    "RankResult",
    "evaluate_ranker",
    "fit_ranker",
    "load_ranker_bundle",
    "rank",
    "ranking_status",
    "ranking_status_for_session",
    "save_ranker_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "RankerConfig",
        "RankerMethod",
        "PointwiseEstimator",
        "PairwiseEstimator",
    }:
        from buildml.ranking import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "RankerPlan",
        "RankerFitResult",
        "RankResult",
        "RankerEvalResult",
    }:
        from buildml.ranking import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_ranker":
        from buildml.ranking.fit import fit_ranker

        return fit_ranker
    if name == "rank":
        from buildml.ranking.rank import rank

        return rank
    if name == "evaluate_ranker":
        from buildml.ranking.evaluate import evaluate_ranker

        return evaluate_ranker
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_ranker_bundle",
        "load_ranker_bundle",
    }:
        from buildml.ranking import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"ranking_status", "ranking_status_for_session"}:
        from buildml.ranking import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.ranking' has no attribute {name!r}")
