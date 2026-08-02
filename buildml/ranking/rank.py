"""Score and order items per query with a frozen RankerPlan."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.ranking.adapters import (
    score_catboost,
    score_lgbm,
    score_listwise_lite,
    score_sklearn_ranker,
    score_xgb,
)
from buildml.ranking.adapters.sklearn import SklearnRankerState
from buildml.ranking.adapters.torch_listwise import ListwiseLiteRanker
from buildml.ranking.catalog import resolve_backend_method
from buildml.ranking.features import (
    feature_matrix,
    partition_frame,
    standardize_apply,
)
from buildml.ranking.models import score_linear, score_pointwise
from buildml.ranking.results import RankerPlan, RankResult
from buildml.ranking.types import RankerBackend


def _assert_backend_matches(plan: RankerPlan, backend: RankerBackend | None) -> None:
    if backend is None:
        return
    resolved_backend, _ = resolve_backend_method(backend=backend, method=plan.method)
    if resolved_backend != plan.backend:
        raise ValidationError(
            f"Requested backend={backend!r} does not match frozen plan "
            f"backend={plan.backend!r}."
        )


def score_rows(
    plan: RankerPlan,
    frame: pd.DataFrame,
    *,
    backend: RankerBackend | None = None,
) -> np.ndarray:
    """Score ranking rows with the frozen train ranker."""
    _assert_backend_matches(plan, backend)
    X_raw = feature_matrix(frame, plan.feature_columns)
    if not np.isfinite(X_raw).all():
        raise ValidationError(
            "LTR features contain NaN/Inf at score time; clean inputs first."
        )
    X = standardize_apply(X_raw, plan.feature_mean_, plan.feature_scale_)
    estimator = plan.estimator_
    method = plan.method
    plan_backend = plan.backend

    if plan_backend == "sklearn":
        if isinstance(estimator, SklearnRankerState):
            return score_sklearn_ranker(estimator, X)
        # Legacy bundles saved before backend routing
        if method == "pointwise":
            if estimator is None:
                raise ValidationError("RankerPlan missing pointwise estimator.")
            return score_pointwise(estimator, X)
        if plan.coef_ is None:
            raise ValidationError("RankerPlan missing pairwise coefficients.")
        return score_linear(plan.coef_, plan.intercept_, X)

    if method == "lambdarank_lgbm":
        return score_lgbm(estimator, X)
    if method == "rank_ndcg_xgb":
        return score_xgb(estimator, X)
    if method == "yetirank_catboost":
        return score_catboost(estimator, X)
    if method == "listwise_lite":
        if not isinstance(estimator, ListwiseLiteRanker):
            raise ValidationError("RankerPlan missing listwise-lite torch ranker.")
        return score_listwise_lite(estimator, X)

    raise ValidationError(
        f"Cannot score RankerPlan with backend={plan_backend!r}, method={method!r}."
    )


def rank_queries(
    plan: RankerPlan,
    frame: pd.DataFrame,
    *,
    query_ids: Sequence[Any] | None = None,
    k: int = 10,
    backend: RankerBackend | None = None,
) -> RankResult:
    """Order items within each query by descending score."""
    if int(k) < 1:
        raise ValidationError("k must be >= 1.")
    if frame.empty:
        return RankResult(
            k=int(k),
            n_queries=0,
            method=plan.method,
            query_ids=(),
            rankings=(),
            scores=(),
            n_candidates=(),
            disclosures=("No rows to rank.",),
            warnings=(),
        )

    qcol = plan.query_column
    icol = plan.item_column
    if query_ids is None:
        ordered_queries = list(dict.fromkeys(frame[qcol].tolist()))
    else:
        ordered_queries = list(query_ids)
        missing = [q for q in ordered_queries if q not in set(frame[qcol].tolist())]
        if missing:
            raise ValidationError(
                f"query_ids not present in the ranking frame: {missing[:5]}"
            )

    scores_all = score_rows(plan, frame, backend=backend)
    frame = frame.copy()
    frame["__score__"] = scores_all

    rankings: list[tuple[Any, ...]] = []
    score_lists: list[tuple[float, ...]] = []
    n_cands: list[int] = []
    for qid in ordered_queries:
        sub = frame.loc[frame[qcol] == qid]
        n_cands.append(int(len(sub)))
        if sub.empty:
            rankings.append(())
            score_lists.append(())
            continue
        ordered = sub.sort_values("__score__", ascending=False)
        top = ordered.head(int(k))
        rankings.append(tuple(top[icol].tolist()))
        score_lists.append(tuple(float(s) for s in top["__score__"].tolist()))

    disclosures = [
        f"Ranked {len(ordered_queries)} query(ies) with frozen "
        f"{plan.backend}/{plan.method} ranker.",
        "Scores use train-fitted standardization; relevance labels are not used "
        "at rank time.",
        "Tabular LTR is not RAG retrieve and not recommender CF.",
    ]
    return RankResult(
        k=int(k),
        n_queries=len(ordered_queries),
        method=plan.method,
        query_ids=tuple(ordered_queries),
        rankings=tuple(rankings),
        scores=tuple(score_lists),
        n_candidates=tuple(n_cands),
        disclosures=tuple(disclosures),
        warnings=(),
    )


def rank(
    dataset: Dataset,
    plan: RankerPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str | None = None,
    query_ids: Sequence[Any] | None = None,
    k: int = 10,
    backend: RankerBackend | None = None,
) -> RankResult:
    """Session-facing rank: order items for queries in a partition or id list."""
    if query_ids is None and partition is None:
        partition = "test"
    if partition is None:
        frame = dataset.frame.copy()
    else:
        frame = partition_frame(dataset, split_plan, partition)
    return rank_queries(
        plan,
        frame,
        query_ids=query_ids,
        k=k,
        backend=backend,
    )
