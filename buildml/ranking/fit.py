"""Fit tabular learning-to-rank models on Session train rows only."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.ranking.features import (
    disclose_query_split,
    feature_matrix,
    resolve_ranking_columns,
    standardize_fit,
    train_partition_frame,
)
from buildml.ranking.models import fit_pairwise_ranksvm, fit_pointwise
from buildml.ranking.results import RankerFitResult, RankerPlan
from buildml.ranking.types import (
    PairwiseEstimator,
    PointwiseEstimator,
    RankerConfig,
    RankerMethod,
)


def fit_ranker(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: RankerMethod = "pointwise",
    query_column: str | None = None,
    item_column: str | None = None,
    relevance_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    pointwise_estimator: PointwiseEstimator = "ridge",
    pairwise_estimator: PairwiseEstimator = "ranksvm",
    max_pairs_per_query: int = 80,
    relevance_threshold: float = 0.0,
    alpha: float = 1.0,
    C: float = 1.0,
    random_state: int | None = 0,
) -> tuple[RankerPlan, RankerFitResult]:
    """Fit a leakage-safe tabular ranker on the Session **train** partition.

    Pipeline
    --------
    1. Resolve query / item / relevance / feature columns.
    2. Disclose query-group split honesty (prefer ``group_split``).
    3. Standardize features on train only.
    4. Fit pointwise relevance regression or pairwise RankSVM-lite.

    Honesty: Session tabular LTR — not a search-engine product, not RAG
    retrieve/generate, not recommender CF. Never trains on holdout rows.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    if method not in {"pointwise", "pairwise"}:
        raise ValidationError(f"Unknown ranker method: {method!r}")
    if int(max_pairs_per_query) < 1:
        raise ValidationError("max_pairs_per_query must be >= 1.")
    if pointwise_estimator not in {"ridge", "hgb"}:
        raise ValidationError(
            "pointwise_estimator must be 'ridge' or 'hgb'."
        )
    if pairwise_estimator not in {"ranksvm"}:
        raise ValidationError("pairwise_estimator must be 'ranksvm'.")

    query_col, item_col, rel_col, feat_cols, disclosures = resolve_ranking_columns(
        dataset,
        query_column=query_column,
        item_column=item_column,
        relevance_column=relevance_column,
        feature_columns=feature_columns,
    )
    warnings: list[str] = []

    group_ok, split_disc, split_warn = disclose_query_split(
        dataset, split_plan, query_col
    )
    disclosures.extend(split_disc)
    warnings.extend(split_warn)

    train = train_partition_frame(dataset, split_plan)
    if len(train) < 4:
        raise ValidationError(
            f"Need ≥4 train rows for LTR; got {len(train)}."
        )
    X_raw = feature_matrix(train, feat_cols)
    if not np.isfinite(X_raw).all():
        raise ValidationError(
            "LTR features contain NaN/Inf on train; impute/clean before fit_ranker."
        )
    y = train[rel_col].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValidationError(
            "relevance labels contain NaN/Inf on train."
        )
    groups = train[query_col].to_numpy()
    n_queries = int(len(np.unique(groups)))
    if n_queries < 2:
        raise ValidationError(
            f"Need ≥2 distinct train queries; got {n_queries}."
        )

    X, mean, scale = standardize_fit(X_raw)
    estimator = None
    coef = None
    intercept = 0.0
    n_pair_examples: int | None = None

    if method == "pointwise":
        estimator = fit_pointwise(
            X,
            y,
            estimator=pointwise_estimator,
            alpha=alpha,
            random_state=random_state,
        )
        if hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float).ravel()
            intercept = float(getattr(estimator, "intercept_", 0.0))
            if np.ndim(estimator.intercept_) > 0:
                intercept = float(np.asarray(estimator.intercept_).ravel()[0])
        disclosures.append(
            f"pointwise: {pointwise_estimator} regresses graded relevance on "
            f"{len(feat_cols)} standardized train features "
            f"({n_queries} queries, {len(train)} rows)."
        )
    else:
        estimator, coef, intercept, n_pair_examples = fit_pairwise_ranksvm(
            X,
            y,
            groups,
            C=C,
            max_pairs_per_query=int(max_pairs_per_query),
            random_state=random_state,
        )
        disclosures.append(
            f"pairwise: RankSVM-lite (LinearSVC) on within-query feature "
            f"differences; {n_pair_examples} oriented pair(s) from train "
            f"(max_pairs_per_query={max_pairs_per_query})."
        )

    disclosures.append(
        "Inference scores items within a query from frozen train features; "
        "holdout relevance labels are never used at fit."
    )

    config = RankerConfig(
        method=method,
        query_column=query_col,
        item_column=item_col,
        relevance_column=rel_col,
        feature_columns=feat_cols,
        pointwise_estimator=pointwise_estimator,
        pairwise_estimator=pairwise_estimator,
        max_pairs_per_query=int(max_pairs_per_query),
        relevance_threshold=float(relevance_threshold),
        random_state=random_state,
        alpha=float(alpha),
        C=float(C),
    )

    plan = RankerPlan(
        method=method,
        query_column=query_col,
        item_column=item_col,
        relevance_column=rel_col,
        feature_columns=feat_cols,
        pointwise_estimator=pointwise_estimator,
        pairwise_estimator=pairwise_estimator,
        n_train_rows=int(len(train)),
        n_train_queries=n_queries,
        n_features=len(feat_cols),
        feature_mean_=mean,
        feature_scale_=scale,
        estimator_=estimator,
        coef_=coef,
        intercept_=intercept,
        max_pairs_per_query=int(max_pairs_per_query),
        relevance_threshold=float(relevance_threshold),
        alpha=float(alpha),
        C=float(C),
        random_state=random_state,
        group_split_disclosed=group_ok,
        split_kind=str(split_plan.kind),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = RankerFitResult(
        method=method,
        n_train_rows=plan.n_train_rows,
        n_train_queries=plan.n_train_queries,
        n_features=plan.n_features,
        query_column=query_col,
        item_column=item_col,
        relevance_column=rel_col,
        feature_columns=feat_cols,
        pointwise_estimator=pointwise_estimator if method == "pointwise" else None,
        pairwise_estimator=pairwise_estimator if method == "pairwise" else None,
        n_pairwise_examples=n_pair_examples,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result
