"""Fit tabular learning-to-rank models on Session train rows only."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.ranking.adapters import (
    build_sklearn_ranker,
    fit_lambdarank_lgbm,
    fit_listwise_lite,
    fit_rank_ndcg_xgb,
    fit_yetirank_catboost,
)
from buildml.ranking.catalog import ranking_capability_matrix, resolve_backend_method
from buildml.ranking.features import (
    disclose_query_split,
    feature_matrix,
    resolve_ranking_columns,
    standardize_fit,
    train_partition_frame,
)
from buildml.ranking.results import RankerFitResult, RankerPlan
from buildml.ranking.types import (
    PairwiseEstimator,
    PointwiseEstimator,
    RankerBackend,
    RankerConfig,
    RankerMethod,
)


def fit_ranker(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: RankerBackend | None = None,
    method: RankerMethod | str | None = None,
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
    n_estimators: int = 120,
    learning_rate: float = 0.08,
    hidden_dim: int = 64,
    epochs: int = 40,
    device: str = "cpu",
    random_state: int | None = 0,
) -> tuple[RankerPlan, RankerFitResult]:
    """Fit a leakage-safe tabular ranker on the Session **train** partition.

    Backends
    --------
    sklearn (fallback):
        Pointwise Ridge/HGB relevance regression or pairwise RankSVM-lite.
    industry (``buildml[ranking-industry]``):
        LightGBM LambdaRank, XGBoost rank:ndcg, or CatBoost YetiRank when
        installed — default backend when available.
    torch (``buildml[torch]``):
        Listwise-lite MLP with per-query softmax loss on graded relevance.

    Pipeline
    --------
    1. Resolve query / item / relevance / feature columns.
    2. Disclose query-group split honesty (prefer ``group_split``).
    3. Standardize features on train only.
    4. Fit the selected backend/method ranker.

    Honesty: Session tabular LTR — not a search-engine product, not RAG
    retrieve/generate, not recommender CF. Never trains on holdout rows.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    default_method = ranking_capability_matrix()["default_method_when_installed"]
    resolved_backend, resolved_method = resolve_backend_method(
        backend=backend,
        method=str(method if method is not None else default_method),
    )

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

    disclosures.append(
        f"Backend={resolved_backend}, method={resolved_method} on "
        f"{len(feat_cols)} standardized train features "
        f"({n_queries} queries, {len(train)} rows)."
    )

    if resolved_backend == "sklearn":
        sklearn_state = build_sklearn_ranker(
            X,
            y,
            groups,
            method=resolved_method,
            pointwise_estimator=pointwise_estimator,
            pairwise_estimator=pairwise_estimator,
            alpha=alpha,
            C=C,
            max_pairs_per_query=int(max_pairs_per_query),
            random_state=random_state,
        )
        estimator = sklearn_state
        coef = sklearn_state.coef_
        intercept = sklearn_state.intercept_
        n_pair_examples = sklearn_state.n_pairwise_examples
        if resolved_method == "pointwise":
            disclosures.append(
                f"sklearn pointwise: {pointwise_estimator} regresses graded relevance."
            )
        else:
            disclosures.append(
                f"sklearn pairwise: RankSVM-lite (LinearSVC); "
                f"{n_pair_examples} oriented pair(s) "
                f"(max_pairs_per_query={max_pairs_per_query})."
            )
    elif resolved_method == "lambdarank_lgbm":
        estimator = fit_lambdarank_lgbm(
            X,
            y,
            groups,
            random_state=random_state,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
        )
        disclosures.append(
            "industry: LightGBM LambdaRank (listwise GBDT; ndcg metric)."
        )
    elif resolved_method == "rank_ndcg_xgb":
        estimator = fit_rank_ndcg_xgb(
            X,
            y,
            groups,
            random_state=random_state,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
        )
        disclosures.append(
            "industry: XGBoost rank:ndcg (listwise GBDT objective)."
        )
    elif resolved_method == "yetirank_catboost":
        estimator = fit_yetirank_catboost(
            X,
            y,
            groups,
            random_state=random_state,
            iterations=int(n_estimators),
            learning_rate=float(learning_rate),
        )
        disclosures.append(
            "industry: CatBoost YetiRank (pairwise/listwise ranker)."
        )
    elif resolved_method == "listwise_lite":
        estimator = fit_listwise_lite(
            X,
            y,
            groups,
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            random_state=random_state,
            device=str(device),
        )
        disclosures.append(
            "torch: listwise-lite MLP with per-query softmax cross-entropy "
            "on normalized relevance grades."
        )
    else:
        raise ValidationError(f"Unsupported ranker method: {resolved_method!r}")

    disclosures.append(
        "Inference scores items within a query from frozen train features; "
        "holdout relevance labels are never used at fit."
    )
    disclosures.append(
        "Tabular LTR metrics (evaluate_ranker) differ from RAG chunk nDCG and "
        "recommender known-item ranking even when metric names overlap."
    )

    config = RankerConfig(
        backend=resolved_backend,
        method=resolved_method,  # type: ignore[arg-type]
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
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        device=str(device),
    )

    plan = RankerPlan(
        backend=resolved_backend,
        method=resolved_method,
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
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        device=str(device),
        random_state=random_state,
        group_split_disclosed=group_ok,
        split_kind=str(split_plan.kind),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = RankerFitResult(
        backend=resolved_backend,
        method=resolved_method,
        n_train_rows=plan.n_train_rows,
        n_train_queries=plan.n_train_queries,
        n_features=plan.n_features,
        query_column=query_col,
        item_column=item_col,
        relevance_column=rel_col,
        feature_columns=feat_cols,
        pointwise_estimator=pointwise_estimator if resolved_method == "pointwise" else None,
        pairwise_estimator=pairwise_estimator if resolved_method == "pairwise" else None,
        n_pairwise_examples=n_pair_examples,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result
