"""Fit recommendation models on Session train interactions only."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.recommenders.features import (
    build_interactions,
    build_user_item_matrix,
    item_feature_matrix,
    resolve_interaction_columns,
    train_partition_frame,
)
from buildml.recommenders.models import (
    fit_item_similarity,
    fit_nmf_factors,
    fit_svd_factors,
    fit_user_similarity,
)
from buildml.recommenders.results import RecommenderFitResult, RecommenderPlan
from buildml.recommenders.types import (
    ColdStartPolicy,
    FeedbackMode,
    RecommenderConfig,
    RecommenderMethod,
)


def fit_recommender(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: RecommenderMethod = "item_knn",
    user_column: str | None = None,
    item_column: str | None = None,
    rating_column: str | None = None,
    feedback: FeedbackMode = "explicit",
    n_neighbors: int = 40,
    n_factors: int = 32,
    min_rating: float | None = None,
    item_feature_columns: Sequence[str] | None = None,
    cold_start: ColdStartPolicy = "popularity",
    random_state: int | None = 0,
) -> tuple[RecommenderPlan, RecommenderFitResult]:
    """Fit a leakage-safe recommender on the Session **train** partition.

    Pipeline
    --------
    1. Resolve user/item/(rating) columns.
    2. Build train-only interactions and user×item matrix.
    3. Fit neighborhood similarities, matrix factorization, or content profiles.
    4. Store popularity prior for cold-start disclosure / fallback.

    Honesty: Session collaborative filtering + optional content features —
    not a Netflix-scale recsys platform. Never trains on holdout interactions.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    if method not in {"item_knn", "user_knn", "svd", "nmf", "content"}:
        raise ValidationError(f"Unknown recommender method: {method!r}")
    if int(n_neighbors) < 1:
        raise ValidationError("n_neighbors must be >= 1.")
    if int(n_factors) < 1:
        raise ValidationError("n_factors must be >= 1.")
    if feedback not in {"explicit", "implicit"}:
        raise ValidationError("feedback must be 'explicit' or 'implicit'.")
    if cold_start not in {"skip", "popularity"}:
        raise ValidationError("cold_start must be 'skip' or 'popularity'.")

    user_col, item_col, rating_col, disclosures = resolve_interaction_columns(
        dataset,
        user_column=user_column,
        item_column=item_column,
        rating_column=rating_column,
        feedback=feedback,
    )
    warnings: list[str] = []

    train = train_partition_frame(dataset, split_plan)
    interactions = build_interactions(
        train,
        user_column=user_col,
        item_column=item_col,
        rating_column=rating_col,
        feedback=feedback,
        min_rating=min_rating,
    )
    matrix, users, items, user_index, item_index = build_user_item_matrix(
        interactions, user_column=user_col, item_column=item_col
    )
    if len(users) < 2 or len(items) < 2:
        raise ValidationError(
            f"Need ≥2 train users and ≥2 train items; got "
            f"{len(users)} users / {len(items)} items."
        )

    mask = matrix != 0
    global_mean = float(matrix[mask].mean()) if mask.any() else 0.0
    item_popularity = mask.sum(axis=0).astype(float)

    similarity = None
    user_factors = None
    item_factors = None
    feat_cols: tuple[str, ...] = ()
    item_features = None
    feat_mean = None
    feat_scale = None

    if method == "item_knn":
        similarity = fit_item_similarity(matrix)
        disclosures.append(
            f"item_knn: cosine item-item similarity on {len(items)} train items; "
            f"n_neighbors={n_neighbors}."
        )
    elif method == "user_knn":
        similarity = fit_user_similarity(matrix)
        disclosures.append(
            f"user_knn: cosine user-user similarity on {len(users)} train users; "
            f"n_neighbors={n_neighbors}."
        )
    elif method == "svd":
        user_factors, item_factors = fit_svd_factors(
            matrix, n_factors=n_factors, random_state=random_state
        )
        disclosures.append(
            f"svd: TruncatedSVD factors "
            f"({user_factors.shape[1]} components) on train matrix; "
            f"scores use train global mean centering."
        )
    elif method == "nmf":
        user_factors, item_factors = fit_nmf_factors(
            matrix, n_factors=n_factors, random_state=random_state
        )
        disclosures.append(
            f"nmf: Non-negative MF ({user_factors.shape[1]} components) on train."
        )
    else:  # content
        if not item_feature_columns:
            raise ValidationError(
                "method='content' requires item_feature_columns= "
                "(numeric columns describing items)."
            )
        feat_cols = tuple(str(c) for c in item_feature_columns)
        item_features, feat_mean, feat_scale = item_feature_matrix(
            train,
            item_column=item_col,
            item_ids=items,
            feature_columns=list(feat_cols),
        )
        disclosures.append(
            f"content: user profiles from rating-weighted train item features "
            f"{list(feat_cols)}; known-item catalog only."
        )

    disclosures.append(
        "Known-item protocol: recommendations are restricted to items observed "
        "in train. Holdout-only items are never candidates."
    )
    disclosures.append(
        f"Cold-start policy={cold_start!r}: users absent from train use "
        + (
            "popularity fallback over train items."
            if cold_start == "popularity"
            else "empty recommendations (skip)."
        )
    )

    config = RecommenderConfig(
        method=method,
        user_column=user_col,
        item_column=item_col,
        rating_column=rating_col,
        feedback=feedback,
        n_neighbors=n_neighbors,
        n_factors=n_factors,
        min_rating=min_rating,
        item_feature_columns=feat_cols or None,
        cold_start=cold_start,
        random_state=random_state,
    )

    plan = RecommenderPlan(
        method=method,
        user_column=user_col,
        item_column=item_col,
        rating_column=rating_col,
        feedback=feedback,
        n_neighbors=int(n_neighbors),
        n_factors=int(n_factors),
        n_train_interactions=int(len(interactions)),
        n_users=len(users),
        n_items=len(items),
        user_ids=users,
        item_ids=items,
        user_index_=user_index,
        item_index_=item_index,
        matrix_=matrix,
        similarity_=similarity,
        user_factors_=user_factors,
        item_factors_=item_factors,
        global_mean_=global_mean,
        item_popularity_=item_popularity,
        item_feature_columns=feat_cols,
        item_features_=item_features,
        item_feature_mean_=feat_mean,
        item_feature_scale_=feat_scale,
        cold_start=cold_start,
        min_rating=min_rating,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = RecommenderFitResult(
        method=method,
        n_train_interactions=plan.n_train_interactions,
        n_users=plan.n_users,
        n_items=plan.n_items,
        feedback=feedback,
        user_column=user_col,
        item_column=item_col,
        rating_column=rating_col,
        n_neighbors=n_neighbors if method in {"item_knn", "user_knn"} else None,
        n_factors=n_factors if method in {"svd", "nmf"} else None,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result
