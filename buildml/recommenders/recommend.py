"""Top-K recommendations from a train-fitted RecommenderPlan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.recommenders.adapters.implicit_lib import score_implicit_model
from buildml.recommenders.adapters.lightfm import score_lightfm_model
from buildml.recommenders.features import partition_frame
from buildml.recommenders.models import (
    popularity_scores,
    score_content,
    score_factorization,
    score_item_knn,
    score_user_knn,
    top_k_from_scores,
)
from buildml.recommenders.results import RecommenderPlan, RecommendResult


def _scores_for_user(
    plan: RecommenderPlan,
    user_idx: int,
    *,
    exclude_train: bool,
) -> np.ndarray:
    exclude = np.zeros(plan.n_items, dtype=bool)
    if exclude_train:
        exclude = plan.matrix_[user_idx] != 0

    method = plan.method
    backend = plan.backend
    if backend == "implicit":
        if plan.backend_model_ is None:
            raise ValidationError("RecommenderPlan missing implicit backend model.")
        return score_implicit_model(
            plan.backend_model_,
            user_idx,
            n_items=plan.n_items,
            exclude_mask=exclude,
        )
    if backend == "lightfm":
        if plan.backend_model_ is None:
            raise ValidationError("RecommenderPlan missing LightFM backend model.")
        return score_lightfm_model(
            plan.backend_model_,
            user_idx,
            n_items=plan.n_items,
            exclude_mask=exclude,
            user_features=plan.lightfm_user_features_,
            item_features=plan.lightfm_item_features_,
        )
    if method == "item_knn":
        if plan.similarity_ is None:
            raise ValidationError("RecommenderPlan missing item similarity.")
        return score_item_knn(
            plan.matrix_,
            plan.similarity_,
            user_idx,
            n_neighbors=plan.n_neighbors,
            exclude_mask=exclude,
        )
    if method == "user_knn":
        if plan.similarity_ is None:
            raise ValidationError("RecommenderPlan missing user similarity.")
        return score_user_knn(
            plan.matrix_,
            plan.similarity_,
            user_idx,
            n_neighbors=plan.n_neighbors,
            exclude_mask=exclude,
        )
    if method in {"svd", "nmf"}:
        if plan.user_factors_ is None or plan.item_factors_ is None:
            raise ValidationError("RecommenderPlan missing factorization factors.")
        return score_factorization(
            plan.user_factors_,
            plan.item_factors_,
            user_idx,
            global_mean=plan.global_mean_ if method == "svd" else 0.0,
            exclude_mask=exclude,
        )
    if method == "content":
        if plan.item_features_ is None:
            raise ValidationError("RecommenderPlan missing item features.")
        return score_content(
            plan.matrix_,
            plan.item_features_,
            user_idx,
            exclude_mask=exclude,
        )
    raise ValidationError(f"Unknown method on plan: {method!r}")


def recommend_for_users(
    plan: RecommenderPlan,
    user_ids: Sequence[Any],
    *,
    k: int = 10,
    exclude_train_items: bool = True,
) -> RecommendResult:
    """Recommend top-K train-catalog items for the given user ids.

    Scores each warm user with the fitted plan method, applies cold-start
    policy for unknown users, and restricts candidates to the train item
    catalog (known-item protocol).

    Parameters
    ----------
    plan:
        Train-fitted :class:`~buildml.recommenders.results.RecommenderPlan`.
    user_ids:
        User entity ids to generate recommendations for.
    k:
        Number of items to recommend per user.
    exclude_train_items:
        When ``True``, suppress items the user already interacted with in train.

    Returns
    -------
    RecommendResult
        Parallel per-user recommendation lists, scores, and cold-start metadata.

    Raises
    ------
    ValidationError
        When ``k`` is less than 1 or the plan lacks required fitted state.
    """
    if int(k) < 1:
        raise ValidationError("k must be >= 1.")

    recs: list[tuple[Any, ...]] = []
    scores_out: list[tuple[float, ...]] = []
    cold: list[Any] = []
    disclosures = list(plan.disclosures[:4])
    warnings: list[str] = []

    for user in user_ids:
        uidx = plan.user_index_.get(user)
        if uidx is None:
            cold.append(user)
            if plan.cold_start == "popularity":
                # Cold user: no train history to exclude
                excl = np.zeros(plan.n_items, dtype=bool)
                sc = popularity_scores(plan.item_popularity_, excl)
                items, scs = top_k_from_scores(sc, plan.item_ids, k)
                recs.append(items)
                scores_out.append(scs)
            else:
                recs.append(())
                scores_out.append(())
            continue

        sc = _scores_for_user(plan, uidx, exclude_train=exclude_train_items)
        # If all scores are zero / -inf (e.g. empty after exclude), fall back
        if not np.isfinite(sc).any() or (
            np.nanmax(np.where(np.isfinite(sc), sc, -np.inf)) <= -np.inf
        ):
            excl = plan.matrix_[uidx] != 0 if exclude_train_items else np.zeros(
                plan.n_items, dtype=bool
            )
            sc = popularity_scores(plan.item_popularity_, excl)
            warnings.append(
                f"User {user!r}: model scores empty; used popularity fallback."
            )
        items, scs = top_k_from_scores(sc, plan.item_ids, k)
        recs.append(items)
        scores_out.append(scs)

    if cold:
        disclosures.append(
            f"{len(cold)} cold-start user(s) not in train; policy={plan.cold_start!r}."
        )
    disclosures.append(
        "Candidates restricted to train item catalog (known-item protocol)."
    )

    return RecommendResult(
        k=int(k),
        n_users=len(user_ids),
        method=plan.method,
        user_ids=tuple(user_ids),
        recommendations=tuple(recs),
        scores=tuple(scores_out),
        cold_start_users=tuple(cold),
        excluded_train_items=exclude_train_items,
        disclosures=tuple(dict.fromkeys(disclosures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def recommend(
    dataset: Dataset,
    plan: RecommenderPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str | None = None,
    user_ids: Sequence[Any] | None = None,
    k: int = 10,
    exclude_train_items: bool = True,
) -> RecommendResult:
    """Recommend top-K items for users from a partition or an explicit list.

    Provide either ``user_ids`` or ``partition`` (unique users in that frame).
    Delegates scoring to :func:`recommend_for_users` after resolving the user
    list.

    Parameters
    ----------
    dataset:
        Session dataset containing user interaction columns.
    plan:
        Train-fitted :class:`~buildml.recommenders.results.RecommenderPlan`.
    split_plan:
        Split definition used when ``partition`` is provided.
    partition:
        Split partition name whose unique users receive recommendations.
    user_ids:
        Explicit list of user entity ids to recommend for.
    k:
        Number of items to recommend per user.
    exclude_train_items:
        When ``True``, suppress items the user already interacted with in train.

    Returns
    -------
    RecommendResult
        Parallel per-user recommendation lists, scores, and cold-start metadata.

    Raises
    ------
    ValidationError
        When neither or both of ``user_ids`` and ``partition`` are supplied,
        the user column is missing, or no users remain to score.
    """
    if user_ids is None and partition is None:
        raise ValidationError("recommend() requires user_ids= or partition=.")
    if user_ids is not None and partition is not None:
        raise ValidationError("Pass only one of user_ids= or partition=.")

    if user_ids is None:
        assert partition is not None
        frame = partition_frame(dataset, split_plan, partition)
        if plan.user_column not in frame.columns:
            raise ValidationError(
                f"user_column {plan.user_column!r} missing from partition={partition!r}."
            )
        resolved = list(dict.fromkeys(frame[plan.user_column].dropna().tolist()))
    else:
        resolved = list(user_ids)

    if not resolved:
        raise ValidationError("No users to recommend for.")

    return recommend_for_users(
        plan,
        resolved,
        k=k,
        exclude_train_items=exclude_train_items,
    )
