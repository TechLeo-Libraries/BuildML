"""Holdout ranking evaluation for recommenders (known-item protocol)."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.recommenders.features import (
    build_interactions,
    mean_average_precision,
    ndcg_at_k,
    partition_frame,
    precision_at_k,
    recall_at_k,
)
from buildml.recommenders.recommend import recommend_for_users
from buildml.recommenders.results import RecommenderEvalResult, RecommenderPlan


def evaluate_recommender(
    dataset: Dataset,
    plan: RecommenderPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "test",
    k: int = 10,
) -> RecommenderEvalResult:
    """Score ranking quality on a holdout partition.

    Protocol
    --------
    - Fit never sees holdout interactions (caller responsibility / Session gate).
    - Ground truth for a user = holdout items that appear in the **train item
      catalog** (known-item protocol). Holdout-only items are disclosed and
      excluded from relevant sets.
    - Users absent from train are cold-start; they do not enter ranking
      averages (counted separately).
    - For each warm user with ≥1 known holdout item: recommend top-K among
      train items, excluding that user's **train** history, then score
      Precision@K, Recall@K, nDCG@K, MAP@K.

    Metrics are micro-averaged over scored users.

    Parameters
    ----------
    dataset:
        Full Session dataset containing holdout interactions.
    plan:
        Train-fitted :class:`~buildml.recommenders.results.RecommenderPlan`;
        must not be refit on holdout data.
    split_plan:
        Split definition used to extract the holdout partition.
    partition:
        Split partition name to score (default ``"test"``).
    k:
        Cutoff for Precision@K, Recall@K, nDCG@K, and MAP@K.

    Returns
    -------
    RecommenderEvalResult
        Micro-averaged ranking metrics, cold-start counts, and protocol
        disclosures.

    Raises
    ------
    ValidationError
        When ``k`` is less than 1.
    """
    if int(k) < 1:
        raise ValidationError("k must be >= 1.")

    holdout = partition_frame(dataset, split_plan, partition)
    interactions = build_interactions(
        holdout,
        user_column=plan.user_column,
        item_column=plan.item_column,
        rating_column=plan.rating_column,
        feedback=plan.feedback,
        min_rating=plan.min_rating,
    )
    n_holdout = int(len(interactions))
    train_item_set = set(plan.item_ids)
    train_user_set = set(plan.user_ids)

    # Relevant items per warm user (known-item filtered)
    relevant: dict = {}
    unknown_item_hits = 0
    cold_users: set = set()
    for user, item in zip(
        interactions[plan.user_column].tolist(),
        interactions[plan.item_column].tolist(),
        strict=True,
    ):
        if user not in train_user_set:
            cold_users.add(user)
            continue
        if item not in train_item_set:
            unknown_item_hits += 1
            continue
        relevant.setdefault(user, set()).add(item)

    warm_users = [u for u, items in relevant.items() if items]
    disclosures = [
        "Holdout scored with frozen train recommender (no refit).",
        "Known-item protocol: relevant set = holdout items ∩ train catalog.",
        f"Cold-start users excluded from metric averages: {len(cold_users)}.",
    ]
    warnings: list[str] = []
    if unknown_item_hits:
        warnings.append(
            f"{unknown_item_hits} holdout interaction(s) reference items never "
            "seen in train; excluded from relevant sets."
        )
    if not warm_users:
        return RecommenderEvalResult(
            partition=str(partition),
            method=plan.method,
            k=int(k),
            n_users_scored=0,
            n_cold_start_users=len(cold_users),
            n_holdout_interactions=n_holdout,
            metrics={
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "map_at_k": 0.0,
            },
            disclosures=tuple(disclosures + ["No warm users with known holdout items."]),
            warnings=tuple(warnings),
        )

    rec = recommend_for_users(
        plan, warm_users, k=k, exclude_train_items=True
    )
    per_rec: list[list] = [list(r) for r in rec.recommendations]
    per_rel: list[set] = [relevant[u] for u in warm_users]

    prec = float(np.mean([precision_at_k(r, s, k) for r, s in zip(per_rec, per_rel)]))
    rec_at = float(np.mean([recall_at_k(r, s, k) for r, s in zip(per_rec, per_rel)]))
    ndcg = float(np.mean([ndcg_at_k(r, s, k) for r, s in zip(per_rec, per_rel)]))
    map_k = mean_average_precision(per_rec, per_rel, k)

    return RecommenderEvalResult(
        partition=str(partition),
        method=plan.method,
        k=int(k),
        n_users_scored=len(warm_users),
        n_cold_start_users=len(cold_users),
        n_holdout_interactions=n_holdout,
        metrics={
            "precision_at_k": prec,
            "recall_at_k": rec_at,
            "ndcg_at_k": ndcg,
            "map_at_k": map_k,
        },
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
