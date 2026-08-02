"""Depth tests: leakage, known-item protocol, ranking metric sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.recommenders.features import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from buildml.recommenders.recommend import recommend_for_users


def _session() -> Session:
    rng = np.random.default_rng(3)
    rows = []
    for user in range(36):
        liked = rng.choice(24, size=7, replace=False)
        for item in liked:
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{item}",
                    "rating": float(rng.integers(3, 6)),
                }
            )
    frame = pd.DataFrame(rows)
    return (
        Session.ingest(frame)
        .set_roles({"user_id": "id", "item_id": "id", "rating": "target"})
        .split(test_size=0.25, validation_size=0.15, random_state=1)
    )


def test_metric_helpers_perfect_ranking() -> None:
    relevant = {"a", "b"}
    recommended = ["a", "b", "c", "d"]
    assert precision_at_k(recommended, relevant, 2) == 1.0
    assert recall_at_k(recommended, relevant, 2) == 1.0
    assert ndcg_at_k(recommended, relevant, 2) == pytest.approx(1.0)


def test_holdout_does_not_expand_item_catalog() -> None:
    session = _session()
    session.fit_recommender(
        method="item_knn",
        user_column="user_id",
        item_column="item_id",
        n_neighbors=10,
    )
    plan = session.recommender_plan
    assert plan is not None
    train_items = set(plan.item_ids)
    # Inject a holdout-only item into a recommend call indirectly via eval path:
    # recommendations must only contain train catalog ids.
    recs = session.recommend(partition="test", k=10)
    for items in recs.recommendations:
        assert set(items).issubset(train_items)


def test_exclude_train_items_from_recommendations() -> None:
    session = _session()
    session.fit_recommender(
        method="svd",
        user_column="user_id",
        item_column="item_id",
        n_factors=8,
        random_state=0,
    )
    plan = session.recommender_plan
    assert plan is not None
    warm = plan.user_ids[0]
    uidx = plan.user_index_[warm]
    train_hist = {
        plan.item_ids[j] for j in range(plan.n_items) if plan.matrix_[uidx, j] != 0
    }
    out = recommend_for_users(plan, [warm], k=20, exclude_train_items=True)
    assert set(out.recommendations[0]).isdisjoint(train_hist)


def test_content_requires_features() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="item_feature_columns"):
        session.fit_recommender(
            method="content",
            user_column="user_id",
            item_column="item_id",
        )


def test_distinct_from_eda_recommendation_schema() -> None:
    from buildml.explain.schemas import Recommendation
    from buildml.recommenders.results import RecommendResult

    assert Recommendation is not RecommendResult
    assert Recommendation.__name__ == "Recommendation"
    assert RecommendResult.__name__ == "RecommendResult"
