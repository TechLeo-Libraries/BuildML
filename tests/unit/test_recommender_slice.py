"""Session-facing slice tests for recommendation systems."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG


def _ratings_frame(n_users: int = 40, n_items: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for user in range(n_users):
        liked = rng.choice(n_items, size=8, replace=False)
        for item in liked:
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{item}",
                    "rating": float(rng.integers(1, 6)),
                    "f1": float(item % 5),
                    "f2": float(item // 5),
                }
            )
    return pd.DataFrame(rows)


def _demo_session(n_users: int = 40) -> Session:
    frame = _ratings_frame(n_users=n_users)
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "user_id": "id",
                "item_id": "id",
                "rating": "target",
                "f1": "feature",
                "f2": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.15, random_state=0)
    )


def test_core_import_and_catalog() -> None:
    import buildml.recommenders as rec

    assert hasattr(rec, "fit_recommender")
    assert hasattr(Session, "fit_recommender")
    for op in (
        "fit_recommender",
        "recommend",
        "evaluate_recommender",
        "save_recommender_bundle",
        "load_recommender_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert (
        "recommender-collaborative-filtering"
        in OPERATION_CATALOG["fit_recommender"].concept_links
    )
    assert (
        "recommender-ranking-metrics"
        in OPERATION_CATALOG["evaluate_recommender"].concept_links
    )
    assert (
        "recommender-bundle-boundary"
        in OPERATION_CATALOG["save_recommender_bundle"].concept_links
    )

    registry = build_default_registry()
    for name in (
        "fit_recommender",
        "recommend",
        "evaluate_recommender",
        "save_recommender_bundle",
        "load_recommender_bundle",
    ):
        assert name in registry


def test_fit_requires_split() -> None:
    frame = _ratings_frame(n_users=10, n_items=8)
    session = Session.ingest(frame).set_roles(
        {"user_id": "id", "item_id": "id", "rating": "target", "f1": "feature", "f2": "feature"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_recommender(user_column="user_id", item_column="item_id")


def test_requires_user_item_columns() -> None:
    session = _demo_session(n_users=20)
    with pytest.raises(ValidationError, match="user_column"):
        session.fit_recommender()


def test_session_fit_recommend_evaluate_bundle(tmp_path: Path) -> None:
    session = _demo_session()
    fit = session.fit_recommender(
        method="item_knn",
        user_column="user_id",
        item_column="item_id",
        n_neighbors=15,
        random_state=0,
    )
    assert fit.n_users >= 2
    assert fit.n_items >= 2
    assert session.recommender_plan is not None

    recs = session.recommend(partition="test", k=5)
    assert recs.n_users > 0
    assert recs.k == 5
    assert len(recs.recommendations) == recs.n_users

    ev = session.evaluate_recommender(partition="test", k=5)
    for key in ("precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"):
        assert key in ev.metrics
        assert 0.0 <= ev.metrics[key] <= 1.0

    session.save_recommender_bundle(tmp_path / "rec")
    other = _demo_session()
    other.load_recommender_bundle(tmp_path / "rec")
    assert other.recommender_plan is not None
    assert other.evaluate_recommender(partition="test", k=5).n_users_scored >= 0


@pytest.mark.parametrize("method", ["user_knn", "svd", "nmf"])
def test_method_variants(method: str) -> None:
    session = _demo_session(n_users=35)
    fit = session.fit_recommender(
        method=method,  # type: ignore[arg-type]
        user_column="user_id",
        item_column="item_id",
        n_neighbors=12,
        n_factors=8,
        random_state=0,
    )
    assert fit.method == method
    metrics = session.evaluate_recommender(partition="validation", k=5).metrics
    assert "ndcg_at_k" in metrics


def test_content_method() -> None:
    session = _demo_session(n_users=30)
    fit = session.fit_recommender(
        method="content",
        user_column="user_id",
        item_column="item_id",
        item_feature_columns=["f1", "f2"],
    )
    assert fit.method == "content"
    recs = session.recommend(partition="test", k=3)
    assert recs.n_users > 0


def test_implicit_feedback() -> None:
    session = _demo_session(n_users=25)
    fit = session.fit_recommender(
        method="item_knn",
        user_column="user_id",
        item_column="item_id",
        feedback="implicit",
        n_neighbors=10,
    )
    assert fit.feedback == "implicit"
    assert session.evaluate_recommender(k=5).metrics


def test_cold_start_disclosure() -> None:
    session = _demo_session(n_users=20)
    session.fit_recommender(
        method="item_knn",
        user_column="user_id",
        item_column="item_id",
        cold_start="skip",
        n_neighbors=8,
    )
    recs = session.recommend(user_ids=["never_seen_user_xyz"], k=5)
    assert "never_seen_user_xyz" in recs.cold_start_users
    assert recs.recommendations[0] == ()


def test_walkthrough_status() -> None:
    session = _demo_session(n_users=20)
    session.fit_recommender(
        method="svd",
        user_column="user_id",
        item_column="item_id",
        n_factors=6,
        random_state=0,
    )
    report = session.walkthrough()
    status = report.recommender_status
    assert status.get("enabled") is True
    assert status.get("method") == "svd"
