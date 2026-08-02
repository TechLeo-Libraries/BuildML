"""Depth tests: query-group leakage disclosure, metrics, pairwise path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.ranking.features import (
    average_precision_at_k,
    mrr_at_k,
    ndcg_at_k_graded,
)


def _judgment_frame(n_queries: int = 30, n_items: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for q in range(n_queries):
        for item in range(n_items):
            f1 = float(rng.normal(q % 4, 0.7))
            f2 = float(item)
            rel = float(max(0, int(3 - abs(f1 - (q % 4)) + (item % 2 == 0))))
            rows.append(
                {
                    "query_id": f"q{q}",
                    "item_id": f"i{item}",
                    "f1": f1,
                    "f2": f2,
                    "relevance": rel,
                }
            )
    return pd.DataFrame(rows)


def _grouped_session() -> Session:
    frame = _judgment_frame()
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "query_id": "group",
                "item_id": "id",
                "relevance": "target",
                "f1": "feature",
                "f2": "feature",
            }
        )
        .group_split(test_size=0.25, validation_size=0.15, random_state=1)
    )


def test_metric_helpers_perfect_ranking() -> None:
    # Perfect graded order: 3, 2, 1, 0
    rels = [3.0, 2.0, 1.0, 0.0]
    assert ndcg_at_k_graded(rels, 3) == pytest.approx(1.0)
    assert average_precision_at_k(rels, 3, threshold=0.0) == pytest.approx(1.0)
    assert mrr_at_k(rels, 3, threshold=0.0) == pytest.approx(1.0)


def test_group_split_marks_disclosed() -> None:
    session = _grouped_session()
    session.fit_ranker(
        method="pointwise",
        query_column="query_id",
        item_column="item_id",
    )
    plan = session.ranker_plan
    assert plan is not None
    assert plan.group_split_disclosed is True
    assert plan.split_kind == "group"


def test_random_split_warns_on_query_overlap() -> None:
    frame = _judgment_frame()
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "query_id": "id",
                "item_id": "id",
                "relevance": "target",
                "f1": "feature",
                "f2": "feature",
            }
        )
        .split(test_size=0.25, validation_size=0.15, random_state=1)
    )
    fit = session.fit_ranker(
        method="pointwise",
        query_column="query_id",
        item_column="item_id",
    )
    assert fit.warnings
    assert any("group_split" in w for w in fit.warnings)


def test_pairwise_fit_and_eval() -> None:
    session = _grouped_session()
    fit = session.fit_ranker(
        method="pairwise",
        query_column="query_id",
        item_column="item_id",
        max_pairs_per_query=40,
        random_state=0,
    )
    assert fit.n_pairwise_examples is not None
    assert fit.n_pairwise_examples > 0
    ev = session.evaluate_ranker(k=5)
    assert set(ev.metrics) >= {"ndcg_at_k", "map_at_k", "mrr_at_k"}
    for value in ev.metrics.values():
        assert 0.0 <= float(value) <= 1.0


def test_holdout_labels_not_required_for_rank() -> None:
    session = _grouped_session()
    session.fit_ranker(
        method="pointwise",
        query_column="query_id",
        item_column="item_id",
    )
    # rank uses features only; works even if we only ask for query ids
    plan = session.ranker_plan
    assert plan is not None
    test_q = session.dataset.frame.iloc[
        list(session._split_plan.test_indices)
    ]["query_id"].iloc[0]
    out = session.rank(query_ids=[test_q], k=3)
    assert out.n_queries == 1
    assert len(out.rankings[0]) <= 3


def test_requires_query_and_item_columns() -> None:
    session = _grouped_session()
    with pytest.raises(ValidationError, match="query_column"):
        session.fit_ranker(item_column="item_id")


def test_distinct_from_recommender_and_rag_schemas() -> None:
    from buildml.ranking.results import RankResult
    from buildml.recommenders.results import RecommendResult

    assert RankResult is not RecommendResult
