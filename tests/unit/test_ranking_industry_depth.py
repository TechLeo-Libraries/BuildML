"""Industry-depth tests for tabular LTR backends (R6.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import torch_spec_available
from buildml.ranking.catalog import (
    list_ranking_methods,
    ranking_capability_matrix,
    resolve_backend_method,
)
from buildml.ranking.extras import (
    lightgbm_available,
    ranking_industry_available,
    xgboost_available,
)


def _judgment_frame(n_queries: int = 24, n_items: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for q in range(n_queries):
        for item in range(n_items):
            f1 = float(rng.normal(q % 4, 0.6))
            rel = float(max(0, int(3 - abs(f1 - (q % 4)) + (item % 2 == 0))))
            rows.append(
                {
                    "query_id": f"q{q}",
                    "item_id": f"i{item}",
                    "f1": f1,
                    "f2": float(item),
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


def test_capability_matrix_sklearn_always_available() -> None:
    matrix = ranking_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "pointwise" in matrix["backends"]["sklearn"]["methods"]
    assert "ltr_vs_rag_vs_recommenders" in matrix


def test_list_ranking_methods_includes_sklearn() -> None:
    assert "pointwise" in list_ranking_methods(backend="sklearn")


def test_resolve_backend_method_sklearn_pointwise() -> None:
    backend, method = resolve_backend_method(backend="sklearn", method="pointwise")
    assert backend == "sklearn"
    assert method == "pointwise"


def test_resolve_industry_requires_extra_when_missing() -> None:
    if ranking_industry_available():
        backend, method = resolve_backend_method(
            backend="industry",
            method="lambdarank_lgbm" if lightgbm_available() else "rank_ndcg_xgb",
        )
        assert backend == "industry"
        assert method in {"lambdarank_lgbm", "rank_ndcg_xgb", "yetirank_catboost"}
    else:
        with pytest.raises(MissingExtraError):
            resolve_backend_method(backend="industry", method="lambdarank_lgbm")


def test_sklearn_pointwise_session_path() -> None:
    session = _grouped_session()
    fit = session.fit_ranker(
        backend="sklearn",
        method="pointwise",
        query_column="query_id",
        item_column="item_id",
    )
    assert fit.backend == "sklearn"
    assert fit.method == "pointwise"
    ev = session.evaluate_ranker(k=5)
    assert 0.0 <= float(ev.metrics["ndcg_at_k"]) <= 1.0


@pytest.mark.skipif(not lightgbm_available(), reason="lightgbm not installed")
def test_lambdarank_lgbm_session_path() -> None:
    session = _grouped_session()
    fit = session.fit_ranker(
        backend="industry",
        method="lambdarank_lgbm",
        query_column="query_id",
        item_column="item_id",
        n_estimators=40,
    )
    assert fit.backend == "industry"
    assert fit.method == "lambdarank_lgbm"
    ev = session.evaluate_ranker(k=5)
    assert ev.n_queries_scored > 0


@pytest.mark.skipif(not xgboost_available(), reason="xgboost not installed")
def test_rank_ndcg_xgb_session_path() -> None:
    session = _grouped_session()
    fit = session.fit_ranker(
        backend="industry",
        method="rank_ndcg_xgb",
        query_column="query_id",
        item_column="item_id",
        n_estimators=40,
    )
    assert fit.method == "rank_ndcg_xgb"
    ranked = session.rank(partition="test", k=3)
    assert ranked.n_queries > 0


@pytest.mark.skipif(not torch_spec_available(), reason="torch not installed")
def test_listwise_lite_session_path() -> None:
    session = _grouped_session()
    try:
        fit = session.fit_ranker(
            backend="torch",
            method="listwise_lite",
            query_column="query_id",
            item_column="item_id",
            epochs=8,
            hidden_dim=16,
        )
    except MissingExtraError:
        pytest.skip("torch not usable in this environment")
    assert fit.backend == "torch"
    assert fit.method == "listwise_lite"
    ev = session.evaluate_ranker(k=5)
    assert set(ev.metrics) >= {"ndcg_at_k", "map_at_k", "mrr_at_k"}


def test_backend_mismatch_raises_on_rank() -> None:
    session = _grouped_session()
    session.fit_ranker(
        backend="sklearn",
        method="pointwise",
        query_column="query_id",
        item_column="item_id",
    )
    with pytest.raises(ValidationError, match="backend"):
        session.rank(partition="test", k=3, backend="industry")


def test_ai_registry_has_ranking_capability_matrix() -> None:
    from buildml.ai.tools import build_default_registry

    registry = build_default_registry()
    assert "ranking_capability_matrix" in registry


def test_session_ranking_capability_matrix() -> None:
    matrix = Session.ranking_capability_matrix()
    assert "backends" in matrix
    assert matrix["backends"]["sklearn"]["available"] is True
