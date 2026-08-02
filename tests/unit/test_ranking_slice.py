"""Session-facing slice tests for tabular learning-to-rank."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.ranking.features import resolve_ranking_columns
from buildml.ranking.models import fit_pointwise, score_pointwise


def test_core_import_and_catalog() -> None:
    import buildml.ranking as ranking

    assert hasattr(ranking, "fit_ranker")
    assert hasattr(ranking, "ranking_capability_matrix")
    assert hasattr(Session, "fit_ranker")
    assert hasattr(Session, "ranking_capability_matrix")
    for op in (
        "fit_ranker",
        "rank",
        "evaluate_ranker",
        "save_ranker_bundle",
        "load_ranker_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "ltr-tabular-ranking" in OPERATION_CATALOG["fit_ranker"].concept_links
    assert "ltr-ranking-metrics" in OPERATION_CATALOG["evaluate_ranker"].concept_links
    assert "ltr-bundle-boundary" in OPERATION_CATALOG["save_ranker_bundle"].concept_links

    registry = build_default_registry()
    for name in (
        "fit_ranker",
        "rank",
        "evaluate_ranker",
        "save_ranker_bundle",
        "load_ranker_bundle",
    ):
        assert name in registry


def test_fit_requires_split() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q0", "q0", "q1", "q1"],
            "item_id": ["a", "b", "a", "b"],
            "f1": [1.0, 2.0, 1.5, 2.5],
            "relevance": [1.0, 0.0, 2.0, 1.0],
        }
    )
    session = Session.ingest(frame).set_roles(
        {
            "query_id": "group",
            "item_id": "id",
            "f1": "feature",
            "relevance": "target",
        }
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_ranker(query_column="query_id", item_column="item_id")


def test_resolve_defaults_feature_roles() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q0", "q0", "q1", "q1"],
            "item_id": ["a", "b", "a", "b"],
            "f1": [1.0, 2.0, 1.5, 2.5],
            "relevance": [1.0, 0.0, 2.0, 1.0],
        }
    )
    session = Session.ingest(frame).set_roles(
        {
            "query_id": "group",
            "item_id": "id",
            "f1": "feature",
            "relevance": "target",
        }
    )
    q, i, r, feats, discs = resolve_ranking_columns(
        session.dataset,
        query_column="query_id",
        item_column="item_id",
        relevance_column=None,
        feature_columns=None,
    )
    assert q == "query_id"
    assert i == "item_id"
    assert r == "relevance"
    assert feats == ("f1",)
    assert any("feature_columns defaulted" in d for d in discs)


def test_feature_columns_must_be_numeric() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q0", "q0"],
            "item_id": ["a", "b"],
            "f1": ["x", "y"],
            "relevance": [1.0, 0.0],
        }
    )
    session = Session.ingest(frame).set_roles(
        {
            "query_id": "group",
            "item_id": "id",
            "f1": "feature",
            "relevance": "target",
        }
    )
    with pytest.raises(ValidationError, match="numeric"):
        resolve_ranking_columns(
            session.dataset,
            query_column="query_id",
            item_column="item_id",
            relevance_column="relevance",
            feature_columns=["f1"],
        )


def test_pointwise_scores_finite() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 0.5, 1.0])
    model = fit_pointwise(X, y, estimator="ridge", alpha=1.0, random_state=0)
    scores = score_pointwise(model, X)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
