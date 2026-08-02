"""Industry recommender backend tests (skip without optional extras)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.recommenders.catalog import (
    default_method_for_feedback,
    recommender_capability_matrix,
    resolve_backend_method,
)
from buildml.recommenders.extras import implicit_available, lightfm_available


def _session(*, explicit: bool = True) -> Session:
    rng = np.random.default_rng(5)
    rows = []
    for user in range(40):
        liked = rng.choice(22, size=6, replace=False)
        for item in liked:
            row = {
                "user_id": f"u{user}",
                "item_id": f"i{item}",
                "f1": float(item % 4),
                "f2": float(item // 4),
                "age": float(20 + user % 25),
            }
            if explicit:
                row["rating"] = float(rng.integers(2, 6))
            rows.append(row)
    frame = pd.DataFrame(rows)
    roles = {
        "user_id": "id",
        "item_id": "id",
        "f1": "feature",
        "f2": "feature",
        "age": "feature",
    }
    if explicit:
        roles["rating"] = "target"
    return (
        Session.ingest(frame)
        .set_roles(roles)
        .split(test_size=0.25, validation_size=0.15, random_state=1)
    )


def test_capability_matrix_lists_backends() -> None:
    matrix = recommender_capability_matrix()
    assert "sklearn" in matrix["backends"]
    assert "implicit" in matrix["backends"]
    assert "lightfm" in matrix["backends"]
    assert matrix["backends"]["implicit"]["available"] == implicit_available()


def test_implicit_default_for_feedback() -> None:
    if implicit_available():
        assert default_method_for_feedback("implicit") == "als"
    else:
        assert default_method_for_feedback("implicit") == "nmf"


def test_resolve_backend_requires_extra() -> None:
    if implicit_available():
        return
    with pytest.raises(MissingExtraError):
        resolve_backend_method(backend="implicit", method="als", feedback="implicit")


@pytest.mark.skipif(not implicit_available(), reason="implicit not installed")
def test_implicit_als_fit_eval() -> None:
    session = _session(explicit=False)
    fit = session.fit_recommender(
        method="als",
        user_column="user_id",
        item_column="item_id",
        feedback="implicit",
        n_factors=8,
        n_iterations=5,
    )
    assert fit.backend == "implicit"
    assert fit.method == "als"
    ev = session.evaluate_recommender(k=5)
    assert set(ev.metrics) >= {"precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"}


@pytest.mark.skipif(not implicit_available(), reason="implicit not installed")
def test_implicit_default_method_on_fit() -> None:
    session = _session(explicit=False)
    fit = session.fit_recommender(
        user_column="user_id",
        item_column="item_id",
        feedback="implicit",
        n_factors=8,
        n_iterations=5,
    )
    assert fit.method == "als"
    assert fit.backend == "implicit"


@pytest.mark.skipif(not lightfm_available(), reason="lightfm not installed")
def test_lightfm_hybrid_fit_eval() -> None:
    session = _session()
    fit = session.fit_recommender(
        method="lightfm",
        user_column="user_id",
        item_column="item_id",
        item_feature_columns=["f1", "f2"],
        user_feature_columns=["age"],
        n_factors=8,
        lightfm_epochs=3,
    )
    assert fit.backend == "lightfm"
    recs = session.recommend(partition="test", k=5)
    assert recs.n_users > 0
    ev = session.evaluate_recommender(k=5)
    assert ev.n_users_scored >= 0
