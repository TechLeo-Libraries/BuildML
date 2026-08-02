"""Deeper unit tests for decision helpers (cost matrix, LP, leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.optimize.allocate import select_lp_allocate


def _multiclass_session() -> Session:
    x, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    return (
        Session.ingest(frame)
        .set_roles(
            {**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.25, random_state=2)
        .fit(LogisticRegression(max_iter=600), task="classification")
    )


def test_cost_matrix_bayes_policy() -> None:
    session = _multiclass_session()
    classes = [str(c) for c in session._fit_result.estimator.classes_]
    # Penalize confusing class 0 -> 2 heavily
    n = len(classes)
    matrix = np.ones((n, n), dtype=float) - np.eye(n)
    matrix[0, n - 1] = 10.0
    fit = session.fit_decision_policy(
        method="cost_matrix",
        partition="validation",
        cost_matrix=matrix.tolist(),
        class_labels=classes,
    )
    assert fit.expected_cost is not None
    applied = session.apply_decisions(partition="test")
    assert applied.n_rows > 0
    assert set(map(str, applied.decisions)).issubset(set(classes))
    eval_result = session.evaluate_decisions(partition="test")
    assert "realized_cost_total" in eval_result.metrics


def test_allow_test_tuning_opt_in() -> None:
    x, y = make_classification(n_samples=200, n_features=5, random_state=3)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(5)])
    frame["y"] = y
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(5)}, "y": "target"})
        .split(test_size=0.3, validation_size=0.2, random_state=3)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    with pytest.raises(LeakageError):
        session.fit_decision_policy(
            method="threshold", partition="test", fp_cost=1.0, fn_cost=2.0
        )
    result = session.fit_decision_policy(
        method="threshold",
        partition="test",
        allow_test_tuning=True,
        fp_cost=1.0,
        fn_cost=2.0,
    )
    assert result.allow_test_tuning is True
    assert any("DANGEROUS" in w for w in result.warnings)


def test_lp_allocate_respects_budget() -> None:
    values = np.array([10.0, 9.0, 8.0, 1.0])
    costs = np.array([5.0, 5.0, 5.0, 5.0])
    result = select_lp_allocate(values, costs, budget=7.5, max_fraction=1.0)
    assert result["selected_cost"] <= 7.5 + 1e-6
    assert result["n_selected"] >= 1
    # Fractional: with equal costs, should prefer highest values
    assert result["selected_ids"][0] in {0, 1, 2}


def test_lp_session_smoke() -> None:
    x, y = make_classification(n_samples=220, n_features=6, random_state=4)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    frame["y"] = y
    frame["cost"] = 1.5
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(6)}, "y": "target"})
        .split(test_size=0.25, validation_size=0.25, random_state=4)
        .fit(LogisticRegression(max_iter=400), task="classification")
    )
    fit = session.fit_decision_policy(
        method="lp_allocate",
        partition="validation",
        budget=10.0,
        cost_column="cost",
        lp_max_fraction=1.0,
    )
    assert fit.selected_cost is not None
    assert fit.selected_cost <= 10.0 + 1e-5
    applied = session.apply_decisions(partition="test")
    assert applied.n_selected >= 1
    assert all(0.0 < f <= 1.0 + 1e-9 for f in applied.fractions)


def test_column_driven_topk_without_model_scores() -> None:
    n = 40
    frame = pd.DataFrame(
        {
            "score": np.linspace(1.0, 0.0, n),
            "cost": np.ones(n),
            "id": [f"r{i}" for i in range(n)],
            "y": [i % 2 for i in range(n)],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"score": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.25, random_state=0)
    )
    # No fit — column-driven allocation
    fit = session.fit_decision_policy(
        method="topk",
        partition="validation",
        capacity=2,
        score_source="column",
        score_column="score",
        cost_column="cost",
        id_column="id",
    )
    assert fit.n_selected == 2
    applied = session.apply_decisions(
        candidates=pd.DataFrame(
            {"score": [0.95, 0.2, 0.85], "cost": [1, 1, 1], "id": ["x", "y", "z"]}
        )
    )
    assert applied.selected_ids[0] == "x"


def test_threshold_requires_both_costs_when_partial() -> None:
    x, y = make_classification(n_samples=160, n_features=5, random_state=5)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(5)])
    frame["y"] = y
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(5)}, "y": "target"})
        .split(test_size=0.25, validation_size=0.25, random_state=5)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    with pytest.raises(ValidationError, match="fp_cost"):
        session.fit_decision_policy(
            method="threshold", partition="validation", fp_cost=1.0
        )
