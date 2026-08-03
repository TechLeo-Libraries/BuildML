"""Session-facing slice tests for decision / optimisation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.optimize.allocate import select_knapsack, select_topk


def _binary_session(*, with_cost: bool = True) -> Session:
    x, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        weights=[0.65, 0.35],
        random_state=0,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    if with_cost:
        frame["cost"] = np.where(y == 1, 2.0, 1.0)
        frame["cid"] = [f"c{i}" for i in range(len(frame))]
    roles = {**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"}
    return (
        Session.ingest(frame)
        .set_roles(roles)
        .split(test_size=0.25, validation_size=0.25, random_state=0)
        .fit(LogisticRegression(max_iter=400), task="classification")
    )


def test_core_import_and_catalog() -> None:
    import buildml.optimize as optimize

    assert hasattr(optimize, "fit_decision_policy")
    assert hasattr(Session, "fit_decision_policy")
    for op in (
        "fit_decision_policy",
        "apply_decisions",
        "evaluate_decisions",
        "save_decision_bundle",
        "load_decision_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "decision-operating-point" in OPERATION_CATALOG["fit_decision_policy"].concept_links
    assert "decision-allocation" in OPERATION_CATALOG["fit_decision_policy"].concept_links
    assert "decision-bundle-boundary" in OPERATION_CATALOG["save_decision_bundle"].concept_links

    registry = build_default_registry()
    for name in (
        "fit_decision_policy",
        "apply_decisions",
        "evaluate_decisions",
        "save_decision_bundle",
        "load_decision_bundle",
    ):
        assert name in registry


def test_requires_split() -> None:
    x, y = make_classification(n_samples=80, n_features=4, random_state=1)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = y
    session = Session.ingest(frame).set_roles(
        {**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"}
    )
    with pytest.raises(ValidationError, match="split"):
        session.fit_decision_policy(method="threshold", fp_cost=1.0, fn_cost=2.0)


def test_refuses_test_tuning_without_opt_in() -> None:
    session = _binary_session()
    with pytest.raises(LeakageError, match="allow_test_tuning"):
        session.fit_decision_policy(
            method="threshold",
            partition="test",
            fp_cost=1.0,
            fn_cost=3.0,
        )


def test_threshold_policy_and_tune_threshold_crosslink() -> None:
    session = _binary_session()
    result = session.fit_decision_policy(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=4.0,
    )
    assert result.threshold is not None
    assert session.decision_plan is not None
    assert session._last_diagnostic is not None
    assert session._last_diagnostic.kind == "threshold_sweep"

    # Classical path still works and does not clear the DecisionPlan
    report = session.tune_threshold(
        partition="validation", fp_cost=1.0, fn_cost=4.0
    )
    assert report.payload["recommendation_basis"] == "min_expected_cost"
    assert session.decision_plan is not None

    eval_result = session.evaluate_decisions(partition="test")
    assert "f1" in eval_result.metrics
    assert eval_result.realized_cost is not None


def test_knapsack_and_topk_helpers() -> None:
    values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    costs = np.array([4.0, 3.0, 2.0, 2.0, 1.0])
    top = select_topk(values, capacity=2, costs=costs)
    assert top["n_selected"] == 2
    assert top["selected_indices"] == (0, 1)

    knap = select_knapsack(values, costs, budget=5.0, solver="dp")
    assert knap["selected_cost"] <= 5.0 + 1e-9
    assert knap["n_selected"] >= 1


def test_allocation_session_smoke() -> None:
    session = _binary_session()
    fit = session.fit_decision_policy(
        method="topk",
        partition="validation",
        capacity=5,
        score_source="model_proba",
        cost_column="cost",
        id_column="cid",
    )
    assert fit.n_selected == 5
    applied = session.apply_decisions(partition="test")
    assert applied.n_selected <= 5
    assert len(applied.selected_ids) == applied.n_selected

    knap = session.fit_decision_policy(
        method="knapsack",
        partition="validation",
        budget=20.0,
        cost_column="cost",
        id_column="cid",
    )
    assert knap.selected_cost is not None
    assert knap.selected_cost <= 20.0 + 1e-6


def test_bundle_roundtrip(tmp_path) -> None:
    session = _binary_session()
    session.fit_decision_policy(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=2.0,
    )
    path = tmp_path / "decision_bundle"
    session.save_decision_bundle(path)
    other = _binary_session()
    other.load_decision_bundle(path, trusted=True)
    assert other.decision_plan is not None
    assert other.decision_plan.method == "threshold"
    assert other.decision_plan.threshold is not None
    eval_result = other.evaluate_decisions(partition="test")
    assert eval_result.n_rows > 0


def test_walkthrough_decision_status() -> None:
    session = _binary_session()
    session.fit_decision_policy(
        method="threshold", partition="validation", fp_cost=1.0, fn_cost=2.0
    )
    report = session.walkthrough()
    assert report.decision_status.get("has_decision_plan") is True
    assert report.decision_status.get("method") == "threshold"
