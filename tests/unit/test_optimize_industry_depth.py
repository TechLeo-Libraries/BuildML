"""Industry-depth tests for optimisation / decision helpers (R6.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.optimize.allocate import select_knapsack_with_backend
from buildml.optimize.catalog import (
    decision_capability_matrix,
    optimize_capability_matrix,
    resolve_backend,
)
from buildml.optimize.extras import pulp_available, xgboost_available


def _binary_session(seed: int = 11) -> Session:
    x, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        random_state=seed,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    frame["y"] = y
    frame["cost"] = np.where(y == 1, 2.0, 1.0)
    return (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(6)}, "y": "target", "cost": "feature"})
        .split(test_size=0.25, validation_size=0.25, random_state=seed)
        .fit(LogisticRegression(max_iter=400), task="classification")
    )


def test_capability_matrix_native_always_available() -> None:
    matrix = decision_capability_matrix()
    assert matrix["backends"]["native"]["available"] is True
    assert optimize_capability_matrix() == matrix
    assert "leakage_discipline" in matrix


def test_resolve_backend_native_knapsack() -> None:
    backend = resolve_backend(method="knapsack", backend="native")
    assert backend == "native"


def test_resolve_backend_defaults_f1_threshold_to_native() -> None:
    session = _binary_session()
    fit = session.fit_decision_policy(method="threshold", partition="validation")
    assert fit.backend in {None, "native"}
    assert fit.threshold is not None


def test_calibrated_threshold_session_smoke() -> None:
    session = _binary_session()
    fit = session.fit_decision_policy(
        method="threshold",
        backend="calibrated",
        partition="validation",
        fp_cost=1.0,
        fn_cost=4.0,
    )
    assert fit.backend == "calibrated"
    assert fit.threshold is not None
    ev = session.evaluate_decisions(partition="test")
    assert "f1" in ev.metrics


@pytest.mark.skipif(not xgboost_available(), reason="xgboost not installed")
def test_xgb_threshold_session_smoke() -> None:
    session = _binary_session()
    fit = session.fit_decision_policy(
        method="threshold",
        backend="xgb",
        partition="validation",
        fp_cost=1.0,
        fn_cost=6.0,
    )
    assert fit.backend == "xgb"
    applied = session.apply_decisions(partition="test")
    assert applied.n_rows > 0


@pytest.mark.skipif(not pulp_available(), reason="pulp not installed")
def test_pulp_knapsack_respects_budget() -> None:
    values = np.array([10.0, 9.0, 8.0, 7.0])
    costs = np.array([5.0, 5.0, 5.0, 5.0])
    result = select_knapsack_with_backend(
        values, costs, budget=7.0, backend="pulp"
    )
    assert result["selected_cost"] <= 7.0 + 1e-6
    assert result["solver_used"] == "pulp_mip"
    assert result["approximate"] is False


def test_missing_extra_raises_for_pulp_when_absent() -> None:
    if pulp_available():
        pytest.skip("pulp installed")
    with pytest.raises(MissingExtraError):
        resolve_backend(method="knapsack", backend="pulp")


def test_backend_unsupported_for_cost_matrix() -> None:
    session = _binary_session()
    with pytest.raises(ValidationError, match="not supported"):
        session.fit_decision_policy(
            method="cost_matrix",
            backend="pulp",
            partition="validation",
            cost_matrix=[[0, 1], [2, 0]],
            class_labels=["0", "1"],
        )


def test_walkthrough_includes_capability_matrix() -> None:
    session = _binary_session()
    session.fit_decision_policy(
        method="threshold",
        partition="validation",
        fp_cost=1.0,
        fn_cost=3.0,
    )
    status = session.walkthrough().decision_status
    assert "capability_matrix" in status
    assert status["capability_matrix"]["backends"]["native"]["available"]
