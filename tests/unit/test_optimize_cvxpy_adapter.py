"""Unit tests for CVXPY LP adapter (skipped when cvxpy unavailable)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from buildml.optimize.allocate import select_lp_allocate_with_backend


@pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None,
    reason="cvxpy not installed",
)
def test_cvxpy_lp_respects_budget() -> None:
    cvxpy = pytest.importorskip("cvxpy")
    del cvxpy
    values = np.array([10.0, 9.0, 8.0, 1.0])
    costs = np.array([5.0, 5.0, 5.0, 5.0])
    result = select_lp_allocate_with_backend(
        values, costs, budget=7.5, backend="cvxpy", max_fraction=1.0
    )
    assert result["selected_cost"] <= 7.5 + 1e-6
    assert result["solver_used"] == "cvxpy"
