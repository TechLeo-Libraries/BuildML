"""CBC discovery must work across bundled and separately installed PuLP solvers."""

from types import SimpleNamespace

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.optimize.adapters.pulp_mip import (
    _cbc_solver,
    _require_optimal_status,
    select_knapsack_pulp,
)


@pytest.mark.parametrize("bundled", [False, True])
def test_cbc_uses_modern_discovery_before_legacy_path(bundled):
    calls = []

    def coin_cmd(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(available=lambda: True)

    pulp = SimpleNamespace(COIN_CMD=coin_cmd)
    if bundled:
        pulp.PULP_CBC_CMD = SimpleNamespace(pulp_cbc_path="legacy-cbc")
    _cbc_solver(pulp)
    assert calls == [{"msg": False}]


def test_cbc_uses_old_bundled_executable_without_deprecated_constructor():
    calls = []

    def coin_cmd(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(available=lambda: kwargs.get("path") == "legacy-cbc")

    pulp = SimpleNamespace(
        COIN_CMD=coin_cmd, PULP_CBC_CMD=SimpleNamespace(pulp_cbc_path="legacy-cbc")
    )
    _cbc_solver(pulp)
    assert calls == [{"msg": False}, {"path": "legacy-cbc", "msg": False}]


def test_cbc_missing_executable_has_actionable_error():
    pulp = SimpleNamespace(COIN_CMD=lambda **_: SimpleNamespace(available=lambda: False))
    with pytest.raises(ValidationError, match="pulp\\[cbc\\]"):
        _cbc_solver(pulp)


def test_unsolved_problem_cannot_claim_exact_optimum(monkeypatch):
    pulp = pytest.importorskip("pulp")
    result = (
        SimpleNamespace(status=pulp.LpSolveStatus.NotSolved, has_solution=False)
        if hasattr(pulp, "LpSolveStatus") else pulp.LpStatusNotSolved
    )
    monkeypatch.setattr(pulp.LpProblem, "solve", lambda *_: result)
    with pytest.raises(ValidationError, match="Not Solved"):
        select_knapsack_pulp(np.array([10.]), np.array([5.]), budget=7.)


def test_real_cbc_returns_optimal_selection():
    pytest.importorskip("pulp")
    result = select_knapsack_pulp(
        np.array([10., 7., 5.]), np.array([6., 4., 3.]), budget=7.,
    )
    assert result["selected_indices"] == (1, 2)
    assert result["selected_value"] == 12.
    assert result["selected_cost"] == 7.
    assert result["status"] == "Optimal"
    assert result["approximate"] is False


def test_pulp4_incumbents_and_missing_solutions_are_not_exact_optima():
    pulp = pytest.importorskip("pulp")
    if not hasattr(pulp, "LpSolveStatus"):
        pytest.skip("PuLP 4 solve statistics")
    for status in pulp.LpSolveStatus:
        for has_solution in (False, True):
            result = SimpleNamespace(status=status, has_solution=has_solution)
            if status == pulp.LpSolveStatus.Optimal and has_solution:
                assert _require_optimal_status(pulp, result) == "Optimal"
            else:
                with pytest.raises(ValidationError, match="failed with status"):
                    _require_optimal_status(pulp, result)


@pytest.mark.parametrize("status", [0, -1, -2, -3, 99])
def test_legacy_status_validation_rejects_nonoptimal_codes(status):
    legacy = SimpleNamespace(LpStatus={1: "Optimal", 0: "Not Solved"}, LpStatusOptimal=1)
    assert _require_optimal_status(legacy, 1) == "Optimal"
    with pytest.raises(ValidationError, match="failed with status"):
        _require_optimal_status(legacy, status)
