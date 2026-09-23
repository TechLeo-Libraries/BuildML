"""CBC discovery must work across bundled and separately installed PuLP solvers."""

from types import SimpleNamespace

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.optimize.adapters.pulp_mip import _cbc_solver, select_knapsack_pulp


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
    monkeypatch.setattr(pulp.LpProblem, "solve", lambda *_: pulp.LpStatusNotSolved)
    with pytest.raises(ValidationError, match="Not Solved"):
        select_knapsack_pulp(np.array([10.]), np.array([5.]), budget=7.)
