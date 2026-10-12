"""Industry knapsack objectives agree with exhaustive small-problem solutions."""
import itertools

import numpy as np
import pytest

from buildml.optimize.adapters.ortools_mip import select_knapsack_ortools


@pytest.mark.parametrize('budget', [0, 1, 4.5, 8, 100])
def test_ortools_matches_exhaustive_optimum(budget):
    pytest.importorskip('ortools')
    values = np.array([3., 7., 9., 2., 4.])
    costs = np.array([1., 3., 4., 2., 0.5])
    result = select_knapsack_ortools(values, costs, budget=budget, ids=np.array(list('abcde')))
    feasible = [sum(values[i] for i in indices) for size in range(6)
                for indices in itertools.combinations(range(5), size)
                if sum(costs[i] for i in indices) <= budget]
    assert result['selected_value'] == pytest.approx(max(feasible))
    assert result['selected_cost'] <= budget
    assert result['approximate'] is False
    assert result['selected_ids'] == tuple('abcde'[i] for i in result['selected_indices'])


def test_feasible_solver_status_does_not_claim_proven_optimality(monkeypatch):
    pytest.importorskip('ortools')
    from ortools.linear_solver import pywraplp

    create = pywraplp.Solver.CreateSolver
    class FeasibleStatus:
        def __init__(self, solver):
            self.solver = solver
        def __getattr__(self, name):
            return getattr(self.solver, name)
        def Solve(self):
            assert self.solver.Solve() == pywraplp.Solver.OPTIMAL
            return pywraplp.Solver.FEASIBLE
    monkeypatch.setattr(pywraplp.Solver, 'CreateSolver', lambda name: FeasibleStatus(create(name)))
    result = select_knapsack_ortools(np.array([3., 5.]), np.array([1., 2.]), budget=2)
    assert result['selected_cost'] <= 2
    assert result['approximate'] is True
    assert result['status'] == pywraplp.Solver.FEASIBLE



def test_broken_native_solver_has_an_actionable_error(monkeypatch):
    import builtins

    monkeypatch.setattr("buildml.optimize.adapters.ortools_mip.require_ortools", lambda: None)
    original = builtins.__import__
    def importer(name, *args, **kwargs):
        if name == "ortools.linear_solver":
            raise ImportError("undefined symbol from incompatible native library")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importer)
    from buildml.core.errors import ValidationError
    with pytest.raises(ValidationError, match=r"buildml\[optimize-industry\]"):
        select_knapsack_ortools(np.array([3.]), np.array([1.]), budget=2.)
