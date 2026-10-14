"""Positive-value zero-cost items remain feasible with no spending budget."""
import numpy as np
import pytest

from buildml.optimize.allocate import select_knapsack


@pytest.mark.parametrize('backend', ['dp', 'greedy', 'pulp', 'ortools'])
@pytest.mark.parametrize('minimum,expected', [(None, 8), (4, 5)])
def test_free_items_are_not_discarded_at_zero_budget(backend, minimum, expected):
    values = np.array([5., 9., 3.])
    costs = np.array([0., 1., 0.])
    if backend in {'dp', 'greedy'}:
        result = select_knapsack(values, costs, budget=0, solver=backend, min_score=minimum)
    elif backend == 'pulp':
        pytest.importorskip('pulp')
        from buildml.optimize.adapters.pulp_mip import select_knapsack_pulp
        result = select_knapsack_pulp(values, costs, budget=0, min_score=minimum)
    else:
        pytest.importorskip('ortools')
        from buildml.optimize.adapters.ortools_mip import select_knapsack_ortools
        result = select_knapsack_ortools(values, costs, budget=0, min_score=minimum)
    assert result['selected_value'] == expected
    assert result['selected_cost'] == 0
    assert set(result['selected_indices']) == ({0, 2} if minimum is None else {0})
