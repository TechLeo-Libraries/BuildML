"""0-1 knapsack via OR-Tools integer MIP (optimize-industry)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.optimize.extras import require_ortools


def select_knapsack_ortools(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve a 0-1 knapsack exactly with an OR-Tools integer MIP.

    Maximizes total value of selected items subject to a single cost budget
    using binary decision variables. Invoked when
    :func:`~buildml.optimize.allocate.select_knapsack_with_backend` resolves
    ``backend='ortools'``.

    Parameters
    ----------
    values:
        Non-negative item values to maximize.
    costs:
        Non-negative item costs aligned with ``values``.
    budget:
        Total cost budget; must be ``>= 0``.
    min_score:
        When set, exclude items below this value floor.
    ids:
        Optional identifier array aligned with ``values``; defaults to
        positional indices.

    Returns
    -------
    dict[str, Any]
        Selected indices, ids, unit fractions, aggregate value/cost, and
        solver/backend metadata.

    Raises
    ------
    ValidationError
        When inputs are misaligned, budgets are invalid, no MIP solver is
        available, or the solve terminates without a feasible/optimal status.
    """
    require_ortools()
    from ortools.linear_solver import pywraplp

    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n = int(values.size)
    if budget < 0:
        raise ValidationError("budget must be >= 0.")
    if costs.shape != values.shape:
        raise ValidationError("costs must align with values.")
    if (costs < 0).any():
        raise ValidationError("costs must be >= 0.")
    if ids is None:
        ids = np.arange(n)
    else:
        ids = np.asarray(ids)

    mask = np.isfinite(values) & np.isfinite(costs)
    if min_score is not None:
        mask &= values >= float(min_score)
    eligible = np.where(mask)[0]
    if eligible.size == 0 or budget == 0:
        return {
            "selected_indices": (),
            "selected_ids": (),
            "fractions": (),
            "n_selected": 0,
            "selected_value": 0.0,
            "selected_cost": 0.0,
            "solver_used": "ortools_mip",
            "approximate": False,
            "backend": "ortools",
        }

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise ValidationError("OR-Tools MIP solver (SCIP/CBC) unavailable.")

    x_vars = {
        int(i): solver.BoolVar(f"x_{i}") for i in eligible.tolist()
    }
    objective = solver.Objective()
    for i in eligible.tolist():
        objective.SetCoefficient(x_vars[int(i)], float(values[i]))
    objective.SetMaximization()

    constraint = solver.Constraint(0.0, float(budget))
    for i in eligible.tolist():
        constraint.SetCoefficient(x_vars[int(i)], float(costs[i]))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValidationError(f"OR-Tools knapsack MIP failed with status {status}.")

    chosen = [
        int(i) for i in eligible.tolist() if x_vars[int(i)].solution_value() > 0.5
    ]
    sel = np.asarray(chosen, dtype=int)
    fracs = [1.0] * len(chosen)
    return {
        "selected_indices": tuple(chosen),
        "selected_ids": tuple(ids[i] for i in chosen),
        "fractions": tuple(fracs),
        "n_selected": len(chosen),
        "selected_value": float(values[sel].sum()) if chosen else 0.0,
        "selected_cost": float(costs[sel].sum()) if chosen else 0.0,
        "solver_used": "ortools_mip",
        "approximate": False,
        "backend": "ortools",
        "status": int(status),
    }
