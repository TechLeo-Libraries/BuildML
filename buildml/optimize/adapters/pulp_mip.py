"""0-1 knapsack via PuLP integer MIP (optimize-industry)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.optimize.extras import require_pulp


def select_knapsack_pulp(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact 0-1 knapsack using PuLP binary variables and CBC."""
    pulp = require_pulp()
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
            "solver_used": "pulp_mip",
            "approximate": False,
            "backend": "pulp",
        }

    prob = pulp.LpProblem("buildml_knapsack", pulp.LpMaximize)
    x_vars = {
        int(i): pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in eligible.tolist()
    }
    prob += pulp.lpSum(float(values[i]) * x_vars[int(i)] for i in eligible.tolist())
    prob += (
        pulp.lpSum(float(costs[i]) * x_vars[int(i)] for i in eligible.tolist())
        <= float(budget)
    )
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] not in {"Optimal", "Not Solved"}:
        raise ValidationError(
            f"PuLP knapsack MIP failed with status {pulp.LpStatus[status]!r}."
        )

    chosen = [
        int(i)
        for i in eligible.tolist()
        if x_vars[int(i)].value is not None and float(x_vars[int(i)].value) > 0.5
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
        "solver_used": "pulp_mip",
        "approximate": False,
        "backend": "pulp",
        "status": pulp.LpStatus[status],
    }
