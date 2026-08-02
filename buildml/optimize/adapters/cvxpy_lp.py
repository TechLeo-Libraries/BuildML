"""Continuous LP allocation via CVXPY (optimize-industry)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.optimize.extras import require_cvxpy


def select_lp_allocate_cvxpy(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    max_fraction: float = 1.0,
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fractional budget allocation via CVXPY convex LP."""
    cp = require_cvxpy()
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n = int(values.size)
    if budget < 0:
        raise ValidationError("budget must be >= 0.")
    if not (0.0 < float(max_fraction) <= 1.0):
        raise ValidationError("lp_max_fraction must be in (0, 1].")
    if costs.shape != values.shape:
        raise ValidationError("costs must align with values.")
    if (costs < 0).any():
        raise ValidationError("costs must be >= 0.")
    if ids is None:
        ids = np.arange(n)
    else:
        ids = np.asarray(ids)

    mask = np.isfinite(values) & np.isfinite(costs) & (costs > 0)
    if min_score is not None:
        mask &= values >= float(min_score)
    free_mask = np.isfinite(values) & np.isfinite(costs) & (costs == 0) & (values > 0)
    if min_score is not None:
        free_mask &= values >= float(min_score)

    fractions = np.zeros(n, dtype=float)
    free_idx = np.where(free_mask)[0]
    fractions[free_idx] = float(max_fraction)

    eligible = np.where(mask)[0]
    if eligible.size == 0:
        chosen = np.where(fractions > 1e-12)[0]
        return {
            "selected_indices": tuple(int(i) for i in chosen.tolist()),
            "selected_ids": tuple(ids[i] for i in chosen.tolist()),
            "fractions": tuple(float(fractions[i]) for i in chosen.tolist()),
            "n_selected": int(chosen.size),
            "selected_value": float((values * fractions).sum()),
            "selected_cost": float((costs * fractions).sum()),
            "solver_used": "cvxpy",
            "approximate": False,
            "backend": "cvxpy",
            "status": "no_positive_cost_items",
        }

    sub_values = values[eligible]
    sub_costs = costs[eligible]
    x = cp.Variable(len(eligible))
    objective = cp.Maximize(sub_values @ x)
    constraints = [
        sub_costs @ x <= float(budget),
        x >= 0.0,
        x <= float(max_fraction),
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except cp.error.SolverError as exc:
        raise ValidationError(f"CVXPY LP allocation failed: {exc}.") from exc

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValidationError(
            f"CVXPY LP allocation failed with status {problem.status!r}."
        )

    solution = np.asarray(x.value, dtype=float).reshape(-1)
    fractions[eligible] = np.clip(solution, 0.0, float(max_fraction))
    chosen = np.where(fractions > 1e-12)[0]
    order = chosen[np.argsort(-values[chosen] * fractions[chosen], kind="mergesort")]
    return {
        "selected_indices": tuple(int(i) for i in order.tolist()),
        "selected_ids": tuple(ids[i] for i in order.tolist()),
        "fractions": tuple(float(fractions[i]) for i in order.tolist()),
        "n_selected": int(order.size),
        "selected_value": float((values * fractions).sum()),
        "selected_cost": float((costs * fractions).sum()),
        "solver_used": "cvxpy",
        "approximate": False,
        "backend": "cvxpy",
        "status": str(problem.status),
    }
