"""0-1 knapsack via PuLP integer MIP (optimize-industry)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.optimize.extras import require_pulp


def _cbc_solver(pulp: Any) -> Any:
    """Find CBC via modern discovery or an older PuLP bundled executable."""
    solver = pulp.COIN_CMD(msg=False)
    if solver.available():
        return solver
    # PuLP 2.x/3.x shipped CBC; passing its path avoids constructing the
    # deprecated PULP_CBC_CMD wrapper. PuLP 4 discovers pulp[cbc] directly.
    bundled_path = getattr(getattr(pulp, "PULP_CBC_CMD", None), "pulp_cbc_path", None)
    if bundled_path:
        solver = pulp.COIN_CMD(path=bundled_path, msg=False)
        if solver.available():
            return solver
    raise ValidationError(
        "CBC solver is unavailable. Install 'pulp[cbc]' or put the CBC executable on PATH."
    )


def _pulp_selected(pulp: Any, variable: Any) -> bool:
    """Return whether a binary PuLP variable is selected after solve.

    ``LpVariable.value`` is a method on current PuLP. ``pulp.value`` is the
    portable reader.
    """
    raw = pulp.value(variable)
    return raw is not None and float(raw) > 0.5


def select_knapsack_pulp(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve a 0-1 knapsack exactly with PuLP and CBC.

    Maximizes total value under a single cost budget using binary integer
    variables. Invoked when
    :func:`~buildml.optimize.allocate.select_knapsack_with_backend` resolves
    ``backend='pulp'``.

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
        When inputs are misaligned, budgets are invalid, or CBC returns a
        non-optimal status.
    """
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
    make_variable = getattr(prob, "add_variable", None)
    if make_variable is None:
        make_variable = pulp.LpVariable
    x_vars = {
        int(i): make_variable(f"x_{i}", cat=pulp.LpBinary) for i in eligible.tolist()
    }
    prob += pulp.lpSum(float(values[i]) * x_vars[int(i)] for i in eligible.tolist())
    prob += (
        pulp.lpSum(float(costs[i]) * x_vars[int(i)] for i in eligible.tolist())
        <= float(budget)
    )
    status = prob.solve(_cbc_solver(pulp))
    if pulp.LpStatus[status] != "Optimal":
        raise ValidationError(
            f"PuLP knapsack MIP failed with status {pulp.LpStatus[status]!r}."
        )

    chosen = [
        int(i)
        for i in eligible.tolist()
        if _pulp_selected(pulp, x_vars[int(i)])
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
