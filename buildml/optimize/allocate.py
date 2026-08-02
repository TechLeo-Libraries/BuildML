"""Constrained selection helpers: top-K, knapsack-lite, and LP allocation."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.optimize.catalog import DecisionBackendName, resolve_backend


def select_topk(
    scores: np.ndarray,
    *,
    capacity: int,
    costs: np.ndarray | None = None,
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select up to ``capacity`` highest-scoring candidates (optional floor)."""
    scores = np.asarray(scores, dtype=float)
    n = int(scores.size)
    if capacity < 1:
        raise ValidationError("capacity must be >= 1 for method='topk'.")
    if costs is None:
        costs = np.ones(n, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if costs.shape != scores.shape:
            raise ValidationError("costs must align with scores.")
    if ids is None:
        ids = np.arange(n)
    else:
        ids = np.asarray(ids)

    mask = np.isfinite(scores)
    if min_score is not None:
        mask &= scores >= float(min_score)
    eligible = np.where(mask)[0]
    if eligible.size == 0:
        return {
            "selected_indices": (),
            "selected_ids": (),
            "fractions": (),
            "n_selected": 0,
            "selected_value": 0.0,
            "selected_cost": 0.0,
        }
    order = eligible[np.argsort(-scores[eligible], kind="mergesort")]
    chosen = order[: int(capacity)]
    fracs = np.ones(chosen.size, dtype=float)
    return {
        "selected_indices": tuple(int(i) for i in chosen.tolist()),
        "selected_ids": tuple(ids[i] for i in chosen.tolist()),
        "fractions": tuple(float(f) for f in fracs.tolist()),
        "n_selected": int(chosen.size),
        "selected_value": float(scores[chosen].sum()),
        "selected_cost": float(costs[chosen].sum()),
    }


def select_knapsack_with_backend(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    backend: DecisionBackendName | None = None,
    solver: str = "dp",
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Route knapsack selection to native DP/greedy or industry MIP backends."""
    resolved = resolve_backend(method="knapsack", backend=backend)
    if resolved == "pulp":
        from buildml.optimize.adapters.pulp_mip import select_knapsack_pulp

        return select_knapsack_pulp(
            values, costs, budget=float(budget), min_score=min_score, ids=ids
        )
    if resolved == "ortools":
        from buildml.optimize.adapters.ortools_mip import select_knapsack_ortools

        return select_knapsack_ortools(
            values, costs, budget=float(budget), min_score=min_score, ids=ids
        )
    return select_knapsack(
        values,
        costs,
        budget=float(budget),
        solver=solver,
        min_score=min_score,
        ids=ids,
    )


def select_lp_allocate_with_backend(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    backend: DecisionBackendName | None = None,
    max_fraction: float = 1.0,
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Route LP allocation to scipy linprog or CVXPY."""
    resolved = resolve_backend(method="lp_allocate", backend=backend)
    if resolved == "cvxpy":
        from buildml.optimize.adapters.cvxpy_lp import select_lp_allocate_cvxpy

        return select_lp_allocate_cvxpy(
            values,
            costs,
            budget=float(budget),
            max_fraction=float(max_fraction),
            min_score=min_score,
            ids=ids,
        )
    return select_lp_allocate(
        values,
        costs,
        budget=float(budget),
        max_fraction=float(max_fraction),
        min_score=min_score,
        ids=ids,
    )


def select_knapsack(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    solver: str = "dp",
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """0-1 knapsack-lite: maximize value under a budget.

    ``solver='dp'`` uses exact integer DP after scaling costs to cents when
    costs are near-integral; falls back to value/cost greedy when the scaled
    state space would exceed a hard bound. ``solver='greedy'`` always uses
    density greedy (disclose approximation).
    """
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
            "solver_used": solver,
            "approximate": False,
        }

    if solver == "greedy":
        return _greedy_knapsack(
            values, costs, budget=float(budget), eligible=eligible, ids=ids
        )
    if solver != "dp":
        raise ValidationError("knapsack_solver must be 'dp' or 'greedy'.")

    # Scale costs to integers when feasible
    sub_costs = costs[eligible]
    sub_values = values[eligible]
    scale, int_costs, int_budget, exact = _integerize_costs(sub_costs, float(budget))
    # Cap DP state: ~5e6 cells soft limit
    if exact and int_budget <= 500_000 and eligible.size * (int_budget + 1) <= 5_000_000:
        chosen_local = _dp_knapsack(sub_values, int_costs, int_budget)
        chosen = eligible[chosen_local]
        fracs = np.ones(chosen.size, dtype=float)
        return {
            "selected_indices": tuple(int(i) for i in chosen.tolist()),
            "selected_ids": tuple(ids[i] for i in chosen.tolist()),
            "fractions": tuple(float(f) for f in fracs.tolist()),
            "n_selected": int(chosen.size),
            "selected_value": float(values[chosen].sum()),
            "selected_cost": float(costs[chosen].sum()),
            "solver_used": "dp",
            "approximate": False,
            "cost_scale": float(scale),
        }
    # Fallback greedy with disclosure
    result = _greedy_knapsack(
        values, costs, budget=float(budget), eligible=eligible, ids=ids
    )
    result["solver_used"] = "greedy"
    result["approximate"] = True
    result["cost_scale"] = float(scale)
    return result


def select_lp_allocate(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    max_fraction: float = 1.0,
    min_score: float | None = None,
    ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Continuous budget allocation via ``scipy.optimize.linprog``.

    Maximizes Σ value_i * x_i subject to Σ cost_i * x_i ≤ budget and
    0 ≤ x_i ≤ max_fraction. This is a fractional knapsack / portfolio-lite
    helper — not a general OR / MIP platform (no PuLP / OR-Tools).
    """
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
    # Zero-cost positive-value items take full fraction for free
    free_mask = np.isfinite(values) & np.isfinite(costs) & (costs == 0) & (values > 0)
    if min_score is not None:
        free_mask &= values >= float(min_score)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover
        raise ValidationError(
            "method='lp_allocate' requires scipy.optimize (transitive via "
            "scikit-learn)."
        ) from exc

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
            "solver_used": "linprog",
            "approximate": False,
            "status": "no_positive_cost_items",
        }

    c = -values[eligible]  # maximize
    a_ub = costs[eligible][None, :]
    b_ub = np.array([float(budget)], dtype=float)
    bounds = [(0.0, float(max_fraction))] * int(eligible.size)
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValidationError(
            f"LP allocation failed: {result.message}. "
            "Check budget / costs / values."
        )
    fractions[eligible] = np.asarray(result.x, dtype=float)
    chosen = np.where(fractions > 1e-12)[0]
    order = chosen[np.argsort(-values[chosen] * fractions[chosen], kind="mergesort")]
    return {
        "selected_indices": tuple(int(i) for i in order.tolist()),
        "selected_ids": tuple(ids[i] for i in order.tolist()),
        "fractions": tuple(float(fractions[i]) for i in order.tolist()),
        "n_selected": int(order.size),
        "selected_value": float((values * fractions).sum()),
        "selected_cost": float((costs * fractions).sum()),
        "solver_used": "linprog",
        "approximate": False,
        "status": str(result.message),
    }


def _greedy_knapsack(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    budget: float,
    eligible: np.ndarray,
    ids: np.ndarray,
) -> dict[str, Any]:
    density = np.where(
        costs[eligible] > 0,
        values[eligible] / costs[eligible],
        np.where(values[eligible] > 0, np.inf, -np.inf),
    )
    order = eligible[np.argsort(-density, kind="mergesort")]
    chosen: list[int] = []
    spent = 0.0
    for idx in order.tolist():
        c = float(costs[idx])
        if c <= budget - spent + 1e-12:
            chosen.append(int(idx))
            spent += c
    fracs = [1.0] * len(chosen)
    sel = np.asarray(chosen, dtype=int)
    return {
        "selected_indices": tuple(chosen),
        "selected_ids": tuple(ids[i] for i in chosen),
        "fractions": tuple(fracs),
        "n_selected": len(chosen),
        "selected_value": float(values[sel].sum()) if chosen else 0.0,
        "selected_cost": float(costs[sel].sum()) if chosen else 0.0,
        "solver_used": "greedy",
        "approximate": True,
    }


def _integerize_costs(
    costs: np.ndarray, budget: float
) -> tuple[float, np.ndarray, int, bool]:
    """Scale costs toward integers; return (scale, int_costs, int_budget, exact)."""
    # Prefer scale=100 for currency-like costs; try scale=1 first
    for scale in (1.0, 10.0, 100.0, 1000.0):
        scaled = np.rint(costs * scale)
        scaled_budget = int(np.floor(budget * scale + 1e-9))
        if np.max(np.abs(costs * scale - scaled)) <= 1e-6:
            return scale, scaled.astype(int), scaled_budget, True
    # Not near-integral — mark inexact (caller may fall back)
    scale = 100.0
    scaled = np.maximum(np.rint(costs * scale), 0).astype(int)
    scaled_budget = int(np.floor(budget * scale + 1e-9))
    return scale, scaled, scaled_budget, False


def _dp_knapsack(
    values: np.ndarray, int_costs: np.ndarray, int_budget: int
) -> np.ndarray:
    """Exact 0-1 knapsack; returns local indices into the eligible arrays."""
    n = int(values.size)
    free = [i for i in range(n) if int(int_costs[i]) == 0 and float(values[i]) > 0]
    paid = [i for i in range(n) if int(int_costs[i]) > 0]
    dp = np.zeros(int_budget + 1, dtype=float)
    take = np.zeros((len(paid), int_budget + 1), dtype=bool)
    for row, i in enumerate(paid):
        w = int(int_costs[i])
        v = float(values[i])
        if w > int_budget:
            continue
        for b in range(int_budget, w - 1, -1):
            cand = dp[b - w] + v
            if cand > dp[b] + 1e-15:
                dp[b] = cand
                take[row, b] = True
    chosen: list[int] = list(free)
    b = int_budget
    for row in range(len(paid) - 1, -1, -1):
        i = paid[row]
        w = int(int_costs[i])
        if b >= w and take[row, b]:
            chosen.append(i)
            b -= w
    chosen.sort()
    return np.asarray(chosen, dtype=int)
