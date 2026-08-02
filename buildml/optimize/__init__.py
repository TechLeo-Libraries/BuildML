"""Optimisation / decision helpers (Session-shaped policies over ML scores).

Phase coverage (internal tracker — depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1–2 complete. Phase 3 — Application systems:
  Recommendation systems (**PASS**).
  Search / LTR (**PASS**).
  Knowledge graphs (**PASS**).
  Optimisation / decision helpers (**PASS** — this module; R6.9 industry depth).
  Synthetic-data systems (**PASS**).

Honesty (this package):
  - Decision helpers for ML scores, costs, and constrained allocations.
  - **Not** a general operations-research platform, MIP suite, or digital twin.
  - Native fallback: threshold_report, numpy knapsack DP/greedy, scipy linprog.
  - Industry depth via ``buildml[optimize-industry]``: PuLP/OR-Tools 0-1 knapsack
    MIP, CVXPY convex LP, XGB cost-sensitive thresholds, sklearn calibration.
  - Never tunes on Session test without ``allow_test_tuning=True`` + disclosure.

Lazy imports — keep the core import graph light.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "ApplyDecisionsResult",
    "CostModel",
    "DecisionConfig",
    "DecisionEvalResult",
    "DecisionFitResult",
    "DecisionMethod",
    "DecisionPlan",
    "apply_decisions",
    "decision_capability_matrix",
    "decision_status",
    "decision_status_for_session",
    "evaluate_decisions",
    "fit_decision_policy",
    "load_decision_bundle",
    "optimize_capability_matrix",
    "save_decision_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DecisionConfig",
        "DecisionMethod",
        "DecisionBackend",
        "CostModel",
        "TuningPartition",
        "ScoreSource",
        "KnapsackSolver",
        "AllocationObjective",
    }:
        from buildml.optimize import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "DecisionPlan",
        "DecisionFitResult",
        "ApplyDecisionsResult",
        "DecisionEvalResult",
    }:
        from buildml.optimize import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_decision_policy":
        from buildml.optimize.fit import fit_decision_policy

        return fit_decision_policy
    if name == "apply_decisions":
        from buildml.optimize.apply import apply_decisions

        return apply_decisions
    if name == "evaluate_decisions":
        from buildml.optimize.evaluate import evaluate_decisions

        return evaluate_decisions
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_decision_bundle",
        "load_decision_bundle",
    }:
        from buildml.optimize import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"decision_status", "decision_status_for_session"}:
        from buildml.optimize import explain_hooks as hooks

        return getattr(hooks, name)
    if name in {"decision_capability_matrix", "optimize_capability_matrix"}:
        from buildml.optimize.catalog import (
            decision_capability_matrix,
            optimize_capability_matrix,
        )

        return (
            decision_capability_matrix
            if name == "decision_capability_matrix"
            else optimize_capability_matrix
        )
    raise AttributeError(f"module 'buildml.optimize' has no attribute {name!r}")
