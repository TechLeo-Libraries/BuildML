"""Thin Session facades over buildml.optimize (decision / optimisation helpers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.optimize.apply import apply_decisions
from buildml.optimize.checkpoint import load_decision_bundle, save_decision_bundle
from buildml.optimize.evaluate import evaluate_decisions
from buildml.optimize.explain_hooks import (
    apply_result_summary,
    eval_result_summary,
    fit_result_summary,
)
from buildml.optimize.fit import fit_decision_policy
from buildml.optimize.types import (
    AllocationObjective,
    DecisionMethod,
    KnapsackSolver,
    ScoreSource,
    TuningPartition,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_decision_policy_op(
    session,
    *,
    method: DecisionMethod = "threshold",
    backend: str | None = None,
    partition: TuningPartition = "validation",
    allow_test_tuning: bool = False,
    fp_cost: float | None = None,
    fn_cost: float | None = None,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    cost_matrix: Sequence[Sequence[float]] | None = None,
    class_labels: list[str] | None = None,
    capacity: int | None = None,
    budget: float | None = None,
    score_source: ScoreSource = "model_proba",
    score_column: str | None = None,
    cost_column: str | None = None,
    value_column: str | None = None,
    id_column: str | None = None,
    knapsack_solver: KnapsackSolver = "dp",
    objective: AllocationObjective = "maximize_score",
    min_score: float | None = None,
    lp_max_fraction: float = 1.0,
):
    """Fit a decision policy on train/validation (test requires opt-in).

    Notes
    -----
    **Leakage:** Defaults to ``partition='validation'``. Tuning on Session
    test requires ``allow_test_tuning=True`` and emits a dangerous-opt-in
    disclosure. ``method='threshold'`` wraps the same engine as
    :meth:`Session.tune_threshold` and also updates ``last_diagnostic``.
    """
    if session._split_plan is None:
        raise ValidationError(
            "A split is required before fit_decision_policy. Call split(...) first."
        )
    plan, result, diagnostic = fit_decision_policy(
        session.dataset,
        session._split_plan,
        session._fit_result,
        method=method,
        backend=backend,  # type: ignore[arg-type]
        partition=partition,
        allow_test_tuning=allow_test_tuning,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        cost_matrix=cost_matrix,
        class_labels=class_labels,
        capacity=capacity,
        budget=budget,
        score_source=score_source,
        score_column=score_column,
        cost_column=cost_column,
        value_column=value_column,
        id_column=id_column,
        knapsack_solver=knapsack_solver,
        objective=objective,
        min_score=min_score,
        lp_max_fraction=lp_max_fraction,
    )
    session._decision_plan = plan
    session._decision_fit_result = result
    session._decision_eval_result = None
    session._decision_apply_result = None
    if diagnostic is not None:
        session._last_diagnostic = diagnostic
    session._record(
        "fit_decision_policy",
        {
            "method": method,
            "backend": backend,
            "partition": partition,
            "allow_test_tuning": allow_test_tuning,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "tp_benefit": tp_benefit,
            "tn_benefit": tn_benefit,
            "cost_matrix": None if cost_matrix is None else [list(r) for r in cost_matrix],
            "class_labels": class_labels,
            "capacity": capacity,
            "budget": budget,
            "score_source": score_source,
            "score_column": score_column,
            "cost_column": cost_column,
            "value_column": value_column,
            "id_column": id_column,
            "knapsack_solver": knapsack_solver,
            "objective": objective,
            "min_score": min_score,
            "lp_max_fraction": lp_max_fraction,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def apply_decisions_op(
    session,
    *,
    partition: PartitionOrAll | None = "test",
    candidates: pd.DataFrame | None = None,
):
    """Apply the frozen DecisionPlan to a partition or candidate frame."""
    plan = getattr(session, "_decision_plan", None)
    if plan is None:
        raise ValidationError(
            "No DecisionPlan. Call fit_decision_policy(...) first."
        )
    result = apply_decisions(
        session.dataset,
        session._split_plan,
        session._fit_result,
        plan,
        partition=None if partition is None else str(partition),
        candidates=candidates,
    )
    session._decision_apply_result = result
    session._record(
        "apply_decisions",
        {
            "partition": partition,
            "candidates": None if candidates is None else f"DataFrame(n={len(candidates)})",
        },
        warnings=tuple(result.warnings),
        result_summary=apply_result_summary(result),
    )
    return result


def evaluate_decisions_op(
    session,
    *,
    partition: PartitionName = "test",
):
    """Evaluate the frozen DecisionPlan on a holdout partition."""
    plan = getattr(session, "_decision_plan", None)
    if plan is None:
        raise ValidationError(
            "No DecisionPlan. Call fit_decision_policy(...) first."
        )
    result = evaluate_decisions(
        session.dataset,
        session._split_plan,
        session._fit_result,
        plan,
        partition=str(partition),
    )
    session._decision_eval_result = result
    session._record(
        "evaluate_decisions",
        {"partition": partition},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_decision_bundle_op(session, path: str | Path) -> Path:
    plan = getattr(session, "_decision_plan", None)
    if plan is None:
        raise ValidationError("No DecisionPlan to save.")
    destination = save_decision_bundle(
        path,
        plan,
        fit_result=getattr(session, "_decision_fit_result", None),
        eval_result=getattr(session, "_decision_eval_result", None),
        apply_result=getattr(session, "_decision_apply_result", None),
    )
    session._record(
        "save_decision_bundle",
        {"path": str(destination)},
        result_summary={"path": str(destination), "format": "buildml.decision_bundle.v1"},
    )
    return destination


def load_decision_bundle_op(session, path: str | Path):
    plan = load_decision_bundle(path)
    session._decision_plan = plan
    session._decision_fit_result = None
    session._decision_eval_result = None
    session._decision_apply_result = None
    session._record(
        "load_decision_bundle",
        {"path": str(path)},
        result_summary={"path": str(path), "method": plan.method},
    )
    return session


def decision_capability_matrix_op() -> dict[str, Any]:
    from buildml.optimize.catalog import decision_capability_matrix

    return decision_capability_matrix()
