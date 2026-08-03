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
    """Fit a decision policy on train or validation without refitting the model.

    Delegates to :func:`buildml.optimize.fit.fit_decision_policy`, stores the
    :class:`~buildml.optimize.results.DecisionPlan` on Session, and records
    the fit. Follow with :func:`apply_decisions_op` or
    :func:`evaluate_decisions_op`.

    Parameters
    ----------
    session:
        Active Session with dataset, split plan, and fitted supervised model.
    method:
        Decision strategy (``threshold``, ``knapsack``, ``lp``, etc.).
    backend:
        Optional solver backend override for MIP/LP methods.
    partition:
        Partition used for threshold/allocation tuning (default ``validation``).
    allow_test_tuning:
        When False, refuse tuning on the test partition.
    fp_cost, fn_cost:
        False-positive and false-negative costs for threshold tuning.
    tp_benefit, tn_benefit:
        True-positive and true-negative benefits for cost-sensitive tuning.
    cost_matrix:
        Optional multi-class cost matrix for threshold methods.
    class_labels:
        Class label order matching ``cost_matrix`` rows/columns.
    capacity:
        Maximum selections for knapsack-style allocation.
    budget:
        Total budget for knapsack or LP allocation methods.
    score_source:
        Where decision scores come from (model probabilities, raw scores, etc.).
    score_column:
        Explicit column for scores when ``score_source`` is column-based.
    cost_column:
        Per-row cost column for knapsack/LP methods.
    value_column:
        Per-row value column for knapsack/LP methods.
    id_column:
        Row identifier column for allocation output.
    knapsack_solver:
        Knapsack solver (``dp`` or ``pulp``).
    objective:
        Allocation objective (maximize score, minimize cost, etc.).
    min_score:
        Minimum score cutoff before allocation.
    lp_max_fraction:
        Maximum fraction of budget any single item may consume in LP mode.

    Returns
    -------
    DecisionFitResult
        Serializable fit summary including tuned threshold or allocation.
        Use :func:`apply_decisions_op` to apply the frozen plan.

    Raises
    ------
    ValidationError
        When no split plan exists on the Session.

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
    """Apply the frozen DecisionPlan to a partition or candidate frame.

    Delegates to :func:`buildml.optimize.apply.apply_decisions` using the
    plan from :func:`fit_decision_policy_op`. Stores apply results on Session
    and records the operation.

    Parameters
    ----------
    session:
        Active Session with a DecisionPlan from :func:`fit_decision_policy_op`.
    partition:
        Split partition to apply decisions to (``train``, ``validation``,
        ``test``, or ``all``). Ignored when ``candidates`` is provided.
    candidates:
        Optional explicit candidate frame instead of a Session partition.

    Returns
    -------
    DecisionApplyResult
        Selected rows, scores, and allocation metadata for the partition.

    Raises
    ------
    ValidationError
        When no DecisionPlan exists on the Session.
    """
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
    """Evaluate the frozen DecisionPlan on a holdout partition.

    Delegates to :func:`buildml.optimize.evaluate.evaluate_decisions` and
    stores evaluation metrics on Session. Requires a prior
    :func:`fit_decision_policy_op`.

    Parameters
    ----------
    session:
        Active Session with a DecisionPlan from :func:`fit_decision_policy_op`.
    partition:
        Holdout partition for evaluation (default ``test``).

    Returns
    -------
    DecisionEvalResult
        Cost, benefit, and confusion-style metrics for the frozen plan.

    Raises
    ------
    ValidationError
        When no DecisionPlan exists on the Session.
    """
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
    """Persist the active DecisionPlan as ``buildml.decision_bundle.v1``.

    Delegates to :func:`buildml.optimize.checkpoint.save_decision_bundle`.
    Reload with :func:`load_decision_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a DecisionPlan from :func:`fit_decision_policy_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no DecisionPlan exists on the Session.
    """
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
    """Load a decision bundle into this Session.

    Delegates to :func:`buildml.optimize.checkpoint.load_decision_bundle`
    and clears prior fit/eval/apply results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded DecisionPlan.
    path:
        Path to a ``buildml.decision_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with DecisionPlan attached for chaining.
    """
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
    """Return the decision/optimization capability matrix for this install.

    Delegates to :func:`buildml.optimize.catalog.decision_capability_matrix`.
    Use before :func:`fit_decision_policy_op` to confirm ``method`` and
    ``backend`` pairs available with current extras.

    Returns
    -------
    dict
        Nested map of method identifiers to supported backends and options.
    """
    from buildml.optimize.catalog import decision_capability_matrix

    return decision_capability_matrix()
