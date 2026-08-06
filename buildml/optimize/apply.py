"""Apply a frozen DecisionPlan to a partition or candidate frame."""

from __future__ import annotations

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.supervised import FitResult
from buildml.optimize.allocate import (
    select_knapsack_with_backend,
    select_lp_allocate_with_backend,
    select_topk,
)
from buildml.optimize.features import partition_frame, require_split
from buildml.optimize.fit import _resolve_allocation_inputs
from buildml.optimize.policies import apply_cost_matrix_policy, apply_threshold_policy
from buildml.optimize.results import ApplyDecisionsResult, DecisionPlan


def apply_decisions(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult | None,
    plan: DecisionPlan,
    *,
    partition: str | None = "test",
    candidates: pd.DataFrame | None = None,
) -> ApplyDecisionsResult:
    """Apply a frozen decision policy to a partition or candidate frame.

    For threshold and cost-matrix methods, scores the Session partition with
    the fitted estimator and applies the stored operating rule. For allocation
    methods, ranks candidates from model scores or explicit columns and runs
    top-K, knapsack, or LP selection. Follow
    :func:`~buildml.optimize.fit.fit_decision_policy` and precede
    :func:`~buildml.optimize.evaluate.evaluate_decisions` on holdout data.

    Parameters
    ----------
    dataset:
        Tabular data containing features (and target when evaluating labels).
    split_plan:
        Split plan for partition scoring; required unless ``candidates`` is
        supplied for column-driven allocation.
    fit_result:
        Session fit result for model-scored threshold/cost/allocation paths.
    plan:
        Frozen :class:`~buildml.optimize.results.DecisionPlan` from fit.
    partition:
        Split name to score; defaults to ``'test'``. Ignored when
        ``candidates`` is provided.
    candidates:
        Optional explicit candidate frame for column-driven allocation.

    Returns
    -------
    ApplyDecisionsResult
        Per-row decisions or selected ids with scores, counts, and honesty
        disclosures.

    Raises
    ------
    ValidationError
        When no plan is attached, required fit/split inputs are missing, or
        plan fields needed for the method are absent.
    """
    if plan is None:
        raise ValidationError("No DecisionPlan. Call fit_decision_policy(...) first.")

    warnings = list(plan.warnings)
    disclosures = [
        f"Applying frozen DecisionPlan method={plan.method!r} "
        f"(fitted on {plan.partition_fitted}).",
        *list(plan.disclosures[:4]),
    ]

    if plan.method == "threshold":
        if fit_result is None:
            raise ValidationError("apply_decisions(threshold) requires Session.fit(...).")
        split = require_split(split_plan)
        part = partition or "test"
        labels, scores, _index = apply_threshold_policy(
            dataset, split, fit_result, plan, partition=part
        )
        n_sel = int((pd.Series(labels).astype(str) == str(plan.positive_class)).sum())
        return ApplyDecisionsResult(
            method=plan.method,
            partition=part,
            n_rows=int(len(labels)),
            n_selected=n_sel,
            decisions=tuple(labels.tolist()),
            scores=tuple(float(s) for s in scores.tolist()),
            threshold=plan.threshold,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    if plan.method == "cost_matrix":
        if fit_result is None:
            raise ValidationError(
                "apply_decisions(cost_matrix) requires Session.fit(...)."
            )
        split = require_split(split_plan)
        part = partition or "test"
        actions, scores, _index = apply_cost_matrix_policy(
            dataset, split, fit_result, plan, partition=part
        )
        return ApplyDecisionsResult(
            method=plan.method,
            partition=part,
            n_rows=int(len(actions)),
            n_selected=int(len(actions)),
            decisions=tuple(actions.tolist()),
            scores=tuple(float(s) for s in scores.tolist()),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    return _apply_allocation(
        dataset,
        split_plan,
        fit_result,
        plan,
        partition=partition,
        candidates=candidates,
        disclosures=disclosures,
        warnings=warnings,
    )


def _apply_allocation(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult | None,
    plan: DecisionPlan,
    *,
    partition: str | None,
    candidates: pd.DataFrame | None,
    disclosures: list[str],
    warnings: list[str],
) -> ApplyDecisionsResult:
    if candidates is not None:
        frame = candidates.copy()
        part_name: str | None = None
        # Column-driven on explicit candidates
        from buildml.optimize.features import column_scores

        ids, values, costs, _pos = column_scores(
            frame,
            score_column=plan.score_column,
            cost_column=plan.cost_column,
            value_column=plan.value_column or plan.score_column,
            id_column=plan.id_column,
        )
    else:
        split = require_split(split_plan)
        part_name = partition or "test"
        frame = partition_frame(dataset, split, part_name)
        values, costs, ids, _src = _resolve_allocation_inputs(
            dataset,
            split,
            fit_result,
            frame,
            partition=part_name,
            score_source=plan.score_source,
            score_column=plan.score_column,
            cost_column=plan.cost_column,
            value_column=plan.value_column,
            id_column=plan.id_column,
            objective=plan.objective,
        )

    if plan.method == "topk":
        if plan.capacity is None:
            raise ValidationError("DecisionPlan.capacity is required for topk.")
        selection = select_topk(
            values,
            capacity=int(plan.capacity),
            costs=costs,
            min_score=plan.min_score,
            ids=ids,
        )
    elif plan.method == "knapsack":
        if plan.budget is None:
            raise ValidationError("DecisionPlan.budget is required for knapsack.")
        plan_backend = plan.config.get("backend") or plan.operating_points.get("backend")
        selection = select_knapsack_with_backend(
            values,
            costs,
            budget=float(plan.budget),
            backend=plan_backend,
            solver=plan.knapsack_solver or "dp",
            min_score=plan.min_score,
            ids=ids,
        )
    elif plan.method == "lp_allocate":
        if plan.budget is None:
            raise ValidationError("DecisionPlan.budget is required for lp_allocate.")
        plan_backend = plan.config.get("backend") or plan.operating_points.get("backend")
        selection = select_lp_allocate_with_backend(
            values,
            costs,
            budget=float(plan.budget),
            backend=plan_backend,
            max_fraction=float(plan.lp_max_fraction),
            min_score=plan.min_score,
            ids=ids,
        )
    else:
        raise ValidationError(f"Unknown allocation method: {plan.method!r}")

    return ApplyDecisionsResult(
        method=plan.method,
        partition=part_name,
        n_rows=int(len(values)),
        n_selected=int(selection["n_selected"]),
        selected_ids=tuple(selection["selected_ids"]),
        selected_indices=tuple(selection["selected_indices"]),
        fractions=tuple(selection["fractions"]),
        scores=tuple(float(values[i]) for i in selection["selected_indices"]),
        selected_value=float(selection["selected_value"]),
        selected_cost=float(selection["selected_cost"]),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
