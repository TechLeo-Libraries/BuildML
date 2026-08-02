"""Evaluate a frozen DecisionPlan on a holdout partition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.supervised import FitResult, _feature_target_frames
from buildml.optimize.apply import apply_decisions
from buildml.optimize.features import multiclass_realized_cost, require_split
from buildml.optimize.policies import evaluate_threshold_policy
from buildml.optimize.results import DecisionEvalResult, DecisionPlan


def evaluate_decisions(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult | None,
    plan: DecisionPlan,
    *,
    partition: str = "test",
) -> DecisionEvalResult:
    """Evaluate a frozen policy on a partition (default: untouched test).

    For threshold / cost_matrix: reports classification + realized cost metrics.
    For allocation: reports selected value/cost/utilization (and label hit-rate
    when a binary target is available and selections can be aligned).
    """
    if plan is None:
        raise ValidationError("No DecisionPlan. Call fit_decision_policy(...) first.")
    split = require_split(split_plan)
    disclosures = [
        f"Evaluating frozen policy method={plan.method!r} on partition={partition}.",
        f"Policy was fitted on partition={plan.partition_fitted}.",
        "Decision helpers for ML scores/costs/allocations — not a general OR platform.",
    ]
    warnings = list(plan.warnings)
    if partition == plan.partition_fitted:
        warnings.append(
            "Evaluation partition equals the policy-fitting partition; "
            "metrics are in-sample for the decision rule."
        )

    if plan.method == "threshold":
        if fit_result is None:
            raise ValidationError("evaluate_decisions(threshold) requires Session.fit(...).")
        payload = evaluate_threshold_policy(
            dataset, split, fit_result, plan, partition=partition
        )
        return DecisionEvalResult(
            partition=partition,
            method=plan.method,
            n_rows=int(payload["n_rows"]),
            metrics=dict(payload["metrics"]),
            realized_cost=payload["realized_cost"],
            n_selected=int(payload["n_selected"]),
            threshold=plan.threshold,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    if plan.method == "cost_matrix":
        if fit_result is None:
            raise ValidationError(
                "evaluate_decisions(cost_matrix) requires Session.fit(...)."
            )
        applied = apply_decisions(
            dataset, split, fit_result, plan, partition=partition
        )
        x, y, _, _, _ = _feature_target_frames(dataset, split, partition)  # type: ignore[arg-type]
        del x
        labels = plan.action_labels or plan.class_labels
        label_to_idx = {c: i for i, c in enumerate(labels)}
        y_true_idx = pd.Series(y).astype(str).map(label_to_idx)
        y_pred_idx = pd.Series(list(applied.decisions)).astype(str).map(label_to_idx)
        if y_true_idx.isna().any() or y_pred_idx.isna().any():
            raise ValidationError("Labels/actions not covered by DecisionPlan class_labels.")
        matrix = plan.cost_matrix_
        if matrix is None and plan.cost_model is not None and plan.cost_model.matrix:
            matrix = np.asarray(plan.cost_model.matrix, dtype=float)
        if matrix is None:
            raise ValidationError("DecisionPlan missing cost_matrix for evaluation.")
        realized = multiclass_realized_cost(
            y_true_idx.to_numpy(dtype=int),
            y_pred_idx.to_numpy(dtype=int),
            matrix,
        )
        accuracy = float(
            (pd.Series(list(applied.decisions)).astype(str).to_numpy()
             == pd.Series(y).astype(str).to_numpy()).mean()
        )
        metrics = {
            "realized_cost_total": float(realized),
            "realized_cost_mean": float(realized / max(len(y_true_idx), 1)),
            "accuracy": accuracy,
        }
        return DecisionEvalResult(
            partition=partition,
            method=plan.method,
            n_rows=int(applied.n_rows),
            metrics=metrics,
            realized_cost=float(realized),
            n_selected=int(applied.n_selected),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    applied = apply_decisions(
        dataset, split_plan, fit_result, plan, partition=partition
    )
    metrics = {
        "n_selected": float(applied.n_selected),
        "selected_value": float(applied.selected_value or 0.0),
        "selected_cost": float(applied.selected_cost or 0.0),
    }
    if plan.method == "topk" and plan.capacity:
        metrics["capacity_utilization"] = float(applied.n_selected) / float(plan.capacity)
    if plan.budget not in (None, 0) and plan.method in {"knapsack", "lp_allocate"}:
        metrics["budget_utilization"] = float(applied.selected_cost or 0.0) / float(
            plan.budget
        )

    # Optional: if binary target exists and we selected by model scores, report
    # positive-rate among selected rows.
    if fit_result is not None and fit_result.task == "classification":
        try:
            _x, y, _, _, _ = _feature_target_frames(dataset, split, partition)  # type: ignore[arg-type]
            y_arr = pd.Series(y).astype(str).to_numpy()
            if applied.selected_indices:
                sel = y_arr[list(applied.selected_indices)]
                # positive = last class (sklearn convention for binary)
                classes = [str(c) for c in fit_result.estimator.classes_]
                if len(classes) == 2:
                    pos = classes[1]
                    metrics["selected_positive_rate"] = float((sel == pos).mean())
        except Exception:
            pass

    return DecisionEvalResult(
        partition=partition,
        method=plan.method,
        n_rows=int(applied.n_rows),
        metrics=metrics,
        n_selected=int(applied.n_selected),
        selected_value=applied.selected_value,
        selected_cost=applied.selected_cost,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
