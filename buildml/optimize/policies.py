"""Threshold and cost-matrix decision policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.diagnostics import DiagnosticReport, threshold_report
from buildml.model.supervised import FitResult, _feature_target_frames
from buildml.optimize.features import (
    binary_confusion_cost,
    multiclass_realized_cost,
    parse_cost_matrix,
    validate_binary_costs,
)
from buildml.optimize.results import DecisionPlan
from buildml.optimize.types import CostModel


def _effective_fit_result(plan: DecisionPlan, fit_result: FitResult) -> FitResult:
    """Use auxiliary industry estimator when stored on the plan."""
    aux = getattr(plan, "aux_estimator_", None)
    if aux is None:
        return fit_result
    return FitResult(
        estimator=aux,
        task=fit_result.task,
        feature_columns=tuple(fit_result.feature_columns),
        target_column=fit_result.target_column,
        n_train_rows=fit_result.n_train_rows,
        weight_column=fit_result.weight_column,
    )


def fit_threshold_policy(
    dataset: Dataset,
    split_plan: SplitPlan,
    fit_result: FitResult,
    *,
    partition: str,
    allow_test_tuning: bool,
    fp_cost: float | None,
    fn_cost: float | None,
    tp_benefit: float,
    tn_benefit: float,
) -> tuple[DecisionPlan, dict[str, Any], DiagnosticReport]:
    """Select a binary operating point via the classical threshold sweep.

    Wraps :func:`buildml.model.diagnostics.threshold_report` so
    ``tune_threshold`` and ``fit_decision_policy(method='threshold')`` share
    one cost model. The DecisionPlan persists the recommended cutoff.
    """
    if fit_result.task != "classification":
        raise ValidationError("method='threshold' requires a classification fit.")
    cost_mode = fp_cost is not None or fn_cost is not None
    if cost_mode:
        fp_cost, fn_cost, tp_benefit, tn_benefit = validate_binary_costs(
            fp_cost, fn_cost, tp_benefit, tn_benefit
        )
    report = threshold_report(
        dataset,
        split_plan,
        fit_result,
        partition=partition,  # type: ignore[arg-type]
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
    )
    payload = report.payload
    recommended = payload["recommended_threshold"]
    threshold = float(recommended["threshold"])
    basis = str(payload["recommendation_basis"])
    positive = str(payload["positive_class"])
    disclosures = [
        "method='threshold' wraps classical threshold_report / Session.tune_threshold.",
        f"Operating point selected on partition={partition} "
        f"(basis={basis}, threshold={threshold:.4f}).",
        "Prefer validation for selection; confirm once on untouched test.",
        "Not a general OR solver — binary score→decision operating-point helper.",
    ]
    warnings: list[str] = []
    if partition == "test" and allow_test_tuning:
        warnings.append(
            "DANGEROUS OPT-IN: policy tuned on Session test "
            "(allow_test_tuning=True). Final metrics on the same partition "
            "are optimistic."
        )
    cost_model = None
    if cost_mode:
        assert fp_cost is not None and fn_cost is not None
        cost_model = CostModel(
            kind="binary_expected_cost",
            fp_cost=float(fp_cost),
            fn_cost=float(fn_cost),
            tp_benefit=float(tp_benefit),
            tn_benefit=float(tn_benefit),
            formula=(
                "fp_cost*FP + fn_cost*FN - tp_benefit*TP - tn_benefit*TN "
                "(totals over the scored partition)"
            ),
        )
    plan = DecisionPlan(
        method="threshold",
        partition_fitted=partition,
        allow_test_tuning=allow_test_tuning,
        threshold=threshold,
        positive_class=positive,
        recommendation_basis=basis,
        cost_model=cost_model,
        class_labels=tuple(str(c) for c in fit_result.estimator.classes_),
        action_labels=tuple(str(c) for c in fit_result.estimator.classes_),
        n_rows_fitted=int(payload.get("n_rows", 0)),
        expected_cost_at_fit=(
            None
            if not cost_mode
            else float(payload.get("expected_cost_at_recommended") or 0.0)
        ),
        operating_points={
            k: dict(v) if isinstance(v, dict) else v
            for k, v in (payload.get("operating_points") or {}).items()
        },
        feature_columns=tuple(fit_result.feature_columns),
        target_column=fit_result.target_column,
        task=fit_result.task,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config={
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "tp_benefit": tp_benefit,
            "tn_benefit": tn_benefit,
        },
    )
    metrics = {
        "threshold": threshold,
        "precision": float(recommended.get("precision", 0.0)),
        "recall": float(recommended.get("recall", 0.0)),
        "f1": float(recommended.get("f1", 0.0)),
    }
    if cost_mode and plan.expected_cost_at_fit is not None:
        metrics["expected_cost_total"] = float(plan.expected_cost_at_fit)
    return plan, metrics, report


def apply_threshold_policy(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    plan: DecisionPlan,
    *,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Apply a frozen binary threshold to model positive-class probabilities."""
    if plan.threshold is None:
        raise ValidationError("DecisionPlan has no threshold.")
    fit_result = _effective_fit_result(plan, fit_result)
    x, _y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)  # type: ignore[arg-type]
    x = x[list(fit_result.feature_columns)]
    if not hasattr(fit_result.estimator, "predict_proba"):
        raise ValidationError("Threshold apply requires predict_proba.")
    classes = [str(c) for c in fit_result.estimator.classes_]
    if len(classes) != 2:
        raise ValidationError("Threshold policy is binary only.")
    positive = plan.positive_class or classes[1]
    pos_idx = classes.index(str(positive))
    proba = np.asarray(fit_result.estimator.predict_proba(x), dtype=float)[:, pos_idx]
    pred_pos = proba >= float(plan.threshold)
    negative = classes[0] if classes[1] == str(positive) else classes[1]
    labels = np.where(pred_pos, str(positive), str(negative))
    return labels, proba, x.index


def evaluate_threshold_policy(
    dataset: Dataset,
    split_plan: SplitPlan,
    fit_result: FitResult,
    plan: DecisionPlan,
    *,
    partition: str,
) -> dict[str, Any]:
    labels, proba, _index = apply_threshold_policy(
        dataset, split_plan, fit_result, plan, partition=partition
    )
    x, y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)  # type: ignore[arg-type]
    del x
    y_bin = pd.Series(y).astype(str)
    positive = str(plan.positive_class or fit_result.estimator.classes_[1])
    y_true = (y_bin == positive).astype(int).to_numpy()
    y_pred = (pd.Series(labels).astype(str) == positive).astype(int).to_numpy()
    metrics: dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
        "mean_positive_score": float(proba.mean()),
    }
    realized = None
    if plan.cost_model is not None and plan.cost_model.kind == "binary_expected_cost":
        cm = plan.cost_model
        costs = binary_confusion_cost(
            y_true,
            y_pred,
            fp_cost=float(cm.fp_cost or 0.0),
            fn_cost=float(cm.fn_cost or 0.0),
            tp_benefit=float(cm.tp_benefit),
            tn_benefit=float(cm.tn_benefit),
        )
        metrics.update(costs)
        realized = float(costs["expected_cost_total"])
    return {
        "metrics": metrics,
        "realized_cost": realized,
        "n_rows": int(len(y_true)),
        "n_selected": int(y_pred.sum()),
        "threshold": plan.threshold,
        "decisions": tuple(labels.tolist()),
        "scores": tuple(float(s) for s in proba.tolist()),
    }


def fit_cost_matrix_policy(
    dataset: Dataset,
    split_plan: SplitPlan,
    fit_result: FitResult,
    *,
    partition: str,
    allow_test_tuning: bool,
    cost_matrix: Any,
    class_labels: list[str] | None,
) -> tuple[DecisionPlan, dict[str, Any]]:
    """Bayes decision under a multiclass cost matrix C[true, action].

    For each row, choose action a minimizing Σ_y P(y|x) C[y, a] using
    ``predict_proba``. The matrix itself is user-supplied (not estimated from
    test labels). Fit-partition metrics report realized cost under true labels
    for audit only.
    """
    if fit_result.task != "classification":
        raise ValidationError("method='cost_matrix' requires classification.")
    if not hasattr(fit_result.estimator, "predict_proba"):
        raise ValidationError("method='cost_matrix' requires predict_proba.")
    if cost_matrix is None:
        raise ValidationError("method='cost_matrix' requires cost_matrix.")

    est_labels = tuple(str(c) for c in fit_result.estimator.classes_)
    matrix, labels = parse_cost_matrix(
        cost_matrix,
        class_labels=class_labels or list(est_labels),
        n_classes=len(est_labels),
    )
    if labels != est_labels:
        # Allow reordering if the same set
        if set(labels) != set(est_labels):
            raise ValidationError(
                "cost_matrix class_labels must match estimator.classes_."
            )
        order = [labels.index(c) for c in est_labels]
        matrix = matrix[np.ix_(order, order)]
        labels = est_labels

    x, y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)  # type: ignore[arg-type]
    x = x[list(fit_result.feature_columns)]
    proba = np.asarray(fit_result.estimator.predict_proba(x), dtype=float)
    # expected cost per action: proba @ C  → shape (n, n_actions)
    expected = proba @ matrix
    actions_idx = expected.argmin(axis=1)
    actions = np.asarray([labels[i] for i in actions_idx])
    y_true_idx = pd.Series(y).astype(str).map({c: i for i, c in enumerate(labels)})
    if y_true_idx.isna().any():
        raise ValidationError("Holdout labels contain classes absent from cost_matrix.")
    realized = multiclass_realized_cost(
        y_true_idx.to_numpy(dtype=int), actions_idx, matrix
    )
    expected_mean = float(expected.min(axis=1).mean())

    disclosures = [
        "method='cost_matrix' chooses argmin_a Σ_y P(y|x) C[y,a] (Bayes decision).",
        "Cost matrix is user-supplied — not estimated from the evaluation partition.",
        "Multi-class expected-cost helper for ML scores — not a general OR platform.",
    ]
    warnings: list[str] = []
    if partition == "test" and allow_test_tuning:
        warnings.append(
            "DANGEROUS OPT-IN: cost_matrix policy audited on Session test "
            "(allow_test_tuning=True)."
        )

    plan = DecisionPlan(
        method="cost_matrix",
        partition_fitted=partition,
        allow_test_tuning=allow_test_tuning,
        cost_model=CostModel(
            kind="multiclass_matrix",
            matrix=[list(map(float, row)) for row in matrix.tolist()],
            class_labels=labels,
            formula="action = argmin_a Σ_y P(y|x) * C[y, a]",
        ),
        class_labels=labels,
        action_labels=labels,
        n_rows_fitted=int(len(actions)),
        expected_cost_at_fit=float(realized),
        feature_columns=tuple(fit_result.feature_columns),
        target_column=fit_result.target_column,
        task=fit_result.task,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config={"class_labels": list(labels)},
        cost_matrix_=matrix,
    )
    metrics = {
        "realized_cost_total": float(realized),
        "expected_cost_mean_under_proba": expected_mean,
        "accuracy": float((actions == pd.Series(y).astype(str).to_numpy()).mean()),
    }
    return plan, metrics


def apply_cost_matrix_policy(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    plan: DecisionPlan,
    *,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    matrix = plan.cost_matrix_
    if matrix is None:
        if plan.cost_model is None or plan.cost_model.matrix is None:
            raise ValidationError("DecisionPlan missing cost_matrix.")
        matrix = np.asarray(plan.cost_model.matrix, dtype=float)
    labels = plan.action_labels or plan.class_labels
    x, _y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)  # type: ignore[arg-type]
    x = x[list(fit_result.feature_columns)]
    proba = np.asarray(fit_result.estimator.predict_proba(x), dtype=float)
    expected = proba @ matrix
    actions_idx = expected.argmin(axis=1)
    actions = np.asarray([labels[i] for i in actions_idx])
    scores = expected.min(axis=1)
    return actions, scores, x.index
