"""Scoring, cost validation, and leakage gates for decision helpers."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from buildml.core.errors import LeakageError, ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.model.supervised import FitResult, _feature_target_frames

TuningPartition = Literal["train", "validation", "test"]


def assert_tuning_partition(
    partition: str,
    *,
    allow_test_tuning: bool,
) -> None:
    """Refuse silent policy tuning on Session test without explicit opt-in."""
    if partition not in {"train", "validation", "test"}:
        raise ValidationError(
            "Decision-policy tuning partition must be 'train', 'validation', "
            f"or 'test'; got {partition!r}."
        )
    if partition == "test" and not allow_test_tuning:
        raise LeakageError(
            "Tuning a decision policy on the Session test partition requires "
            "allow_test_tuning=True (dangerous opt-in). Prefer "
            "partition='validation' to select the operating point, then "
            "evaluate_decisions(partition='test') once with the frozen policy. "
            "Selecting thresholds/allocations on test biases the final estimate."
        )


def require_split(split_plan: SplitPlan | None) -> SplitPlan:
    if split_plan is None:
        raise ValidationError(
            "A split is required before fitting or evaluating a decision policy."
        )
    return split_plan


def require_fit_result(fit_result: FitResult | None) -> FitResult:
    if fit_result is None:
        raise ValidationError(
            "No fitted estimator. Call Session.fit(...) before model-score "
            "decision helpers, or pass score_column / cost_column for "
            "column-driven allocation."
        )
    return fit_result


def validate_binary_costs(
    fp_cost: float | None,
    fn_cost: float | None,
    tp_benefit: float,
    tn_benefit: float,
) -> tuple[float, float, float, float]:
    if fp_cost is None or fn_cost is None:
        raise ValidationError(
            "Cost-sensitive threshold policy requires both fp_cost and fn_cost "
            "(>= 0). Omit both only when wrapping a pure F1 operating point "
            "via method='threshold' without costs."
        )
    for name, value in (
        ("fp_cost", fp_cost),
        ("fn_cost", fn_cost),
        ("tp_benefit", tp_benefit),
        ("tn_benefit", tn_benefit),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError(f"{name} must be a finite number")
        if not np.isfinite(float(value)):
            raise ValidationError(f"{name} must be a finite number")
        if name in {"fp_cost", "fn_cost"} and float(value) < 0:
            raise ValidationError(f"{name} must be >= 0")
    return float(fp_cost), float(fn_cost), float(tp_benefit), float(tn_benefit)


def parse_cost_matrix(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    class_labels: Sequence[str] | None,
    n_classes: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    matrix = np.asarray(cost_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(
            f"cost_matrix must be square; got shape {tuple(matrix.shape)}."
        )
    if not np.isfinite(matrix).all():
        raise ValidationError("cost_matrix must contain only finite numbers.")
    if n_classes is not None and matrix.shape[0] != int(n_classes):
        raise ValidationError(
            f"cost_matrix shape {matrix.shape[0]} does not match "
            f"n_classes={n_classes}."
        )
    if class_labels is None:
        labels = tuple(str(i) for i in range(matrix.shape[0]))
    else:
        labels = tuple(str(x) for x in class_labels)
        if len(labels) != matrix.shape[0]:
            raise ValidationError(
                "class_labels length must match cost_matrix dimension."
            )
    return matrix, labels


def partition_frame(
    dataset: Dataset,
    split_plan: SplitPlan,
    partition: str,
) -> pd.DataFrame:
    if partition == "all":
        return dataset.frame.copy()
    return frame_for_partition(dataset, split_plan, partition).copy()  # type: ignore[arg-type]


def model_scores(
    dataset: Dataset,
    split_plan: SplitPlan,
    fit_result: FitResult,
    partition: str,
    *,
    score_source: str = "model_proba",
) -> tuple[pd.Index, np.ndarray, np.ndarray | None, Any]:
    """Return (index, scores, proba_or_None, y_true_or_None) for a partition."""
    x, y, _, _, _ = _feature_target_frames(dataset, split_plan, partition)  # type: ignore[arg-type]
    x = x[list(fit_result.feature_columns)]
    estimator = fit_result.estimator
    y_arr = None if y is None else np.asarray(y)
    if score_source == "model_proba":
        if not hasattr(estimator, "predict_proba"):
            raise ValidationError(
                "score_source='model_proba' requires predict_proba on the "
                "fitted estimator."
            )
        proba = np.asarray(estimator.predict_proba(x), dtype=float)
        if proba.ndim == 1:
            scores = proba
        elif proba.shape[1] == 2:
            scores = proba[:, 1]
        else:
            # multiclass: use max class probability as a generic score
            scores = proba.max(axis=1)
        return x.index, scores.astype(float), proba, y_arr
    if score_source == "model_decision_function":
        if not hasattr(estimator, "decision_function"):
            raise ValidationError(
                "score_source='model_decision_function' requires "
                "decision_function on the fitted estimator."
            )
        raw = np.asarray(estimator.decision_function(x), dtype=float)
        if raw.ndim > 1:
            scores = raw.max(axis=1)
        else:
            scores = raw
        return x.index, scores.astype(float), None, y_arr
    raise ValidationError(f"Unknown score_source for model scoring: {score_source!r}")


def column_scores(
    frame: pd.DataFrame,
    *,
    score_column: str | None,
    cost_column: str | None,
    value_column: str | None,
    id_column: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids, scores/values, costs, positions) from candidate columns."""
    if score_column is None and value_column is None:
        raise ValidationError(
            "Column-driven allocation requires score_column or value_column."
        )
    value_col = value_column or score_column
    assert value_col is not None
    if value_col not in frame.columns:
        raise ValidationError(f"value/score column {value_col!r} not in frame.")
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValidationError(f"Column {value_col!r} contains non-finite values.")

    if cost_column is None:
        costs = np.ones(len(frame), dtype=float)
    else:
        if cost_column not in frame.columns:
            raise ValidationError(f"cost_column {cost_column!r} not in frame.")
        costs = pd.to_numeric(frame[cost_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(costs).all():
            raise ValidationError(f"cost_column {cost_column!r} contains non-finite values.")
        if (costs < 0).any():
            raise ValidationError("cost_column values must be >= 0.")

    if id_column is None:
        ids = np.asarray(frame.index)
    else:
        if id_column not in frame.columns:
            raise ValidationError(f"id_column {id_column!r} not in frame.")
        ids = frame[id_column].to_numpy()
    positions = np.arange(len(frame), dtype=int)
    return ids, values, costs, positions


def binary_confusion_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = (
        float(fp_cost) * fp
        + float(fn_cost) * fn
        - float(tp_benefit) * tp
        - float(tn_benefit) * tn
    )
    n = int(len(y_true))
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "expected_cost_total": float(total),
        "expected_cost_mean": float(total / n) if n else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": (
            float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        ),
    }


def multiclass_realized_cost(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    total = 0.0
    for t, p in zip(y_true_idx.tolist(), y_pred_idx.tolist(), strict=True):
        total += float(cost_matrix[int(t), int(p)])
    return float(total)
