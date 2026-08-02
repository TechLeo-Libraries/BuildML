"""Evaluate a frozen multi-task plan on holdout partitions (never for fit)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.multitask.features import decode_multitask_predictions, matrix_from_frame
from buildml.multitask.results import MultiTaskEvalResult, MultiTaskPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_multitask(
    dataset: Dataset,
    plan: MultiTaskPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
) -> MultiTaskEvalResult:
    """Score per-task and aggregate metrics on a holdout partition.

    Holdout rows are never used for fitting. Aggregates are unweighted means
    across tasks.
    """
    if plan is None:
        raise ValidationError("No MultiTaskPlan. Call fit_multitask first.")

    if partition == "all":
        frame = dataset._ensure_pandas()
        part_name = "all"
    else:
        if split_plan is None:
            raise ValidationError(
                f"partition='{partition}' requires a SplitPlan. Call session.split(...)."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
        part_name = str(partition)

    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing feature columns for evaluation: {missing}")
    for col in plan.target_columns:
        if col not in frame.columns:
            raise ValidationError(
                f"Target column {col!r} missing from evaluation frame."
            )

    disclosures = [
        "Multi-task evaluation scores a holdout partition; rows were never "
        "used for fitting.",
        "Per-task metrics are reported individually; aggregates are unweighted "
        "means across tasks.",
        f"method={plan.method}, task={plan.task}, "
        f"n_tasks={len(plan.target_columns)}.",
    ]
    warnings: list[str] = []
    n_rows = int(len(frame))
    per_task: dict[str, dict[str, float]] = {}
    metrics: dict[str, float] = {}

    if n_rows < 1:
        warnings.append("Evaluation partition is empty; metrics are empty.")
        return MultiTaskEvalResult(
            partition=part_name,
            method=plan.method,
            task=plan.task,
            n_rows=0,
            metrics=metrics,
            per_task_metrics=per_task,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    # Drop rows with any null target among the multi-task targets.
    target_block = frame[list(plan.target_columns)]
    if target_block.isna().any().any():
        warnings.append(
            "Evaluation partition contains null targets; those rows are dropped "
            "from metrics."
        )
        mask = ~target_block.isna().any(axis=1)
        frame = frame.loc[mask]
        n_rows = int(len(frame))

    if n_rows < 1:
        warnings.append("No labeled evaluation rows after dropping nulls.")
        return MultiTaskEvalResult(
            partition=part_name,
            method=plan.method,
            task=plan.task,
            n_rows=0,
            metrics=metrics,
            per_task_metrics=per_task,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    x = matrix_from_frame(frame, list(plan.columns))
    raw = plan.estimator_.predict(x)
    preds = decode_multitask_predictions(
        raw,
        plan.target_columns,
        task=plan.task,
        label_encoders=plan.label_encoders_,
    )

    if plan.task == "classification":
        accs: list[float] = []
        f1s: list[float] = []
        f1ws: list[float] = []
        for col in plan.target_columns:
            y_true = frame[col].astype(str).to_numpy()
            y_pred = np.asarray([str(v) for v in preds[col]])
            task_metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "f1_weighted": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
            }
            per_task[col] = task_metrics
            accs.append(task_metrics["accuracy"])
            f1s.append(task_metrics["f1_macro"])
            f1ws.append(task_metrics["f1_weighted"])
        metrics = {
            "mean_accuracy": float(np.mean(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "mean_f1_weighted": float(np.mean(f1ws)),
        }
    else:
        maes: list[float] = []
        rmses: list[float] = []
        r2s: list[float] = []
        for col in plan.target_columns:
            y_true = frame[col].to_numpy(dtype=float)
            y_hat = np.asarray(preds[col], dtype=float)
            task_metrics = {
                "mae": float(mean_absolute_error(y_true, y_hat)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_hat))),
                "r2": float(r2_score(y_true, y_hat)),
            }
            per_task[col] = task_metrics
            maes.append(task_metrics["mae"])
            rmses.append(task_metrics["rmse"])
            r2s.append(task_metrics["r2"])
        metrics = {
            "mean_mae": float(np.mean(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "mean_r2": float(np.mean(r2s)),
        }

    return MultiTaskEvalResult(
        partition=part_name,
        method=plan.method,
        task=plan.task,
        n_rows=n_rows,
        metrics=metrics,
        per_task_metrics=per_task,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
