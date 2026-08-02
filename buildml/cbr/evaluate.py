"""Holdout evaluation for case-based reasoning plans."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from buildml.cbr.features import classification_accuracy, regression_metrics
from buildml.cbr.predict import predict_cbr
from buildml.cbr.results import CbrEvalResult, CbrPlan
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_cbr(
    dataset: Dataset,
    plan: CbrPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    k: int | None = None,
) -> CbrEvalResult:
    """Score CBR predictions on a holdout partition (no memory update / no refit)."""
    pred = predict_cbr(
        dataset,
        plan,
        split_plan,
        partition=partition,
        k=k,
        return_traces=True,
    )
    y_true = _targets(dataset, split_plan, partition, plan.target_column)
    if len(y_true) != len(pred.predictions):
        raise ValidationError(
            "Prediction length does not match evaluation target length."
        )

    if plan.task == "classification":
        metrics = {
            "accuracy": classification_accuracy(
                y_true.tolist(), list(pred.predictions)
            )
        }
    else:
        metrics = regression_metrics(
            y_true.to_numpy(dtype=float),
            np.asarray(pred.predictions, dtype=float),
        )

    mean_dist = None
    if pred.traces:
        all_d = [d for t in pred.traces for d in t.distances]
        if all_d:
            mean_dist = float(np.mean(all_d))

    return CbrEvalResult(
        partition=str(partition),
        task=plan.task,
        n_rows=pred.n_rows,
        metrics=metrics,
        mean_neighbor_distance=mean_dist,
        disclosures=(
            "Holdout evaluation only — case memory was not updated from this "
            "partition.",
            *plan.disclosures[:2],
        ),
        warnings=(),
    )


def _targets(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
    target_column: str,
) -> pd.Series:
    from buildml.data.splits import frame_for_partition

    if partition == "all":
        frame = dataset._ensure_pandas()
    else:
        if split_plan is None:
            raise ValidationError(
                "evaluate_cbr requires a SplitPlan unless partition='all'."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
    if target_column not in frame.columns:
        raise ValidationError(
            f"Target column {target_column!r} missing from evaluation frame."
        )
    if frame[target_column].isna().any():
        raise ValidationError(
            "Evaluation partition has null targets; refuse silent drop."
        )
    return frame[target_column]
