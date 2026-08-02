"""Evaluate a TDA supervised head on holdout topological features."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.tda.features import (
    classification_metrics,
    partition_frame,
    regression_metrics,
)
from buildml.tda.predict import predict_tda
from buildml.tda.results import TdaEvalResult, TdaPlan


def evaluate_tda(
    dataset: Dataset,
    plan: TdaPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "validation",
) -> TdaEvalResult:
    """Score the train-fitted TDA head on a holdout partition.

    Leakage: transform + head are frozen from train; this only scores.
    """
    if plan.head_estimator_ is None or plan.head == "none":
        raise ValidationError(
            "evaluate_tda requires a supervised head. Refit with head!='none'."
        )
    if plan.task is None:
        raise ValidationError("TdaPlan.task is missing; cannot evaluate.")

    target = dataset.require_target()
    frame = partition_frame(dataset, split_plan, partition)
    if target not in frame.columns:
        raise ValidationError(f"Target column {target!r} missing from partition.")
    if frame[target].isna().any():
        raise ValidationError(
            f"Target column {target!r} has nulls on partition={partition!r}."
        )

    pred = predict_tda(dataset, plan, split_plan, partition=partition)
    y_true = frame[target]
    if plan.task == "classification":
        metrics = classification_metrics(list(y_true), list(pred.predictions))
    else:
        metrics = regression_metrics(
            y_true.to_numpy(dtype=float),
            np.asarray(pred.predictions, dtype=float),
        )

    return TdaEvalResult(
        partition=str(partition),
        task=plan.task,
        n_rows=int(len(y_true)),
        metrics=metrics,
        vectorization=plan.vectorization,
        disclosures=(
            "Holdout scored with frozen train TDA transformer + head (no refit).",
            *plan.disclosures[:3],
        ),
        warnings=tuple(plan.warnings),
    )
