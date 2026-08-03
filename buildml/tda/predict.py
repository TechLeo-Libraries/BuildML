"""Predict with the optional supervised head on topological features."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.tda.features import decode_predictions
from buildml.tda.results import TdaPlan, TdaPredictResult
from buildml.tda.transform import transform_tda


def predict_tda(
    dataset: Dataset,
    plan: TdaPlan,
    split_plan: SplitPlan | None,
    *,
    partition: str = "test",
) -> TdaPredictResult:
    """Predict on a partition using the train-fitted TDA supervised head.

    Runs :func:`transform_tda` with the frozen plan, then applies the sklearn head
    fitted on train topological features. Classification predictions are decoded
    through the plan's label encoder when present.

    Parameters
    ----------
    dataset:
        Session dataset.
    plan:
        Train-fitted plan with ``head_estimator_`` and ``task`` set.
    split_plan:
        Split plan for the requested partition.
    partition:
        ``train``, ``validation``, ``test``, or ``all``.

    Returns
    -------
    TdaPredictResult
        Decoded predictions and partition metadata.

    Raises
    ------
    ValidationError
        When ``head='none'`` was used at fit time or ``plan.task`` is missing.
    """
    if plan.head_estimator_ is None or plan.head == "none":
        raise ValidationError(
            "No TDA supervised head. Refit with head!='none' or use transform_tda."
        )
    if plan.task is None:
        raise ValidationError("TdaPlan.task is missing; cannot predict.")

    transformed = transform_tda(dataset, plan, split_plan, partition=partition)
    x = transformed.features
    raw = plan.head_estimator_.predict(x)
    if plan.task == "classification":
        if plan.label_encoder_ is None:
            preds: list[Any] = [int(v) for v in np.asarray(raw).tolist()]
        else:
            preds = decode_predictions(np.asarray(raw), plan.label_encoder_)
    else:
        preds = [float(v) for v in np.asarray(raw, dtype=float).tolist()]

    return TdaPredictResult(
        partition=str(partition),
        n_rows=len(preds),
        task=plan.task,
        predictions=tuple(preds),
        disclosures=(
            "Predictions from train-fitted head on frozen TDA features.",
        ),
    )
