"""Predict with a frozen semi-supervised plan (no refit)."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.semisupervised.features import decode_predictions, matrix_from_frame
from buildml.semisupervised.results import SemiSupervisedPlan, SemiSupervisedPredictResult

PartitionOrAll = PartitionName | Literal["all"]


def predict_semisupervised(
    dataset: Dataset,
    plan: SemiSupervisedPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
    prediction_column: str = "semisupervised_prediction",
) -> tuple[Dataset | None, SemiSupervisedPredictResult]:
    """Score a frozen semi-supervised plan on a partition (no refit)."""
    if plan is None:
        raise ValidationError("No SemiSupervisedPlan. Call fit_semisupervised first.")
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
        raise ValidationError(f"Missing feature columns for prediction: {missing}")
    x = matrix_from_frame(frame, list(plan.columns))
    raw = plan.estimator_.predict(x)
    preds = decode_predictions(raw, plan.label_encoder_)

    disclosures = [
        "Predictions reuse the train-fitted SemiSupervisedPlan (no refit).",
        "Pseudo-labels invented during self-training / graph propagation are not "
        "written back onto Session targets unless attach=True writes a prediction column.",
    ]
    new_dataset: Dataset | None = None
    if attach:
        if partition != "all":
            raise ValidationError(
                "attach=True requires partition='all' so prediction columns align "
                "with the full Session frame."
            )
        full = dataset._ensure_pandas().copy()
        if prediction_column in full.columns:
            raise ValidationError(
                f"prediction_column '{prediction_column}' already exists on the dataset"
            )
        full[prediction_column] = preds
        roles = dict(dataset.roles)
        roles[prediction_column] = ColumnRole.FEATURE
        new_dataset = Dataset.from_transformed(
            dataset,
            full,
            schema=schema_from_dataframe(full),
            roles=roles,
        )
        disclosures.append(
            f"Attached prediction column {prediction_column!r} as a feature role "
            "(does not mutate the target)."
        )

    result = SemiSupervisedPredictResult(
        partition=part_name,
        n_rows=int(len(preds)),
        predictions=tuple(preds),
        method=plan.method,
        attached=attach,
        prediction_column=prediction_column,
        disclosures=tuple(disclosures),
    )
    return new_dataset, result
