"""Predict with a frozen multi-task plan (no refit)."""

from __future__ import annotations

from typing import Literal

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.multitask.features import decode_multitask_predictions, matrix_from_frame
from buildml.multitask.results import MultiTaskPlan, MultiTaskPredictResult

PartitionOrAll = PartitionName | Literal["all"]


def predict_multitask(
    dataset: Dataset,
    plan: MultiTaskPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
    prediction_prefix: str | None = None,
) -> tuple[Dataset | None, MultiTaskPredictResult]:
    """Score a frozen multi-task plan on a partition (no refit)."""
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
        raise ValidationError(f"Missing feature columns for predict_multitask: {missing}")

    x = matrix_from_frame(frame, list(plan.columns))
    raw = plan.estimator_.predict(x)
    prefix = prediction_prefix or str(
        plan.config.get("prediction_prefix", "multitask_pred")
    )
    predictions = decode_multitask_predictions(
        raw,
        plan.target_columns,
        task=plan.task,
        label_encoders=plan.label_encoders_,
    )

    disclosures = [
        "Predictions reuse the train-fitted MultiTaskPlan (no refit).",
        f"Per-task prediction columns use prefix {prefix!r} when attach=True.",
    ]
    warnings: list[str] = []
    new_dataset: Dataset | None = None
    if attach:
        if partition != "all":
            raise ValidationError(
                "attach=True requires partition='all' so prediction columns "
                "align with the full Session frame."
            )
        full = dataset._ensure_pandas().copy()
        roles = dict(dataset.roles)
        for task_col, preds in predictions.items():
            col_name = f"{prefix}_{task_col}"
            if col_name in full.columns:
                raise ValidationError(
                    f"Prediction column {col_name!r} already exists on the dataset."
                )
            full[col_name] = list(preds)
            roles[col_name] = ColumnRole.FEATURE
        new_dataset = Dataset.from_transformed(
            dataset,
            full,
            schema=schema_from_dataframe(full),
            roles=roles,
        )
        disclosures.append(
            f"Attached prediction columns with prefix {prefix!r} as feature "
            "roles (does not mutate targets)."
        )

    result = MultiTaskPredictResult(
        partition=part_name,
        n_rows=int(len(frame)),
        method=plan.method,
        task=plan.task,
        target_columns=plan.target_columns,
        predictions=predictions,
        attached=attach,
        prediction_prefix=prefix,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return new_dataset, result
