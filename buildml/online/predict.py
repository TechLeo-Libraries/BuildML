"""Predict with a fitted online / continual learner."""

from __future__ import annotations

from typing import Literal

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.online.features import decode_predictions, matrix_from_frame
from buildml.online.results import OnlinePlan, OnlinePredictResult

PartitionOrAll = PartitionName | Literal["all"]


def predict_online(
    dataset: Dataset,
    plan: OnlinePlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
) -> OnlinePredictResult:
    """Predict with the incremental estimator (no update / no leakage into fit)."""
    if plan is None:
        raise ValidationError("No OnlinePlan. Call fit_online first.")

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
        raise ValidationError(f"Missing feature columns for predict_online: {missing}")

    x = matrix_from_frame(frame, list(plan.columns))
    raw = plan.estimator_.predict(x)
    if plan.task == "classification":
        preds = tuple(decode_predictions(raw, plan.label_encoder_))
    else:
        preds = tuple(float(v) for v in raw)

    return OnlinePredictResult(
        partition=part_name,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=int(len(frame)),
        predictions=preds,
        disclosures=(
            "predict_online does not update the online estimator.",
            f"Predictions from estimator={plan.estimator_name} after "
            f"n_updates={plan.n_updates}.",
        ),
        warnings=(),
    )
