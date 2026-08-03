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
    """Predict with the incremental estimator without updating the model.

    Scores a holdout or full-dataset partition using the current partial_fit
    state; predictions do not leak back into training.

    Parameters
    ----------
    dataset:
        BuildML dataset with feature columns from the plan.
    plan:
        Fitted :class:`~buildml.online.results.OnlinePlan`.
    split_plan:
        Split plan; required unless ``partition='all'``.
    partition:
        Partition to predict on (``validation``, ``test``, ``train``, or ``all``).

    Returns
    -------
    OnlinePredictResult
        Decoded predictions and disclosure fields.

    Raises
    ------
    ValidationError
        When plan, partition, or column preconditions are invalid.
    """
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
