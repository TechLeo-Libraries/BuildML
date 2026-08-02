"""Predict with a fitted global federated model (no update)."""

from __future__ import annotations

from typing import Literal

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.federated.features import decode_predictions, matrix_from_frame
from buildml.federated.results import FederatedPlan, FederatedPredictResult

PartitionOrAll = PartitionName | Literal["all"]


def predict_federated(
    dataset: Dataset,
    plan: FederatedPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
) -> FederatedPredictResult:
    """Predict with the global federated estimator (no local update / no leakage)."""
    if plan is None:
        raise ValidationError("No FederatedPlan. Call fit_federated first.")

    if partition == "all":
        frame = dataset._ensure_pandas()
        part_name = "all"
    else:
        if split_plan is None:
            raise ValidationError(
                f"partition='{partition}' requires a SplitPlan. "
                "Call session.split(...)."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
        part_name = str(partition)

    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"Missing feature columns for predict_federated: {missing}"
        )

    x = matrix_from_frame(frame, list(plan.columns))
    raw = plan.estimator_.predict(x)
    if plan.task == "classification":
        preds = tuple(decode_predictions(raw, plan.label_encoder_))
    else:
        preds = tuple(float(v) for v in raw)

    return FederatedPredictResult(
        partition=part_name,
        method=plan.method,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=int(len(frame)),
        predictions=preds,
        disclosures=(
            "predict_federated does not update client or global models.",
            f"Predictions from global estimator={plan.estimator_name} after "
            f"{len(plan.round_history)} federated round(s).",
            "Honesty: local FL simulation — not a distributed FL platform.",
        ),
        warnings=(),
    )
