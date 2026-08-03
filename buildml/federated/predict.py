"""Predict with a fitted global federated model (no update)."""

from __future__ import annotations

from typing import Literal

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.federated.catalog import resolve_backend
from buildml.federated.features import decode_predictions, matrix_from_frame
from buildml.federated.results import FederatedPlan, FederatedPredictResult
from buildml.federated.types import FederatedBackend

PartitionOrAll = PartitionName | Literal["all"]


def predict_federated(
    dataset: Dataset,
    plan: FederatedPlan,
    split_plan: SplitPlan | None,
    *,
    backend: FederatedBackend | None = None,
    partition: PartitionOrAll = "test",
) -> FederatedPredictResult:
    """Predict with the global federated estimator without local updates.

    Uses the aggregated global model from ``fit_federated``; no client data is
    used for training and no parameter updates occur during prediction.

    Parameters
    ----------
    dataset:
        BuildML dataset with feature columns matching the fitted plan.
    plan:
        Fitted :class:`~buildml.federated.results.FederatedPlan`.
    split_plan:
        Split plan; required unless ``partition='all'``.
    backend:
        Optional backend override; must match ``plan.backend`` when set.
    partition:
        Partition to predict on (``validation``, ``test``, ``train``, or
        ``all``).

    Returns
    -------
    FederatedPredictResult
        Decoded predictions and honesty disclosures.

    Raises
    ------
    ValidationError
        When plan is missing, backend mismatches, split is required, or
        feature columns are absent.
    """
    if plan is None:
        raise ValidationError("No FederatedPlan. Call fit_federated first.")

    if backend is not None:
        resolved = resolve_backend(backend, method=plan.method)
        plan_backend = str(getattr(plan, "backend", "native") or "native")
        if resolved != plan_backend:
            raise ValidationError(
                f"backend={backend!r} does not match FederatedPlan.backend="
                f"{plan_backend!r}. Refit or omit backend= on predict."
            )

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
            f"Predictions from backend={getattr(plan, 'backend', 'native')} "
            f"global estimator={plan.estimator_name} after "
            f"{len(plan.round_history)} federated round(s).",
            "Honesty: local FL simulation: not a distributed FL platform.",
        ),
        warnings=(),
    )
