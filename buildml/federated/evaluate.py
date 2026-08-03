"""Evaluate a global federated model on holdout partitions (never for training)."""

from __future__ import annotations

from typing import Any, Literal

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
from buildml.federated.catalog import resolve_backend
from buildml.federated.features import (
    client_ids_in_frame,
    decode_predictions,
    encode_labels,
    frame_for_client,
    matrix_from_frame,
)
from buildml.federated.results import FederatedEvalResult, FederatedPlan
from buildml.federated.types import FederatedBackend

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_federated(
    dataset: Dataset,
    plan: FederatedPlan,
    split_plan: SplitPlan | None,
    *,
    backend: FederatedBackend | None = None,
    partition: PartitionOrAll = "validation",
    per_client: bool = True,
) -> FederatedEvalResult:
    """Score the global federated model on a holdout partition.

    Holdout rows are never used for local client updates. Optional
    ``per_client=True`` reports metrics sliced by the client column on the
    evaluation partition (still evaluation-only).

    Parameters
    ----------
    dataset:
        BuildML dataset with features, target, and client columns.
    plan:
        Fitted :class:`~buildml.federated.results.FederatedPlan`.
    split_plan:
        Split plan; required unless ``partition='all'``.
    backend:
        Optional backend override; must match ``plan.backend`` when set.
    partition:
        Holdout partition to score (``validation``, ``test``, ``train``, or
        ``all``).
    per_client:
        When ``True``, compute metrics per client id on the evaluation frame.

    Returns
    -------
    FederatedEvalResult
        Aggregate and optional per-client holdout metrics with disclosures.

    Raises
    ------
    ValidationError
        When plan is missing, backend mismatches, split is required, or
        columns are absent from the evaluation frame.
    """
    if plan is None:
        raise ValidationError("No FederatedPlan. Call fit_federated first.")

    if backend is not None:
        resolved = resolve_backend(backend, method=plan.method)
        plan_backend = str(getattr(plan, "backend", "native") or "native")
        if resolved != plan_backend:
            raise ValidationError(
                f"backend={backend!r} does not match FederatedPlan.backend="
                f"{plan_backend!r}. Refit or omit backend= on evaluate."
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
        raise ValidationError(f"Missing feature columns for evaluation: {missing}")
    if plan.target_column not in frame.columns:
        raise ValidationError(
            f"Target column {plan.target_column!r} missing from evaluation frame."
        )

    disclosures = [
        "Federated evaluation scores a holdout partition; rows were never "
        "used for local client updates during fit_federated.",
        f"Global model: backend={getattr(plan, 'backend', 'native')}, "
        f"method={plan.method}, estimator={plan.estimator_name}, "
        f"n_rounds_completed={len(plan.round_history)}, "
        f"n_clients_trained={len(plan.client_ids)}.",
        "Honesty: local FL simulation metrics: not a networked FL benchmark.",
    ]
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    per_client_metrics: dict[str, dict[str, float]] = {}

    n_rows = int(len(frame))
    if n_rows < 1:
        warnings.append("Evaluation partition is empty; metrics are empty.")
        return FederatedEvalResult(
            partition=part_name,
            method=plan.method,
            estimator_name=plan.estimator_name,
            task=plan.task,
            n_rows=0,
            metrics=metrics,
            per_client_metrics=per_client_metrics,
            n_clients_evaluated=0,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    y_true = frame[plan.target_column]
    if y_true.isna().any():
        warnings.append(
            "Evaluation partition contains null targets; those rows are "
            "dropped from metrics."
        )
        mask = ~y_true.isna()
        frame = frame.loc[mask]
        y_true = y_true.loc[mask]
        n_rows = int(len(frame))

    if n_rows < 1:
        warnings.append("No labeled evaluation rows after dropping nulls.")
        return FederatedEvalResult(
            partition=part_name,
            method=plan.method,
            estimator_name=plan.estimator_name,
            task=plan.task,
            n_rows=0,
            metrics=metrics,
            per_client_metrics=per_client_metrics,
            n_clients_evaluated=0,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    x = matrix_from_frame(frame, list(plan.columns))
    pred_raw = plan.estimator_.predict(x)
    metrics = _compute_metrics(
        y_true,
        pred_raw,
        task=plan.task,
        label_encoder=plan.label_encoder_,
    )

    n_clients_evaluated = 0
    if per_client and plan.client_column in frame.columns:
        client_ids = client_ids_in_frame(frame, plan.client_column)
        for cid in client_ids:
            cframe = frame_for_client(frame, plan.client_column, cid)
            if len(cframe) < 1:
                continue
            cx = matrix_from_frame(cframe, list(plan.columns))
            cpred = plan.estimator_.predict(cx)
            per_client_metrics[str(cid)] = _compute_metrics(
                cframe[plan.target_column],
                cpred,
                task=plan.task,
                label_encoder=plan.label_encoder_,
            )
            n_clients_evaluated += 1
        disclosures.append(
            f"Per-client holdout metrics reported for {n_clients_evaluated} "
            f"client id(s) present on partition={part_name!r}."
        )
    elif per_client:
        warnings.append(
            f"Client column {plan.client_column!r} missing from evaluation "
            "frame; skipped per-client metrics."
        )

    return FederatedEvalResult(
        partition=part_name,
        method=plan.method,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=n_rows,
        metrics=metrics,
        per_client_metrics=per_client_metrics,
        n_clients_evaluated=n_clients_evaluated,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _compute_metrics(
    y_true: Any,
    pred_raw: np.ndarray,
    *,
    task: str,
    label_encoder: Any,
) -> dict[str, float]:
    if task == "classification":
        if label_encoder is not None:
            y_enc, _, _ = encode_labels(y_true, label_encoder=label_encoder)
            # pred_raw is already encoded class indices from sklearn.
            y_hat = np.asarray(pred_raw)
            preds_labels = decode_predictions(y_hat, label_encoder)
            _ = preds_labels
            acc = float(accuracy_score(y_enc, y_hat))
            try:
                f1 = float(f1_score(y_enc, y_hat, average="macro", zero_division=0))
            except Exception:  # noqa: BLE001
                f1 = float("nan")
            return {"accuracy": acc, "f1_macro": f1}
        y_hat = np.asarray(pred_raw)
        return {
            "accuracy": float(accuracy_score(y_true, y_hat)),
        }

    y = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(pred_raw, dtype=float)
    return {
        "r2": float(r2_score(y, y_hat)),
        "mae": float(mean_absolute_error(y, y_hat)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_hat))),
    }
