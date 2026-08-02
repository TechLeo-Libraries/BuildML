"""Evaluate a frozen online learner on holdout partitions (never for updates)."""

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
from buildml.online.adapters.river import RiverOnlineWrapper
from buildml.online.features import chunk_drift_notes, decode_predictions, matrix_from_frame
from buildml.online.results import OnlineEvalResult, OnlinePlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_online(
    dataset: Dataset,
    plan: OnlinePlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    drift_check: bool = True,
) -> OnlineEvalResult:
    """Score the incremental estimator on a holdout partition.

    Holdout rows are never used for ``partial_fit``. Prefer validation/test
    after streaming train chunks. When ``drift_check=True``, River ADWIN /
    Page-Hinkley detectors (industry backend) or native mean-shift notes are
    surfaced without updating the model.
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
        raise ValidationError(f"Missing feature columns for evaluation: {missing}")
    if plan.target_column not in frame.columns:
        raise ValidationError(
            f"Target column {plan.target_column!r} missing from evaluation frame."
        )

    x = matrix_from_frame(frame, list(plan.columns))
    n_rows = int(len(frame))
    cfg = plan.config or {}
    drift_on = bool(cfg.get("drift_disclose", True)) and drift_check
    disclosures = [
        "Online evaluation scores a holdout partition; rows were never used "
        "for partial_fit updates.",
        f"Incremental state: n_seen_rows={plan.n_seen_rows}, "
        f"n_updates={plan.n_updates}, backend={plan.backend}, "
        f"estimator={plan.estimator_name}.",
    ]
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    drift_notes: list[str] = []
    drift_detected = False

    if n_rows < 1:
        warnings.append("Evaluation partition is empty; metrics are empty.")
        return OnlineEvalResult(
            partition=part_name,
            estimator_name=plan.estimator_name,
            task=plan.task,
            n_rows=0,
            n_seen_rows=plan.n_seen_rows,
            n_updates=plan.n_updates,
            metrics=metrics,
            drift_detected=False,
            drift_notes=tuple(),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    y_true = frame[plan.target_column]
    if y_true.isna().any():
        warnings.append(
            "Evaluation partition contains null targets; those rows are dropped "
            "from metrics."
        )
        mask = ~y_true.isna()
        x = x[mask.to_numpy()]
        y_true = y_true.loc[mask]
        n_rows = int(mask.sum())

    if n_rows < 1:
        warnings.append("No labeled evaluation rows after dropping nulls.")
        return OnlineEvalResult(
            partition=part_name,
            estimator_name=plan.estimator_name,
            task=plan.task,
            n_rows=0,
            n_seen_rows=plan.n_seen_rows,
            n_updates=plan.n_updates,
            metrics=metrics,
            drift_detected=False,
            drift_notes=tuple(),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    raw = plan.estimator_.predict(x)
    y_eval = _evaluation_targets(plan, y_true)

    if drift_on:
        drift_detected, drift_notes = _drift_on_evaluate(plan, x, y_eval)

    if plan.task == "classification":
        preds = decode_predictions(raw, plan.label_encoder_)
        y_true_s = y_true.astype(str).to_numpy()
        y_pred_s = np.asarray([str(v) for v in preds])
        metrics["accuracy"] = float(accuracy_score(y_true_s, y_pred_s))
        metrics["f1_macro"] = float(
            f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0)
        )
    else:
        y_num = y_true.to_numpy(dtype=float)
        y_hat = np.asarray(raw, dtype=float)
        metrics["mae"] = float(mean_absolute_error(y_num, y_hat))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_num, y_hat)))
        metrics["r2"] = float(r2_score(y_num, y_hat))

    if drift_notes:
        disclosures.extend(drift_notes)

    return OnlineEvalResult(
        partition=part_name,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=n_rows,
        n_seen_rows=plan.n_seen_rows,
        n_updates=plan.n_updates,
        metrics=metrics,
        drift_detected=drift_detected,
        drift_notes=tuple(drift_notes),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _evaluation_targets(plan: OnlinePlan, y_true: Any) -> np.ndarray:
    if plan.task == "classification":
        from buildml.online.features import encode_classification_targets

        codes, _, _ = encode_classification_targets(
            y_true,
            label_encoder=plan.label_encoder_,
            classes=plan.classes_,
        )
        return codes
    return y_true.to_numpy(dtype=float)


def _drift_on_evaluate(
    plan: OnlinePlan,
    x: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[bool, list[str]]:
    cfg = plan.config or {}
    detector = str(cfg.get("drift_detector", "mean_shift"))
    notes: list[str] = []
    detected = False

    if detector in {"adwin", "page_hinkley"} and isinstance(
        plan.estimator_, RiverOnlineWrapper
    ):
        detected, notes = plan.estimator_.evaluate_drift_stream(x, y_eval)
        return detected, notes

    if detector == "none":
        return False, []

    notes.extend(
        chunk_drift_notes(
            x,
            plan.init_feature_means_,
            columns=plan.columns,
            enabled=True,
        )
    )
    return detected, notes
