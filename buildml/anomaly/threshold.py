"""Threshold tuning on validation for frozen anomaly plans."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve, roc_curve

from buildml.anomaly.features import matrix_from_frame
from buildml.anomaly.fit import anomaly_scores
from buildml.anomaly.results import AnomalyPlan, AnomalyThresholdTuneResult
from buildml.anomaly.types import ThresholdTuningMetric
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition

TuningPartition = PartitionName


def tune_anomaly_threshold(
    dataset: Dataset,
    plan: AnomalyPlan,
    split_plan: SplitPlan | None,
    *,
    partition: TuningPartition = "validation",
    label_column: str | None = None,
    positive_label: Any | None = None,
    metric: ThresholdTuningMetric = "f1",
    fbeta: float = 2.0,
    allow_test_tuning: bool = False,
) -> AnomalyThresholdTuneResult:
    """Pick a threshold on validation labels only (never test unless explicitly allowed).

Integrates the same leakage discipline as ``Session.tune_threshold`` /
``fit_decision_policy``: tune on train or validation, evaluate on untouched
test for final claims.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
plan:
    Fitted plan object carrying model state and feature contract.
split_plan:
    Train/validation/test split; fit uses train partition only.
partition:
    Holdout partition name or ``all`` for the full frame.
label_column:
    label column (str | None).
positive_label:
    positive label (Any | None).
metric:
    Distance or evaluation metric name.
fbeta:
    fbeta (float).
allow_test_tuning:
    allow test tuning (bool).

Returns
-------
AnomalyThresholdTuneResult
    Serializable result summary (AnomalyThresholdTuneResult) for history recording.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if split_plan is None:
        raise ValidationError(
            "tune_anomaly_threshold requires a SplitPlan. Call session.split(...) first."
        )
    if partition == "test" and not allow_test_tuning:
        raise ValidationError(
            "Refusing to tune anomaly thresholds on the test partition. "
            "Use partition='validation' or partition='train', or pass "
            "allow_test_tuning=True for exploratory analysis only."
        )
    frame = frame_for_partition(dataset, split_plan, partition)
    resolved_label = _resolve_label_column(dataset, frame, label_column)
    if resolved_label is None:
        raise ValidationError(
            "tune_anomaly_threshold requires label_column or a single target role "
            "on the tuning partition."
        )
    pos = plan.positive_label if positive_label is None else positive_label
    y_true = _binary_from_column(frame, resolved_label, positive_label=pos)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValidationError(
            "Threshold tuning requires both classes on the tuning partition."
        )

    x = matrix_from_frame(frame, list(plan.columns))
    scores = anomaly_scores(plan, x)
    threshold, tune_metrics, disclosures = _search_threshold(
        scores,
        y_true,
        metric=metric,
        fbeta=fbeta,
        contamination=plan.contamination,
    )
    disclosures.extend(
        [
            f"Threshold tuned on partition='{partition}' only; test was not used.",
            "Score calibration: higher anomaly_score = more anomalous.",
            "Tuned threshold replaces plan threshold for subsequent score/eval calls "
            "when applied via session.tune_anomaly_threshold(update_plan=True).",
        ]
    )
    if plan.mode != "supervised":
        disclosures.append(
            "Labels were not used to fit the detector; they tune the operating "
            "point only (evaluation/threshold policy)."
        )

    old_threshold = float(plan.threshold_)
    flags = (scores >= threshold).astype(int)
    return AnomalyThresholdTuneResult(
        partition=str(partition),
        metric=metric,
        old_threshold=old_threshold,
        threshold=float(threshold),
        n_rows=int(len(scores)),
        alert_rate=float(flags.mean()),
        label_column=resolved_label,
        positive_rate=float(y_true.mean()),
        tune_metrics=tune_metrics,
        disclosures=tuple(dict.fromkeys(disclosures)),
    )


def apply_threshold_tune(plan: AnomalyPlan, tuned: AnomalyThresholdTuneResult) -> None:
    """Mutate a plan's threshold after validation tuning (in-place).

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
plan:
    Fitted plan object carrying model state and feature contract.
tuned:
    tuned (AnomalyThresholdTuneResult).
    """
    plan.threshold_ = float(tuned.threshold)
    plan.threshold_policy = "validation_tuned"
    extra = list(plan.disclosures) + list(tuned.disclosures)
    plan.disclosures = tuple(dict.fromkeys(extra))


def _search_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    metric: ThresholdTuningMetric,
    fbeta: float,
    contamination: float,
) -> tuple[float, dict[str, float], list[str]]:
    disclosures: list[str] = []
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    metrics: dict[str, float] = {}

    if metric == "precision_at_contamination":
        q = 1.0 - float(contamination)
        thr = float(np.quantile(scores, q))
        disclosures.append(
            f"precision_at_contamination: threshold at train-style quantile "
            f"1-contamination={q:.4f} applied on tuning scores."
        )
        flags = (scores >= thr).astype(int)
        prec = float(flags[y_true == 1].sum()) / max(int(flags.sum()), 1)
        metrics["precision_at_tuned_threshold"] = prec
        metrics["alert_rate"] = float(flags.mean())
        return thr, metrics, disclosures

    if metric == "youden":
        fpr, tpr, roc_thr = roc_curve(y_true, scores)
        youden = tpr - fpr
        idx = int(np.argmax(youden))
        thr = float(roc_thr[idx])
        disclosures.append("youden: maximized TPR - FPR on tuning partition ROC.")
        metrics["youden"] = float(youden[idx])
        return thr, metrics, disclosures

    precision, recall, thr_pr = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns thresholds len = n_points - 1
    best_thr = float(scores.max())
    best_score = -1.0
    for idx, thr in enumerate(thr_pr):
        pred = (scores >= thr).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, pred, zero_division=0))
        else:
            score = float(fbeta_score(y_true, pred, beta=fbeta, zero_division=0))
        if score >= best_score:
            best_score = score
            best_thr = float(thr)
    metrics["best_metric_value"] = best_score
    disclosures.append(
        f"{metric}: grid over PR-curve thresholds on tuning partition "
        f"(best={best_score:.4f}, threshold={best_thr:.6g})."
    )
    return best_thr, metrics, disclosures


def _resolve_label_column(
    dataset: Dataset,
    frame: Any,
    label_column: str | None,
) -> str | None:
    if label_column is not None:
        if label_column not in frame.columns:
            raise ValidationError(
                f"label_column '{label_column}' not found on the tuning partition"
            )
        return label_column
    targets = dataset.role_columns(ColumnRole.TARGET)
    if len(targets) == 1 and targets[0] in frame.columns:
        return targets[0]
    return None


def _binary_from_column(frame: Any, column: str, *, positive_label: Any) -> np.ndarray:
    series = frame[column]
    if series.isna().any():
        raise ValidationError(
            f"label_column '{column}' contains nulls; drop or impute before tuning"
        )
    return (series.to_numpy() == positive_label).astype(int)
