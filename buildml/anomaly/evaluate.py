"""Evaluate train-fitted anomaly detectors (thresholded + optional labeled metrics)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from buildml.anomaly.results import AnomalyEvalResult, AnomalyPlan
from buildml.anomaly.score import score_anomalies
from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_anomaly(
    dataset: Dataset,
    plan: AnomalyPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    label_column: str | None = None,
    positive_label: Any | None = None,
    k: int | None = None,
    override_threshold: float | None = None,
) -> AnomalyEvalResult:
    """Score a frozen anomaly plan on a partition and summarize alert behavior.

    Always reports threshold, alert rate, and score summary stats. When a binary
    ``label_column`` (or the Session target role) is available, also reports
    precision / recall / F1 / PR-AUC / ROC-AUC and precision@k / recall@k with
    class-imbalance disclosures.

    Labeled metrics are **not** causal fraud proof and do not turn unsupervised
    fit into supervised training (labels are evaluation-only unless the plan
    mode was supervised).
    """
    _, scored = score_anomalies(
        dataset,
        plan,
        split_plan,
        partition=partition,
        attach=False,
        override_threshold=override_threshold,
    )
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

    scores = np.asarray(scored.scores, dtype=float)
    flags = np.asarray(scored.flags, dtype=int)
    metrics: dict[str, float] = {
        "alert_rate": float(scored.alert_rate),
        "n_flagged": float(scored.n_flagged),
        "threshold": float(scored.threshold),
        **{f"score_{k}": v for k, v in scored.score_stats.items()},
    }
    disclosures = list(scored.disclosures)
    disclosures.extend(
        [
            "Alert rate is the fraction of rows with anomaly_score >= threshold on "
            "this partition under a frozen plan.",
            "This path is distinct from EDA IsolationForest screens, preprocess "
            "outlier fences, and unsupervised ClusterPlan labels.",
        ]
    )
    warnings: list[str] = []
    recommendations: list[str] = []
    labeled: dict[str, float] = {}
    positive_rate: float | None = None
    resolved_label: str | None = None

    resolved_label = _resolve_label_column(dataset, frame, label_column)
    if resolved_label is not None:
        pos = plan.positive_label if positive_label is None else positive_label
        y_true = _binary_from_column(frame, resolved_label, positive_label=pos)
        positive_rate = float(y_true.mean())
        metrics["positive_rate"] = positive_rate
        n_unique_labels = int(pd.Series(frame[resolved_label]).nunique(dropna=True))
        if n_unique_labels > 2:
            warnings.append(
                f"label_column '{resolved_label}' has {n_unique_labels} unique values. "
                "If this column was scaled/encoded, prefer a target-role label "
                "(excluded from scale) or an untransformed label column."
            )
        disclosures.append(
            f"Labeled metrics use column '{resolved_label}' with "
            f"positive_label={pos!r} (positive_rate={positive_rate:.4f}). "
            "Under class imbalance, prefer PR-AUC and precision/recall@k over accuracy."
        )
        if plan.mode != "supervised":
            disclosures.append(
                "Labels were not used to fit this unsupervised/novelty plan; "
                "they are evaluation-only."
            )
        if positive_rate <= 0.0 or positive_rate >= 1.0:
            warnings.append(
                "Partition has a single class under the chosen positive_label; "
                "labeled ranking metrics are unavailable."
            )
        else:
            try:
                labeled["average_precision"] = float(
                    average_precision_score(y_true, scores)
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"average_precision unavailable: {exc}")
            try:
                labeled["roc_auc"] = float(roc_auc_score(y_true, scores))
            except Exception as exc:  # pragma: no cover
                warnings.append(f"roc_auc unavailable: {exc}")

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, flags, average="binary", zero_division=0
            )
            labeled["precision"] = float(precision)
            labeled["recall"] = float(recall)
            labeled["f1"] = float(f1)
            labeled["f1_thresholded"] = float(f1_score(y_true, flags, zero_division=0))

            k_eff = int(k) if k is not None else max(int(round(positive_rate * len(y_true))), 1)
            k_eff = max(1, min(k_eff, len(y_true)))
            order = np.argsort(-scores)
            top = order[:k_eff]
            hits = int(y_true[top].sum())
            labeled["precision_at_k"] = float(hits) / float(k_eff)
            labeled["recall_at_k"] = float(hits) / float(max(int(y_true.sum()), 1))
            labeled["k"] = float(k_eff)
            disclosures.append(
                f"precision_at_k / recall_at_k use k={k_eff} "
                f"({'caller-specified' if k is not None else 'default ≈ positive_rate * n'})."
            )

        if positive_rate is not None and positive_rate < 0.05:
            recommendations.append(
                "Severe class imbalance detected; report PR-AUC and alert_rate "
                "beside precision/recall, and avoid accuracy-only summaries."
            )

    if part_name == "train":
        recommendations.append(
            "Train-partition anomaly metrics are optimistic for threshold selection; "
            "prefer validation/test for operational claims."
        )
    else:
        recommendations.append(
            "Report partition name, threshold policy, and alert_rate with every metric."
        )
    if plan.mode == "novelty":
        recommendations.append(
            "Novelty detectors assume the fit subset was normal-only; distribution "
            "shift in 'normal' can inflate holdout alerts."
        )
    if plan.threshold_policy != "validation_tuned":
        recommendations.append(
            "When labels exist on validation, prefer session.tune_anomaly_threshold "
            "(partition='validation') before final test evaluation — same leakage "
            "discipline as Session.tune_threshold."
        )
    if plan.used_reduce_components:
        recommendations.append(
            "Detector was fit in PCA component space; interpret scores via loadings "
            "before naming original-feature root causes (associational only)."
        )
    recommendations.append(
        "ClusterPlan labels (fit_clusters) are complementary structure signals, not "
        "anomaly flags — do not conflate the APIs."
    )

    return AnomalyEvalResult(
        partition=part_name,
        method=plan.method,
        mode=plan.mode,
        n_rows=int(scored.n_rows),
        n_flagged=int(scored.n_flagged),
        alert_rate=float(scored.alert_rate),
        threshold=float(scored.threshold),
        threshold_policy=scored.threshold_policy,
        metrics=metrics,
        labeled_metrics=labeled,
        positive_rate=positive_rate,
        label_column=resolved_label,
        disclosures=tuple(dict.fromkeys(disclosures)),
        warnings=tuple(warnings),
        recommendations=tuple(recommendations),
    )


def _resolve_label_column(
    dataset: Dataset,
    frame: pd.DataFrame,
    label_column: str | None,
) -> str | None:
    if label_column is not None:
        if label_column not in frame.columns:
            raise ValidationError(
                f"label_column '{label_column}' not found on the evaluation partition"
            )
        return label_column
    targets = dataset.role_columns(ColumnRole.TARGET)
    if len(targets) == 1 and targets[0] in frame.columns:
        return targets[0]
    return None


def _binary_from_column(
    frame: pd.DataFrame,
    column: str,
    *,
    positive_label: Any,
) -> np.ndarray:
    series = frame[column]
    if series.isna().any():
        raise ValidationError(
            f"label_column '{column}' contains nulls; drop or impute before labeled metrics"
        )
    return (series.to_numpy() == positive_label).astype(int)
