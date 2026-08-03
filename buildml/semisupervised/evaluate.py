"""Evaluate a frozen semi-supervised plan on labeled holdout rows only."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.semisupervised.features import is_unlabeled_mask
from buildml.semisupervised.predict import predict_semisupervised
from buildml.semisupervised.results import SemiSupervisedEvalResult, SemiSupervisedPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_semisupervised(
    dataset: Dataset,
    plan: SemiSupervisedPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    unlabeled_marker: Any = None,
) -> SemiSupervisedEvalResult:
    """Evaluate predictions against ground-truth labels on a partition.

Only rows with non-missing targets contribute to metrics. Unlabeled holdout
rows are counted and disclosed, never scored as invented truths. This does
not refit and does not use unlabeled holdout rows for model selection.

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
unlabeled_marker:
    Sentinel marking unlabeled rows; ``None`` uses NaN/NA.

Returns
-------
SemiSupervisedEvalResult
    Serializable result summary (SemiSupervisedEvalResult) for history recording.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    _, scored = predict_semisupervised(
        dataset,
        plan,
        split_plan,
        partition=partition,
        attach=False,
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

    target = plan.target_column
    if target not in frame.columns:
        raise ValidationError(f"Target column {target!r} missing from evaluation frame.")

    marker = unlabeled_marker
    if marker is None:
        marker = (plan.config or {}).get("unlabeled_marker")

    unlabeled = is_unlabeled_mask(frame[target], marker)
    n_rows = int(len(frame))
    n_unlabeled = int(unlabeled.sum())
    n_labeled = n_rows - n_unlabeled
    disclosures = list(scored.disclosures)
    disclosures.extend(
        [
            "Semi-supervised evaluation scores only labeled rows on the partition.",
            "Unlabeled holdout rows are never treated as ground truth and are not "
            "used to invent labels for model selection.",
            f"Eval mix: n_labeled={n_labeled}, n_unlabeled={n_unlabeled} of n_rows={n_rows}.",
        ]
    )
    warnings: list[str] = []
    metrics: dict[str, float] = {}

    if n_labeled < 1:
        warnings.append(
            "No labeled rows on this partition; metrics are empty. "
            "Provide holdout labels or evaluate a labeled split."
        )
        return SemiSupervisedEvalResult(
            partition=part_name,
            method=plan.method,
            n_rows=n_rows,
            n_labeled_eval=0,
            n_unlabeled_eval=n_unlabeled,
            metrics=metrics,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    y_true = frame.loc[~unlabeled, target]
    # Align predictions to labeled rows
    pred_all = np.asarray(scored.predictions, dtype=object)
    y_pred = pred_all[~unlabeled]

    # Normalize both sides to string for robust comparison across int/str labels
    y_true_s = y_true.astype(str).to_numpy()
    y_pred_s = np.asarray([str(v) for v in y_pred])

    metrics["accuracy"] = float(accuracy_score(y_true_s, y_pred_s))
    metrics["f1_macro"] = float(
        f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
    )
    metrics["f1_weighted"] = float(
        f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_s, y_pred_s, average="macro", zero_division=0
    )
    metrics["precision_macro"] = float(precision)
    metrics["recall_macro"] = float(recall)
    metrics["support_labeled"] = float(n_labeled)
    _ = f1, support

    return SemiSupervisedEvalResult(
        partition=part_name,
        method=plan.method,
        n_rows=n_rows,
        n_labeled_eval=n_labeled,
        n_unlabeled_eval=n_unlabeled,
        metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
