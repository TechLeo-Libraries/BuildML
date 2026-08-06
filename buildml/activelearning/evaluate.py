"""Evaluate a frozen active learner on labeled holdout rows only."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from buildml.activelearning.features import decode_predictions, matrix_from_frame
from buildml.activelearning.results import ActiveLearningEvalResult, ActiveLearningPlan
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.semisupervised.features import is_unlabeled_mask

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_active_learning(
    dataset: Dataset,
    plan: ActiveLearningPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    unlabeled_marker: Any = None,
) -> ActiveLearningEvalResult:
    """Score the active learner against ground-truth labels on a partition.

    Only labeled rows contribute to metrics. Unlabeled holdout rows are counted
    and disclosed. This does not query the pool and does not refit.

    Parameters
    ----------
    dataset:
        BuildML dataset containing evaluation features and target.
    plan:
        Fitted :class:`~buildml.activelearning.results.ActiveLearningPlan`.
    split_plan:
        Split plan required when ``partition`` is not ``all``.
    partition:
        Holdout partition name or ``all`` for the full frame.
    unlabeled_marker:
        Optional unlabeled sentinel; defaults to ``plan.config`` value.

    Returns
    -------
    ActiveLearningEvalResult
        Metrics on labeled rows plus labeled/unlabeled counts and disclosures.

    Raises
    ------
    ValidationError
        When no plan exists, columns are missing, or partition needs a split plan.
    """
    if plan is None:
        raise ValidationError("No ActiveLearningPlan. Call fit_active_learner first.")

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

    x = matrix_from_frame(frame, list(plan.columns))
    raw = plan.estimator_.predict(x)
    preds = decode_predictions(raw, plan.label_encoder_)

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
    disclosures = [
        "Active-learning evaluation scores only labeled rows on the partition.",
        "Unlabeled holdout rows are never treated as ground truth.",
        "This evaluation does not query the unlabeled pool.",
        f"Eval mix: n_labeled={n_labeled}, n_unlabeled={n_unlabeled} of n_rows={n_rows}.",
        f"Queries used so far: {plan.n_queries_used} (budget={plan.label_budget}).",
    ]
    warnings: list[str] = []
    metrics: dict[str, float] = {}

    if n_labeled < 1:
        warnings.append(
            "No labeled rows on this partition; metrics are empty. "
            "Provide holdout labels or evaluate a labeled split."
        )
        return ActiveLearningEvalResult(
            partition=part_name,
            strategy=plan.strategy,
            n_rows=n_rows,
            n_labeled_eval=0,
            n_unlabeled_eval=n_unlabeled,
            n_queries_used=plan.n_queries_used,
            metrics=metrics,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    y_true = frame.loc[~unlabeled, target]
    pred_all = np.asarray(preds, dtype=object)
    y_pred = pred_all[~unlabeled]
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

    return ActiveLearningEvalResult(
        partition=part_name,
        strategy=plan.strategy,
        n_rows=n_rows,
        n_labeled_eval=n_labeled,
        n_unlabeled_eval=n_unlabeled,
        n_queries_used=plan.n_queries_used,
        metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
