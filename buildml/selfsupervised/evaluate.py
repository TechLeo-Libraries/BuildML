"""Evaluate an SSL head on labeled holdout rows (frozen pretext + head)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.selfsupervised.features import matrix_from_frame
from buildml.selfsupervised.results import (
    SSLHeadPlan,
    SelfSupervisedEvalResult,
    SelfSupervisedPlan,
)
from buildml.semisupervised.features import is_unlabeled_mask

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_ssl(
    dataset: Dataset,
    ssl_plan: SelfSupervisedPlan,
    head_plan: SSLHeadPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    unlabeled_marker: Any = None,
) -> SelfSupervisedEvalResult:
    """Score frozen SSL representations + head on labeled partition rows only."""
    if ssl_plan is None or head_plan is None:
        raise ValidationError(
            "evaluate_ssl requires both a SelfSupervisedPlan and an SSLHeadPlan."
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

    target = head_plan.target_column
    if target not in frame.columns:
        raise ValidationError(f"Target column {target!r} missing from evaluation frame.")
    missing = [c for c in ssl_plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing SSL feature columns: {missing}")

    x = matrix_from_frame(frame, list(ssl_plan.columns))
    emb = np.asarray(ssl_plan.encoder_.transform(x), dtype=float)
    unlabeled = is_unlabeled_mask(frame[target], unlabeled_marker)
    n_rows = int(len(frame))
    n_unlabeled = int(unlabeled.sum())
    n_labeled = n_rows - n_unlabeled

    disclosures = [
        "SSL evaluation scores labeled holdout rows only under a frozen pretext + head.",
        "Unlabeled holdout rows are never treated as ground truth.",
        f"Eval mix: n_labeled={n_labeled}, n_unlabeled={n_unlabeled} of n_rows={n_rows}.",
    ]
    warnings: list[str] = []
    metrics: dict[str, float] = {}

    if n_labeled < 1:
        warnings.append("No labeled rows on this partition; metrics are empty.")
        return SelfSupervisedEvalResult(
            partition=part_name,
            n_rows=n_rows,
            n_labeled_eval=0,
            n_unlabeled_eval=n_unlabeled,
            metrics=metrics,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    label_encoder = getattr(head_plan.estimator_, "_buildml_label_encoder_", None)
    y_true = frame.loc[~unlabeled, target].astype(str).to_numpy()
    pred_codes = head_plan.estimator_.predict(emb[~unlabeled])
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(np.asarray(pred_codes, dtype=int))
        y_pred_s = np.asarray([str(v) for v in y_pred])
    else:
        y_pred_s = np.asarray([str(v) for v in pred_codes])

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred_s))
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred_s, average="macro", zero_division=0)
    )
    metrics["f1_weighted"] = float(
        f1_score(y_true, y_pred_s, average="weighted", zero_division=0)
    )
    metrics["support_labeled"] = float(n_labeled)

    return SelfSupervisedEvalResult(
        partition=part_name,
        n_rows=n_rows,
        n_labeled_eval=n_labeled,
        n_unlabeled_eval=n_unlabeled,
        metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
