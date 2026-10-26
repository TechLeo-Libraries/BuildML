"""Evaluate an SSL head on labeled holdout rows (frozen pretext + head)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.selfsupervised.features import encode_ssl_frame
from buildml.selfsupervised.results import (
    SelfSupervisedEvalResult,
    SelfSupervisedPlan,
    SSLHeadPlan,
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
    """Score frozen SSL representations and head on labeled holdout rows.

    Encodes the evaluation partition through the train-fitted pretext encoder,
    then predicts with the frozen head. Unlabeled rows are counted but never
    treated as ground truth.

    Parameters
    ----------
    dataset:
        Session dataset containing evaluation features and targets.
    ssl_plan:
        Train-fitted :class:`~buildml.selfsupervised.results.SelfSupervisedPlan`.
    head_plan:
        Supervised :class:`~buildml.selfsupervised.results.SSLHeadPlan`.
    split_plan:
        Split plan defining validation/test partitions; required unless
        ``partition='all'``.
    partition:
        Holdout partition name or ``'all'`` for the full frame.
    unlabeled_marker:
        Target value treated as unlabeled (default NaN).

    Returns
    -------
    SelfSupervisedEvalResult
        Labeled-row metrics, counts, disclosures, and warnings.

    Raises
    ------
    ValidationError
        When plans are missing, the partition is invalid, or required columns
        are absent.
    """
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
    if head_plan.representation_columns != ssl_plan.representation_columns:
        raise ValidationError("SSL head and pretext representation columns do not match.")
    missing = [c for c in ssl_plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing SSL feature columns: {missing}")
    unlabeled = is_unlabeled_mask(frame[target], unlabeled_marker)
    n_rows = int(len(frame))
    n_unlabeled = int(unlabeled.sum())
    n_labeled = n_rows - n_unlabeled

    disclosures = [
        "SSL evaluation scores labeled rows under a frozen pretext + head.",
        "Unlabeled holdout rows are never treated as ground truth.",
        "Current partition names do not establish independence from original pretext/head "
        "training data; verify provenance after bundle loading or dataset replacement.",
        f"Eval mix: n_labeled={n_labeled}, n_unlabeled={n_unlabeled} of n_rows={n_rows}.",
    ]
    warnings: list[str] = []
    if partition in {"all", "train"}:
        warnings.append(
            f"partition='{partition}' can include pretext/head training rows; "
            "treat these metrics as diagnostic, not independent holdout evidence."
        )
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
    raw_target = frame.loc[~unlabeled, target]
    raw_classes = label_encoder is not None and getattr(head_plan.estimator_, "_buildml_raw_class_labels_", False)
    if raw_classes and label_encoder is not None:
        if not raw_target.isin(label_encoder.classes_).all():
            raise ValidationError("Evaluation class labels are outside the SSL head classes.")
        y_true = label_encoder.transform(raw_target.to_numpy())
    else:
        y_true = raw_target.astype(str).to_numpy()
    emb = encode_ssl_frame(frame.loc[~unlabeled], ssl_plan)
    pred_codes = head_plan.estimator_.predict(emb)
    if raw_classes:
        y_pred_s = np.asarray(pred_codes)
    elif label_encoder is not None:
        # Older serialized heads stored string labels; preserve that contract.
        y_pred = label_encoder.inverse_transform(np.asarray(pred_codes, dtype=int))
        y_pred_s = np.asarray([str(v) for v in y_pred])
        warnings.append("Legacy SSL head uses stored string labels; refit to preserve raw class types.")
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
