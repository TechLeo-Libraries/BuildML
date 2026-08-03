"""Attach a supervised head on SSL representations (labeled train only)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.selfsupervised.features import matrix_from_frame
from buildml.selfsupervised.results import (
    SSLHeadFitResult,
    SSLHeadPlan,
    SelfSupervisedPlan,
)
from buildml.selfsupervised.types import SSLHeadEstimator
from buildml.semisupervised.features import is_unlabeled_mask


_HEADS = {
    "logistic_regression": lambda rs: LogisticRegression(max_iter=500, random_state=rs),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(random_state=rs),
}


def finetune_ssl_head(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    ssl_plan: SelfSupervisedPlan,
    *,
    estimator: SSLHeadEstimator = "logistic_regression",
    random_state: int | None = 0,
    unlabeled_marker: Any = None,
) -> tuple[SSLHeadPlan, SSLHeadFitResult]:
    """Fit a supervised head on SSL embeddings using labeled train rows only.

    Unlabeled train targets (NaN by default) are skipped: not used as invented
    labels. Holdout partitions are not used during head fit.

    Parameters
    ----------
    dataset:
        Session dataset with target column and SSL feature/text/image columns.
    split_plan:
        Split plan defining the train partition.
    ssl_plan:
        Train-fitted :class:`~buildml.selfsupervised.results.SelfSupervisedPlan`.
    estimator:
        Head classifier key (``logistic_regression`` or ``hist_gradient_boosting``).
    random_state:
        Seed forwarded to the sklearn head estimator.
    unlabeled_marker:
        Target value treated as unlabeled (default NaN).

    Returns
    -------
    tuple[SSLHeadPlan, SSLHeadFitResult]
        Fitted head plan and compact fit report.

    Raises
    ------
    ValidationError
        When the SSL plan is missing, labeled train rows are insufficient, or
        required columns are absent.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if ssl_plan is None:
        raise ValidationError("finetune_ssl_head requires a fitted SelfSupervisedPlan.")

    target = dataset.require_target()
    train = frame_for_partition(dataset, split_plan, "train")
    modality = getattr(ssl_plan, "modality", "tabular")
    if modality == "text":
        col = ssl_plan.columns[0]
        if col not in train.columns:
            raise ValidationError(f"Missing text column {col!r}.")
        emb = np.asarray(
            ssl_plan.encoder_.transform(train[col].astype(str).tolist()), dtype=float
        )
    elif modality == "vision":
        col = ssl_plan.columns[0]
        if col not in train.columns:
            raise ValidationError(f"Missing image column {col!r}.")
        emb = np.asarray(ssl_plan.encoder_.transform(train[col].tolist()), dtype=float)
    else:
        missing = [c for c in ssl_plan.columns if c not in train.columns]
        if missing:
            raise ValidationError(f"Missing SSL feature columns: {missing}")
        x_raw = matrix_from_frame(train, list(ssl_plan.columns))
        emb = np.asarray(ssl_plan.encoder_.transform(x_raw), dtype=float)
    unlabeled = is_unlabeled_mask(train[target], unlabeled_marker)
    n_unlabeled = int(unlabeled.sum())
    n_labeled = int((~unlabeled).sum())
    if n_labeled < 2:
        raise ValidationError(
            "finetune_ssl_head needs at least 2 labeled train rows "
            f"(found n_labeled={n_labeled}, n_unlabeled_skipped={n_unlabeled})."
        )

    y_labeled = train.loc[~unlabeled, target]
    encoder = LabelEncoder()
    y_codes = encoder.fit_transform(y_labeled.astype(str))
    if len(encoder.classes_) < 2:
        raise ValidationError(
            "SSL head fine-tune requires at least 2 classes among labeled train rows."
        )
    x_labeled = emb[~unlabeled]

    key = str(estimator).lower().replace("-", "_")
    if key not in _HEADS:
        raise ValidationError(
            f"Unknown SSL head estimator={estimator!r}. Supported: {sorted(_HEADS)}"
        )
    model = _HEADS[key](random_state)
    model.fit(x_labeled, y_codes)

    disclosures = [
        "SSL head fit uses labeled train rows only on frozen SSL representations.",
        f"Skipped n_unlabeled_train={n_unlabeled} rows (missing target / marker).",
        "Holdout partitions are evaluation-only; pretext encoder is not updated.",
        "For vision/audio/speech transfer heads, use Session.attach_backbone_head "
        "after load_pretrained_backbone (optional torch/speech extras).",
    ]
    warnings: list[str] = []
    if n_unlabeled == 0:
        warnings.append(
            "All train rows were labeled; SSL head fine-tune is fully supervised "
            "on representations (pretext still used unlabeled+labeled features)."
        )

    plan = SSLHeadPlan(
        estimator_name=key,
        target_column=target,
        representation_columns=ssl_plan.representation_columns,
        n_labeled_train=n_labeled,
        n_unlabeled_skipped=n_unlabeled,
        classes_=tuple(encoder.classes_),
        estimator_=model,
        task="classification",
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    # Stash encoder on plan via estimator attributes for predict decode
    plan.estimator_._buildml_label_encoder_ = encoder  # type: ignore[attr-defined]
    plan.estimator_._buildml_ssl_columns_ = tuple(ssl_plan.columns)  # type: ignore[attr-defined]

    result = SSLHeadFitResult(
        estimator_name=key,
        n_labeled_train=n_labeled,
        n_unlabeled_skipped=n_unlabeled,
        representation_columns=ssl_plan.representation_columns,
        target_column=target,
        classes=tuple(encoder.classes_),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result
