"""Train-only semi-supervised fit (label propagation / spreading / self-training)."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier,
)

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.semisupervised.features import (
    encode_targets_for_sklearn,
    matrix_from_frame,
    resolve_semisupervised_columns,
)
from buildml.semisupervised.results import SemiSupervisedFitResult, SemiSupervisedPlan
from buildml.semisupervised.types import SemiSupervisedConfig, SemiSupervisedMethod

_BASE_ESTIMATORS = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=500, random_state=rs
    ),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(
        random_state=rs
    ),
}


def fit_semisupervised(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: SemiSupervisedMethod = "label_propagation",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    kernel: str = "knn",
    n_neighbors: int = 7,
    max_iter: int = 1000,
    alpha: float = 0.2,
    base_estimator: str = "logistic_regression",
    threshold: float = 0.75,
    criterion: str = "threshold",
    k_best: int = 10,
    max_self_train_iter: int = 10,
    unlabeled_marker: Any = None,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
) -> tuple[SemiSupervisedPlan, SemiSupervisedFitResult]:
    """Fit a semi-supervised classifier on the train partition only.

    Label missingness
    -----------------
    Rows with missing targets (NaN by default) are unlabeled. Sklearn's ``-1``
    convention is applied internally. Validation/test partitions are never used
    to invent labels or to select the model during this fit.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    target = dataset.require_target()
    train = frame_for_partition(dataset, split_plan, "train")
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    x = matrix_from_frame(train, cols)
    y_sk, encoder, classes, n_labeled, n_unlabeled = encode_targets_for_sklearn(
        train[target],
        unlabeled_marker=unlabeled_marker,
    )
    n_train = int(x.shape[0])
    warnings: list[str] = []

    disclosures.extend(
        [
            "Semi-supervised fit uses the train partition only. "
            "Unlabeled rows are target missingness (NaN by default) mapped to "
            "sklearn's -1 convention internally.",
            "Validation/test rows are never used to invent labels or to select "
            "the estimator during fit_semisupervised.",
            f"Train label mix: n_labeled={n_labeled}, n_unlabeled={n_unlabeled} "
            f"of n_train={n_train}.",
        ]
    )
    if n_unlabeled == 0:
        warnings.append(
            "No unlabeled train rows detected; semi-supervised methods reduce to "
            "supervised fit on the labeled train set (disclosed)."
        )

    estimator = _build_estimator(
        method=method,
        random_state=random_state,
        kernel=kernel,
        n_neighbors=n_neighbors,
        max_iter=max_iter,
        alpha=alpha,
        base_estimator=base_estimator,
        threshold=threshold,
        criterion=criterion,
        k_best=k_best,
        max_self_train_iter=max_self_train_iter,
    )
    try:
        estimator.fit(x, y_sk)
    except Exception as exc:  # noqa: BLE001 — surface as ValidationError
        raise ValidationError(
            f"Semi-supervised fit failed for method={method!r}: {exc}"
        ) from exc

    config = SemiSupervisedConfig(
        method=method,
        columns=tuple(cols),
        random_state=random_state,
        kernel=kernel,
        n_neighbors=n_neighbors,
        max_iter=max_iter,
        alpha=alpha,
        base_estimator=base_estimator,
        threshold=threshold,
        criterion=criterion,
        k_best=k_best,
        max_self_train_iter=max_self_train_iter,
        unlabeled_marker=unlabeled_marker,
        prefer_reduce_components=prefer_reduce_components,
    )
    plan = SemiSupervisedPlan(
        method=method,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        n_labeled_train=n_labeled,
        n_unlabeled_train=n_unlabeled,
        classes_=classes,
        estimator_=estimator,
        label_encoder_=encoder,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = SemiSupervisedFitResult(
        method=method,
        n_train_rows=n_train,
        n_labeled_train=n_labeled,
        n_unlabeled_train=n_unlabeled,
        columns=tuple(cols),
        target_column=target,
        classes=classes,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _build_estimator(
    *,
    method: SemiSupervisedMethod,
    random_state: int | None,
    kernel: str,
    n_neighbors: int,
    max_iter: int,
    alpha: float,
    base_estimator: str,
    threshold: float,
    criterion: str,
    k_best: int,
    max_self_train_iter: int,
) -> Any:
    if method == "label_propagation":
        return LabelPropagation(
            kernel=kernel,
            n_neighbors=int(n_neighbors),
            max_iter=int(max_iter),
        )
    if method == "label_spreading":
        return LabelSpreading(
            kernel=kernel,
            n_neighbors=int(n_neighbors),
            max_iter=int(max_iter),
            alpha=float(alpha),
        )
    if method == "self_training":
        key = str(base_estimator).lower().replace("-", "_")
        if key not in _BASE_ESTIMATORS:
            raise ValidationError(
                f"Unknown base_estimator={base_estimator!r}. "
                f"Supported: {sorted(_BASE_ESTIMATORS)}"
            )
        base = _BASE_ESTIMATORS[key](random_state)
        return SelfTrainingClassifier(
            clone(base),
            threshold=float(threshold),
            criterion=str(criterion),
            k_best=int(k_best),
            max_iter=int(max_self_train_iter),
        )
    raise ValidationError(f"Unsupported semi-supervised method '{method}'")
