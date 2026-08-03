"""Sklearn semi-supervised adapters (core fallback)."""

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
from buildml.semisupervised.types import SklearnSemiSupervisedMethod

_BASE_ESTIMATORS = {
    "logistic_regression": lambda rs: LogisticRegression(max_iter=500, random_state=rs),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(random_state=rs),
}


def build_sklearn_estimator(
    *,
    method: SklearnSemiSupervisedMethod,
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
    """Construct a sklearn estimator ready for fit or scoring.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
kernel:
    kernel (str).
n_neighbors:
    n neighbors (int).
max_iter:
    max iter (int).
alpha:
    alpha (float).
base_estimator:
    base estimator (str).
threshold:
    Decision threshold applied to anomaly scores.
criterion:
    criterion (str).
k_best:
    k best (int).
max_self_train_iter:
    max self train iter (int).

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
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
    raise ValidationError(f"Unsupported sklearn semi-supervised method '{method}'")
