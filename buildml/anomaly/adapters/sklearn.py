"""Sklearn unsupervised anomaly detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from buildml.anomaly.types import SklearnAnomalyMethod
from buildml.core.errors import ValidationError


def build_sklearn_unsupervised_estimator(
    *,
    method: SklearnAnomalyMethod,
    contamination: float,
    random_state: int | None,
    n_estimators: int,
    max_samples: str | int | float,
    n_neighbors: int,
    nu: float,
    kernel: str,
    gamma: str | float,
) -> Any:
    """Construct a sklearn unsupervised estimator ready for fit or scoring.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.
contamination:
    Expected outlier fraction for sklearn-style detectors.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
n_estimators:
    n estimators (int).
max_samples:
    max samples (str | int | float).
n_neighbors:
    n neighbors (int).
nu:
    nu (float).
kernel:
    kernel (str).
gamma:
    gamma (str | float).

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if method == "isolation_forest":
        return IsolationForest(
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            contamination=float(contamination),
            random_state=random_state,
        )
    if method == "lof":
        if n_neighbors < 2:
            raise ValidationError("lof n_neighbors must be >= 2")
        return LocalOutlierFactor(
            n_neighbors=int(n_neighbors),
            contamination=float(contamination),
            novelty=True,
        )
    if method == "one_class_svm":
        if not 0.0 < float(nu) <= 1.0:
            raise ValidationError("one_class_svm nu must be in (0, 1]")
        return OneClassSVM(kernel=kernel, gamma=gamma, nu=float(nu))
    raise ValidationError(f"Unsupported sklearn anomaly method '{method}'")


def sklearn_anomaly_scores(estimator: Any, *, method: str, x: np.ndarray) -> np.ndarray:
    """Perform sklearn anomaly scores for the Session-facing workflow step.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
estimator:
    Fitted model object used for scoring or prediction.
method:
    Method or strategy identifier for the resolved backend.
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if method in {"isolation_forest", "lof"}:
        return -np.asarray(estimator.score_samples(x), dtype=float)
    if method == "one_class_svm":
        return -np.asarray(estimator.decision_function(x), dtype=float)
    raise ValidationError(f"Unsupported sklearn anomaly method '{method}' for scoring")
