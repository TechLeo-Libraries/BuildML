"""PyOD industry anomaly detectors (HBOS, COPOD, ECOD, DeepSVDD)."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from buildml.anomaly.extras import require_pyod
from buildml.anomaly.types import PyODAnomalyMethod
from buildml.core.errors import ValidationError


def _filter_ctor_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs unsupported by the installed PyOD constructor (API drift)."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def build_pyod_estimator(
    *,
    method: PyODAnomalyMethod,
    contamination: float,
    random_state: int | None,
    n_neighbors: int,
    n_features: int | None = None,
) -> Any:
    """Construct a pyod estimator ready for fit or scoring.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.
contamination:
    Expected outlier fraction for sklearn-style detectors.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
n_neighbors:
    n neighbors (int).
n_features:
    n features (int | None).

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    require_pyod(feature=f"PyOD method '{method}'")
    _ = n_neighbors  # reserved for neighborhood-style PyOD models
    if method == "hbos":
        from pyod.models.hbos import HBOS

        # PyOD 3.x HBOS dropped random_state; pass only supported knobs.
        kwargs = _filter_ctor_kwargs(
            HBOS,
            {
                "contamination": float(contamination),
                "n_bins": 10,
                "alpha": 0.1,
                "tol": 0.5,
                "random_state": random_state,
            },
        )
        return HBOS(**kwargs)
    if method == "copod":
        from pyod.models.copod import COPOD

        kwargs = _filter_ctor_kwargs(
            COPOD,
            {
                "contamination": float(contamination),
                "random_state": random_state,
            },
        )
        return COPOD(**kwargs)
    if method == "ecod":
        from pyod.models.ecod import ECOD

        kwargs = _filter_ctor_kwargs(
            ECOD,
            {
                "contamination": float(contamination),
                "random_state": random_state,
            },
        )
        return ECOD(**kwargs)
    if method == "deepsvdd":
        # PyOD 3.x renamed deepsvdd → deep_svdd and requires n_features.
        try:
            from pyod.models.deep_svdd import DeepSVDD
        except ImportError:  # pragma: no cover - PyOD < 3
            from pyod.models.deepsvdd import DeepSVDD  # type: ignore[no-redef]

        if n_features is None:
            raise ValidationError(
                "PyOD DeepSVDD requires n_features (pass fit matrix width)."
            )
        kwargs = _filter_ctor_kwargs(
            DeepSVDD,
            {
                "n_features": int(n_features),
                "contamination": float(contamination),
                "epochs": 20,
                "batch_size": 64,
                "random_state": random_state,
            },
        )
        return DeepSVDD(**kwargs)
    raise ValidationError(f"Unsupported PyOD anomaly method '{method}'")


def pyod_anomaly_scores(estimator: Any, *, method: str, x: np.ndarray) -> np.ndarray:
    """Perform pyod anomaly scores for the Session-facing workflow step.

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
    require_pyod(feature="PyOD anomaly scoring")
    # PyOD: higher decision_function => more anomalous (aligned with BuildML contract).
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(x), dtype=float)
    if hasattr(estimator, "decision_scores_") and x.shape[0] == getattr(
        estimator, "decision_scores_", np.array([])
    ).shape[0]:
        return np.asarray(estimator.decision_scores_, dtype=float)
    raise ValidationError(
        f"PyOD estimator for method='{method}' lacks decision_function scoring."
    )
