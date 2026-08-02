"""PyOD industry anomaly detectors (HBOS, COPOD, ECOD, DeepSVDD)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.anomaly.extras import require_pyod
from buildml.anomaly.types import PyODAnomalyMethod
from buildml.core.errors import ValidationError


def build_pyod_estimator(
    *,
    method: PyODAnomalyMethod,
    contamination: float,
    random_state: int | None,
    n_neighbors: int,
) -> Any:
    require_pyod(feature=f"PyOD method '{method}'")
    if method == "hbos":
        from pyod.models.hbos import HBOS

        return HBOS(
            contamination=float(contamination),
            n_bins=10,
            alpha=0.1,
            tol=0.5,
            random_state=random_state,
        )
    if method == "copod":
        from pyod.models.copod import COPOD

        return COPOD(contamination=float(contamination), random_state=random_state)
    if method == "ecod":
        from pyod.models.ecod import ECOD

        return ECOD(contamination=float(contamination), random_state=random_state)
    if method == "deepsvdd":
        from pyod.models.deepsvdd import DeepSVDD

        return DeepSVDD(
            contamination=float(contamination),
            epochs=20,
            batch_size=64,
            random_state=random_state,
        )
    raise ValidationError(f"Unsupported PyOD anomaly method '{method}'")


def pyod_anomaly_scores(estimator: Any, *, method: str, x: np.ndarray) -> np.ndarray:
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
