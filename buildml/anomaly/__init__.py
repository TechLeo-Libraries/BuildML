"""Anomaly / fraud detection domain (train-fit / holdout-score Session path).

Phase coverage (internal tracker: depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete** with this package):
  1. Unsupervised learning: done (see ``buildml.unsupervised``).
  2. Ensemble learning: done (see ``buildml.ensemble``).
  3. AutoML: done (see ``buildml.automl``).
  4. Time-series forecasting: done (see ``buildml.forecasting``).
  5. Anomaly / fraud detection: dedicated path beyond EDA IsolationForest.
     **This module.**

Industry depth (R5.2):
  - Core sklearn: IsolationForest, LOF, One-Class SVM.
  - PyOD (``buildml[anomaly-industry]``): HBOS, COPOD, ECOD, DeepSVDD.
  - Torch autoencoder reconstruction error (``buildml[torch]``).
  - Supervised fraud scorers: HGB (core), XGBoost/LightGBM (industry extra).

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. Industry PyOD and
GBDT fraud scorers use ``buildml[anomaly-industry]``. Torch AE uses
``buildml[torch]``.

Lazy imports: core never grows heavy anomaly stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "AnomalyBackend",
    "AnomalyConfig",
    "AnomalyEvalResult",
    "AnomalyFitResult",
    "AnomalyMethod",
    "AnomalyMode",
    "AnomalyPlan",
    "AnomalyScoreResult",
    "AnomalyThresholdTuneResult",
    "ThresholdPolicy",
    "ThresholdTuningMetric",
    "anomaly_capability_matrix",
    "anomaly_status",
    "anomaly_status_for_session",
    "evaluate_anomaly",
    "fit_detector",
    "list_anomaly_methods",
    "load_anomaly_bundle",
    "save_anomaly_bundle",
    "score_anomalies",
    "tune_anomaly_threshold",
]


def __getattr__(name: str) -> Any:
    if name in {
        "AnomalyMethod",
        "AnomalyMode",
        "AnomalyBackend",
        "ThresholdPolicy",
        "ThresholdTuningMetric",
        "AnomalyConfig",
    }:
        from buildml.anomaly import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "AnomalyPlan",
        "AnomalyFitResult",
        "AnomalyScoreResult",
        "AnomalyEvalResult",
        "AnomalyThresholdTuneResult",
    }:
        from buildml.anomaly import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_detector":
        from buildml.anomaly.fit import fit_detector

        return fit_detector
    if name == "score_anomalies":
        from buildml.anomaly.score import score_anomalies

        return score_anomalies
    if name == "evaluate_anomaly":
        from buildml.anomaly.evaluate import evaluate_anomaly

        return evaluate_anomaly
    if name == "tune_anomaly_threshold":
        from buildml.anomaly.threshold import tune_anomaly_threshold

        return tune_anomaly_threshold
    if name in {"anomaly_capability_matrix", "list_anomaly_methods"}:
        from buildml.anomaly import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_anomaly_bundle",
        "load_anomaly_bundle",
    }:
        from buildml.anomaly import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"anomaly_status", "anomaly_status_for_session"}:
        from buildml.anomaly import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.anomaly' has no attribute {name!r}")
