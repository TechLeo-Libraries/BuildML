"""Anomaly detector backend adapters."""

from buildml.anomaly.adapters.pyod import build_pyod_estimator, pyod_anomaly_scores
from buildml.anomaly.adapters.sklearn import (
    build_sklearn_unsupervised_estimator,
    sklearn_anomaly_scores,
)
from buildml.anomaly.adapters.supervised import build_supervised_estimator, supervised_anomaly_scores
from buildml.anomaly.adapters.torch_ae import (
    TorchAnomalyAutoencoder,
    build_torch_autoencoder,
    torch_ae_anomaly_scores,
)

__all__ = [
    "TorchAnomalyAutoencoder",
    "build_pyod_estimator",
    "build_sklearn_unsupervised_estimator",
    "build_supervised_estimator",
    "build_torch_autoencoder",
    "pyod_anomaly_scores",
    "sklearn_anomaly_scores",
    "supervised_anomaly_scores",
    "torch_ae_anomaly_scores",
]
