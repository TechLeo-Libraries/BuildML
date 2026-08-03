"""Configuration types for the anomaly / fraud Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SklearnAnomalyMethod = Literal["isolation_forest", "lof", "one_class_svm"]
PyODAnomalyMethod = Literal["hbos", "copod", "ecod", "deepsvdd"]
TorchAnomalyMethod = Literal["autoencoder"]
SupervisedAnomalyMethod = Literal["supervised_hgb", "supervised_xgb", "supervised_lgbm"]

AnomalyMethod = (
    SklearnAnomalyMethod | PyODAnomalyMethod | TorchAnomalyMethod | SupervisedAnomalyMethod
)
AnomalyBackend = Literal["sklearn", "pyod", "torch"]
AnomalyMode = Literal["unsupervised", "novelty", "supervised"]
ThresholdPolicy = Literal[
    "contamination",
    "quantile",
    "score_threshold",
    "decision_zero",
    "validation_tuned",
]
ThresholdTuningMetric = Literal["f1", "fbeta", "precision_at_contamination", "youden"]


@dataclass(slots=True)
class AnomalyConfig:
    """User-facing anomaly knobs (serializable summary)."""

    method: AnomalyMethod = "isolation_forest"
    backend: AnomalyBackend | None = None
    mode: AnomalyMode = "unsupervised"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    contamination: float = 0.05
    threshold_policy: ThresholdPolicy = "contamination"
    score_threshold: float | None = None
    quantile: float | None = None
    # IsolationForest
    n_estimators: int = 100
    max_samples: str | int | float = "auto"
    # LOF / PyOD neighborhood-style
    n_neighbors: int = 20
    # One-Class SVM
    nu: float = 0.05
    kernel: str = "rbf"
    gamma: str | float = "scale"
    # Torch autoencoder
    latent_dim: int = 8
    ae_epochs: int = 40
    ae_batch_size: int = 64
    # Novelty / supervised labels
    normal_label_column: str | None = None
    normal_label_value: Any = 0
    positive_label: Any = 1
    prefer_reduce_components: bool = True
    flag_column: str = "is_anomaly"
    score_column: str = "anomaly_score"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a JSON-friendly dict for history and bundles.

Omits private estimator and encoder fields so bundles and history records stay lightweight while preserving teaching disclosures.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "mode": self.mode,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "contamination": self.contamination,
            "threshold_policy": self.threshold_policy,
            "score_threshold": self.score_threshold,
            "quantile": self.quantile,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "n_neighbors": self.n_neighbors,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "latent_dim": self.latent_dim,
            "ae_epochs": self.ae_epochs,
            "ae_batch_size": self.ae_batch_size,
            "normal_label_column": self.normal_label_column,
            "normal_label_value": self.normal_label_value,
            "positive_label": self.positive_label,
            "prefer_reduce_components": self.prefer_reduce_components,
            "flag_column": self.flag_column,
            "score_column": self.score_column,
        }
