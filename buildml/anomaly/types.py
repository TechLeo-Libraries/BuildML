"""Configuration types for the anomaly / fraud Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AnomalyMethod = Literal[
    "isolation_forest",
    "lof",
    "one_class_svm",
    "supervised_hgb",
]
AnomalyMode = Literal["unsupervised", "novelty", "supervised"]
ThresholdPolicy = Literal[
    "contamination",
    "quantile",
    "score_threshold",
    "decision_zero",
]


@dataclass(slots=True)
class AnomalyConfig:
    """User-facing anomaly knobs (serializable summary)."""

    method: AnomalyMethod = "isolation_forest"
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
    # LOF
    n_neighbors: int = 20
    # One-Class SVM
    nu: float = 0.05
    kernel: str = "rbf"
    gamma: str | float = "scale"
    # Novelty / supervised labels
    normal_label_column: str | None = None
    normal_label_value: Any = 0
    positive_label: Any = 1
    prefer_reduce_components: bool = True
    flag_column: str = "is_anomaly"
    score_column: str = "anomaly_score"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
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
            "normal_label_column": self.normal_label_column,
            "normal_label_value": self.normal_label_value,
            "positive_label": self.positive_label,
            "prefer_reduce_components": self.prefer_reduce_components,
            "flag_column": self.flag_column,
            "score_column": self.score_column,
        }
