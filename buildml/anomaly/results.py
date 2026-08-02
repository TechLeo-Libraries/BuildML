"""Typed results for anomaly / fraud detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnomalyPlan:
    """Train-fitted anomaly detector plan (estimator + score/threshold contract).

    Distinct from EDA IsolationForest screens, preprocess ``OutlierPlan`` fences,
    unsupervised ``ClusterPlan``, and Session checkpoints. Persist via
    ``buildml.anomaly_bundle.v1``.

    Score convention
    ----------------
    ``anomaly_score`` is always oriented so that **higher means more anomalous**.
    Flags are ``anomaly_score >= threshold`` unless a method-native predict path
    is recorded with disclosure.
    """

    method: str
    mode: str
    columns: tuple[str, ...]
    n_train_rows: int
    n_fit_rows: int
    threshold_policy: str
    threshold_: float
    contamination: float
    train_alert_rate_: float
    train_score_stats_: dict[str, float]
    flag_column: str
    score_column: str
    estimator_: Any = field(repr=False)
    positive_label: Any = 1
    normal_label_column: str | None = None
    normal_label_value: Any = 0
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mode": self.mode,
            "columns": list(self.columns),
            "n_train_rows": self.n_train_rows,
            "n_fit_rows": self.n_fit_rows,
            "threshold_policy": self.threshold_policy,
            "threshold": self.threshold_,
            "contamination": self.contamination,
            "train_alert_rate": self.train_alert_rate_,
            "train_score_stats": dict(self.train_score_stats_),
            "flag_column": self.flag_column,
            "score_column": self.score_column,
            "positive_label": self.positive_label,
            "normal_label_column": self.normal_label_column,
            "normal_label_value": self.normal_label_value,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class AnomalyFitResult:
    """Outcome of fitting an anomaly detector on the train partition."""

    method: str
    mode: str
    n_train_rows: int
    n_fit_rows: int
    columns: tuple[str, ...]
    threshold_policy: str
    threshold: float
    contamination: float
    train_alert_rate: float
    train_score_stats: dict[str, float]
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mode": self.mode,
            "n_train_rows": self.n_train_rows,
            "n_fit_rows": self.n_fit_rows,
            "columns": list(self.columns),
            "threshold_policy": self.threshold_policy,
            "threshold": self.threshold,
            "contamination": self.contamination,
            "train_alert_rate": self.train_alert_rate,
            "train_score_stats": dict(self.train_score_stats),
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"AnomalyFit · {self.method} · mode={self.mode} · "
            f"n_fit={self.n_fit_rows}/{self.n_train_rows} · "
            f"threshold={self.threshold:.6g} ({self.threshold_policy}) · "
            f"train_alert_rate={self.train_alert_rate:.4f}"
        )
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class AnomalyScoreResult:
    """Scores and flags for one partition under a frozen AnomalyPlan."""

    partition: str
    method: str
    mode: str
    n_rows: int
    n_flagged: int
    alert_rate: float
    threshold: float
    threshold_policy: str
    scores: tuple[float, ...]
    flags: tuple[int, ...]
    score_stats: dict[str, float] = field(default_factory=dict)
    attached: bool = False
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "mode": self.mode,
            "n_rows": self.n_rows,
            "n_flagged": self.n_flagged,
            "alert_rate": self.alert_rate,
            "threshold": self.threshold,
            "threshold_policy": self.threshold_policy,
            "score_stats": dict(self.score_stats),
            "attached": self.attached,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class AnomalyEvalResult:
    """Anomaly evaluation on a partition (thresholded + optional labeled metrics)."""

    partition: str
    method: str
    mode: str
    n_rows: int
    n_flagged: int
    alert_rate: float
    threshold: float
    threshold_policy: str
    metrics: dict[str, float] = field(default_factory=dict)
    labeled_metrics: dict[str, float] = field(default_factory=dict)
    positive_rate: float | None = None
    label_column: str | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "mode": self.mode,
            "n_rows": self.n_rows,
            "n_flagged": self.n_flagged,
            "alert_rate": self.alert_rate,
            "threshold": self.threshold,
            "threshold_policy": self.threshold_policy,
            "metrics": dict(self.metrics),
            "labeled_metrics": dict(self.labeled_metrics),
            "positive_rate": self.positive_rate,
            "label_column": self.label_column,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        print(
            f"AnomalyEval · {self.method} · mode={self.mode} · "
            f"partition={self.partition} · n={self.n_rows} · "
            f"alert_rate={self.alert_rate:.4f} · threshold={self.threshold:.6g}"
        )
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for key, value in self.labeled_metrics.items():
            print(f"  labeled.{key}: {value:.6f}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")
