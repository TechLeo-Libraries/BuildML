"""Typed results for Bayesian / probabilistic Session path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProbabilisticPlan:
    """Fitted probabilistic estimator + optional conformal calibration state.

    Persist via ``buildml.probabilistic_bundle.v1``. Distinct from Session
    checkpoints and from classical FitResult. Honesty: sklearn Bayesian /
    GP / Naive Bayes estimators with optional split-conformal intervals —
    not a probabilistic-programming / MCMC platform (PyMC/Stan).
    """

    estimator_name: str
    task: str
    columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    n_fit_rows: int
    n_conformal_calib_rows: int
    alpha: float
    conformal: bool
    interval_method: str
    classes_: tuple[Any, ...] | None
    estimator_: Any = field(repr=False)
    label_encoder_: Any = field(repr=False, default=None)
    conformal_quantile_: float | None = None
    conformal_fit_indices_: tuple[Any, ...] = ()
    conformal_calib_indices_: tuple[Any, ...] = ()
    supports_return_std: bool = False
    supports_predict_proba: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_name": self.estimator_name,
            "task": self.task,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_fit_rows": self.n_fit_rows,
            "n_conformal_calib_rows": self.n_conformal_calib_rows,
            "alpha": self.alpha,
            "conformal": self.conformal,
            "interval_method": self.interval_method,
            "classes": None if self.classes_ is None else list(self.classes_),
            "conformal_quantile": self.conformal_quantile_,
            "supports_return_std": self.supports_return_std,
            "supports_predict_proba": self.supports_predict_proba,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class ProbabilisticFitResult:
    """Outcome of fitting a probabilistic / Bayesian estimator on train."""

    estimator_name: str
    task: str
    n_train_rows: int
    n_fit_rows: int
    n_conformal_calib_rows: int
    columns: tuple[str, ...]
    target_column: str
    alpha: float
    conformal: bool
    interval_method: str
    classes: tuple[Any, ...] | None
    conformal_quantile: float | None = None
    used_reduce_components: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_train_rows": self.n_train_rows,
            "n_fit_rows": self.n_fit_rows,
            "n_conformal_calib_rows": self.n_conformal_calib_rows,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "alpha": self.alpha,
            "conformal": self.conformal,
            "interval_method": self.interval_method,
            "classes": None if self.classes is None else list(self.classes),
            "conformal_quantile": self.conformal_quantile,
            "used_reduce_components": self.used_reduce_components,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ProbabilisticEvalResult:
    """Holdout evaluation with proper scoring + interval coverage."""

    partition: str
    estimator_name: str
    task: str
    n_rows: int
    alpha: float
    metrics: dict[str, float]
    interval_coverage: float | None = None
    mean_interval_width: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "alpha": self.alpha,
            "metrics": dict(self.metrics),
            "interval_coverage": self.interval_coverage,
            "mean_interval_width": self.mean_interval_width,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ProbabilisticPredictResult:
    """Point predictions (and optional proba / std summaries)."""

    partition: str
    estimator_name: str
    task: str
    n_rows: int
    predictions: tuple[Any, ...]
    std: tuple[float, ...] | None = None
    probabilities: tuple[tuple[float, ...], ...] | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "has_std": self.std is not None,
            "has_probabilities": self.probabilities is not None,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ProbabilisticIntervalResult:
    """Predictive intervals / sets for a partition."""

    partition: str
    estimator_name: str
    task: str
    n_rows: int
    alpha: float
    method: str
    lower: tuple[float, ...] | None = None
    upper: tuple[float, ...] | None = None
    point: tuple[Any, ...] = ()
    std: tuple[float, ...] | None = None
    prediction_sets: tuple[tuple[Any, ...], ...] | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "estimator_name": self.estimator_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "alpha": self.alpha,
            "method": self.method,
            "has_lower_upper": self.lower is not None and self.upper is not None,
            "has_std": self.std is not None,
            "has_prediction_sets": self.prediction_sets is not None,
            "n_points": len(self.point),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
