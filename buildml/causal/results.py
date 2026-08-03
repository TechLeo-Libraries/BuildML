"""Typed results for Session-facing causal ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.causal.types import CausalAssumptions


@dataclass(slots=True)
class CausalPlan:
    """Fitted causal plan: assumptions + nuisance models + train estimates.

    Persist via ``buildml.causal_bundle.v1``. Distinct from Session checkpoints
    and from classical / probabilistic plans. Honesty: backdoor ATE under
    caller-declared assumptions — native sklearn nuisances, optional DoWhy /
    EconML when ``buildml[causal-industry]`` is installed — not causal
    discovery.
    """

    method: str
    assumptions: CausalAssumptions
    treatment_column: str
    outcome_column: str
    confounder_columns: tuple[str, ...]
    outcome_kind: str
    treatment_levels: tuple[Any, Any]
    n_train_rows: int
    n_treated: int
    n_control: int
    ate: float
    ate_std: float | None
    ate_ci_low: float | None
    ate_ci_high: float | None
    bootstrap_samples: int
    clip_propensity: tuple[float, float]
    outcome_model_name: str
    propensity_model_name: str
    backend: str = "native"
    mu0_: Any = field(repr=False, default=None)
    mu1_: Any = field(repr=False, default=None)
    propensity_: Any = field(repr=False, default=None)
    backend_artifact_: Any = field(repr=False, default=None)
    cate_std: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the causal plan for bundles and history logs.

        Captures assumptions, train ATE, bootstrap metadata, and backend
        configuration without embedding fitted nuisance models or DoWhy/EconML
        artifacts.

        Returns
        -------
        dict[str, Any]
            Plan metadata, column contract, ATE summaries, and disclosures.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "assumptions": self.assumptions.to_dict(),
            "treatment_column": self.treatment_column,
            "outcome_column": self.outcome_column,
            "confounder_columns": list(self.confounder_columns),
            "outcome_kind": self.outcome_kind,
            "treatment_levels": list(self.treatment_levels),
            "n_train_rows": self.n_train_rows,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "ate": self.ate,
            "ate_std": self.ate_std,
            "ate_ci_low": self.ate_ci_low,
            "ate_ci_high": self.ate_ci_high,
            "bootstrap_samples": self.bootstrap_samples,
            "clip_propensity": list(self.clip_propensity),
            "outcome_model_name": self.outcome_model_name,
            "propensity_model_name": self.propensity_model_name,
            "cate_std": self.cate_std,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class CausalFitResult:
    """Outcome of fitting causal nuisance models and train ATE."""

    method: str
    estimand: str
    identification: str
    treatment_column: str
    outcome_column: str
    confounder_columns: tuple[str, ...]
    n_train_rows: int
    n_treated: int
    n_control: int
    ate: float
    ate_std: float | None = None
    ate_ci_low: float | None = None
    ate_ci_high: float | None = None
    bootstrap_samples: int = 0
    outcome_kind: str = "continuous"
    backend: str = "native"
    cate_std: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the causal plan for bundles and history logs.

        Captures assumptions, train ATE, bootstrap metadata, and backend
        configuration without embedding fitted nuisance models or DoWhy/EconML
        artifacts.

        Returns
        -------
        dict[str, Any]
            Plan metadata, column contract, ATE summaries, and disclosures.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "estimand": self.estimand,
            "identification": self.identification,
            "treatment_column": self.treatment_column,
            "outcome_column": self.outcome_column,
            "confounder_columns": list(self.confounder_columns),
            "n_train_rows": self.n_train_rows,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "ate": self.ate,
            "ate_std": self.ate_std,
            "ate_ci_low": self.ate_ci_low,
            "ate_ci_high": self.ate_ci_high,
            "bootstrap_samples": self.bootstrap_samples,
            "outcome_kind": self.outcome_kind,
            "cate_std": self.cate_std,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CausalEstimateResult:
    """Effect estimate on a chosen partition using fitted nuisances."""

    partition: str
    method: str
    estimand: str
    n_rows: int
    n_treated: int
    n_control: int
    ate: float
    ate_std: float | None = None
    ate_ci_low: float | None = None
    ate_ci_high: float | None = None
    bootstrap_samples: int = 0
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the causal plan for bundles and history logs.

        Captures assumptions, train ATE, bootstrap metadata, and backend
        configuration without embedding fitted nuisance models or DoWhy/EconML
        artifacts.

        Returns
        -------
        dict[str, Any]
            Plan metadata, column contract, ATE summaries, and disclosures.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "estimand": self.estimand,
            "n_rows": self.n_rows,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "ate": self.ate,
            "ate_std": self.ate_std,
            "ate_ci_low": self.ate_ci_low,
            "ate_ci_high": self.ate_ci_high,
            "bootstrap_samples": self.bootstrap_samples,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CausalEvalResult:
    """Holdout evaluation of nuisance fit quality + effect estimate."""

    partition: str
    method: str
    estimand: str
    n_rows: int
    ate: float
    ate_std: float | None
    ate_ci_low: float | None
    ate_ci_high: float | None
    metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the causal plan for bundles and history logs.

        Captures assumptions, train ATE, bootstrap metadata, and backend
        configuration without embedding fitted nuisance models or DoWhy/EconML
        artifacts.

        Returns
        -------
        dict[str, Any]
            Plan metadata, column contract, ATE summaries, and disclosures.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "estimand": self.estimand,
            "n_rows": self.n_rows,
            "ate": self.ate,
            "ate_std": self.ate_std,
            "ate_ci_low": self.ate_ci_low,
            "ate_ci_high": self.ate_ci_high,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CausalRefuteResult:
    """Sensitivity / placebo disclosure; DoWhy refuters when backend='dowhy'."""

    kind: str
    method: str
    original_ate: float
    refute_ate: float
    ate_shift: float
    n_rows: int
    backend: str = "native"
    refute_p_value: float | None = None
    refute_details: dict[str, Any] | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the causal plan for bundles and history logs.

        Captures assumptions, train ATE, bootstrap metadata, and backend
        configuration without embedding fitted nuisance models or DoWhy/EconML
        artifacts.

        Returns
        -------
        dict[str, Any]
            Plan metadata, column contract, ATE summaries, and disclosures.
        """
        return {
            "kind": self.kind,
            "method": self.method,
            "backend": self.backend,
            "original_ate": self.original_ate,
            "refute_ate": self.refute_ate,
            "ate_shift": self.ate_shift,
            "n_rows": self.n_rows,
            "refute_p_value": self.refute_p_value,
            "refute_details": dict(self.refute_details or {}),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
