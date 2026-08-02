"""Typed results for Session-facing causal ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.causal.types import CausalAssumptions


@dataclass(slots=True)
class CausalPlan:
    """Fitted causal plan: assumptions + nuisance models + train estimates.

    Persist via ``buildml.causal_bundle.v1``. Distinct from Session checkpoints
    and from classical / probabilistic plans. Honesty: backdoor ATE via
    T-learner / IPW / AIPW under caller-declared assumptions — not causal
    discovery and not an automatic DoWhy/EconML platform.
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
    mu0_: Any = field(repr=False, default=None)
    mu1_: Any = field(repr=False, default=None)
    propensity_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
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
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
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
    """Simple sensitivity / placebo disclosure (not a full DoWhy suite)."""

    kind: str
    method: str
    original_ate: float
    refute_ate: float
    ate_shift: float
    n_rows: int
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "method": self.method,
            "original_ate": self.original_ate,
            "refute_ate": self.refute_ate,
            "ate_shift": self.ate_shift,
            "n_rows": self.n_rows,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
