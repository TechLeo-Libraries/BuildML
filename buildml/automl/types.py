"""Configuration types for the AutoML Session path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AutoMLMethod = Literal["grid", "randomized", "optuna", "evolutionary"]
AutoMLBackend = Literal["native", "optuna", "flaml", "autogluon"]
AutoMLSelection = Literal["cv", "nested", "validation"]
CandidateKind = Literal["single", "voting", "stacking"]
EnsembleMode = Literal["voting", "stacking", "both"]


@dataclass(slots=True)
class AutoMLBudget:
    """Hard compute caps for AutoML (honest, non-NAS search)."""

    max_trials: int = 20
    max_families: int | None = None
    max_recipe_strategies: int | None = None
    max_ensemble_trials: int = 4
    max_time_seconds: float | None = None
    study_storage: str | None = None
    enable_pruning: bool = True
    multi_objective: bool = False
    secondary_metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_trials": self.max_trials,
            "max_families": self.max_families,
            "max_recipe_strategies": self.max_recipe_strategies,
            "max_ensemble_trials": self.max_ensemble_trials,
            "max_time_seconds": self.max_time_seconds,
            "study_storage": self.study_storage,
            "enable_pruning": self.enable_pruning,
            "multi_objective": self.multi_objective,
            "secondary_metric": self.secondary_metric,
        }


@dataclass(slots=True)
class AutoMLConfig:
    """User-facing AutoML knobs (serializable summary)."""

    backend: AutoMLBackend = "native"
    method: AutoMLMethod = "randomized"
    selection: AutoMLSelection = "cv"
    task: Literal["classification", "regression", "auto"] = "auto"
    n_trials: int = 20
    cv: int = 3
    outer_cv: int = 3
    ranking_metric: str | None = None
    include_recipe_search: bool = True
    include_ensembles: bool = False
    max_ensemble_bases: int = 3
    random_state: int | None = 0
    families: tuple[str, ...] | None = None
    budget: AutoMLBudget = field(default_factory=AutoMLBudget)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "method": self.method,
            "selection": self.selection,
            "task": self.task,
            "n_trials": self.n_trials,
            "cv": self.cv,
            "outer_cv": self.outer_cv,
            "ranking_metric": self.ranking_metric,
            "include_recipe_search": self.include_recipe_search,
            "include_ensembles": self.include_ensembles,
            "max_ensemble_bases": self.max_ensemble_bases,
            "random_state": self.random_state,
            "families": None if self.families is None else list(self.families),
            "budget": self.budget.to_dict(),
            "extras": dict(self.extras),
        }
