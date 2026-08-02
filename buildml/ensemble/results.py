"""Typed results for native ensemble fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class EnsemblePlan:
    """Train-fitted ensemble plan (strategy metadata + fitted estimator).

    The fitted sklearn-compatible estimator also lives on Session ``FitResult``
    so classical ``evaluate`` / ``predict`` / ``save_pipeline`` keep working.
    Persist the plan + disclosures via ``buildml.ensemble_bundle.v1``.
    """

    strategy: Literal["voting", "stacking", "blending"]
    task: Literal["classification", "regression"]
    estimator_names: tuple[str, ...]
    feature_columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    estimator_: Any = field(repr=False)
    final_estimator_name: str | None = None
    voting: str | None = None
    cv: int | None = None
    passthrough: bool = False
    holdout_fraction: float | None = None
    blend_method: str | None = None
    refit_bases_on_full_train: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task": self.task,
            "estimator_names": list(self.estimator_names),
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "estimator": type(self.estimator_).__name__,
            "final_estimator_name": self.final_estimator_name,
            "voting": self.voting,
            "cv": self.cv,
            "passthrough": self.passthrough,
            "holdout_fraction": self.holdout_fraction,
            "blend_method": self.blend_method,
            "refit_bases_on_full_train": self.refit_bases_on_full_train,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class EnsembleFitResult:
    """Outcome of fitting a native ensemble on the train partition."""

    strategy: Literal["voting", "stacking", "blending"]
    task: Literal["classification", "regression"]
    estimator_names: tuple[str, ...]
    n_train_rows: int
    feature_columns: tuple[str, ...]
    target_column: str
    final_estimator_name: str | None = None
    voting: str | None = None
    cv: int | None = None
    holdout_fraction: float | None = None
    blend_method: str | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task": self.task,
            "estimator_names": list(self.estimator_names),
            "n_train_rows": self.n_train_rows,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "final_estimator_name": self.final_estimator_name,
            "voting": self.voting,
            "cv": self.cv,
            "holdout_fraction": self.holdout_fraction,
            "blend_method": self.blend_method,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a high-signal ensemble fit digest."""
        print(
            f"Ensemble · {self.strategy} · {self.task} · "
            f"bases={list(self.estimator_names)} · n_train={self.n_train_rows}"
        )
        for tip in self.disclosures[:8]:
            print(f"  - {tip}")
