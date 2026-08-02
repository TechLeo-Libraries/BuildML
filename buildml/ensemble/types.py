"""Configuration types for the native ensemble Session path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EnsembleStrategy = Literal["voting", "stacking", "blending"]
VotingMethod = Literal["hard", "soft"]
BlendMethod = Literal["predict", "predict_proba"]


@dataclass(slots=True)
class EnsembleConfig:
    """User-facing ensemble knobs (serializable summary)."""

    strategy: EnsembleStrategy
    estimator_names: tuple[str, ...]
    task: Literal["classification", "regression", "auto"] = "auto"
    # Voting
    voting: VotingMethod = "hard"
    weights: tuple[float, ...] | None = None
    # Stacking
    cv: int = 5
    passthrough: bool = False
    stack_method: str = "auto"
    final_estimator_name: str | None = None
    # Blending (holdout inside train)
    holdout_fraction: float = 0.2
    blend_method: BlendMethod = "predict_proba"
    random_state: int | None = 0
    refit_bases_on_full_train: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "estimator_names": list(self.estimator_names),
            "task": self.task,
            "voting": self.voting,
            "weights": None if self.weights is None else list(self.weights),
            "cv": self.cv,
            "passthrough": self.passthrough,
            "stack_method": self.stack_method,
            "final_estimator_name": self.final_estimator_name,
            "holdout_fraction": self.holdout_fraction,
            "blend_method": self.blend_method,
            "random_state": self.random_state,
            "refit_bases_on_full_train": self.refit_bases_on_full_train,
            "extras": dict(self.extras),
        }
