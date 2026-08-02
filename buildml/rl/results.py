"""Typed results for imitation learning and reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ImitationPlan:
    """Train-fitted behavioral cloning policy (state → action).

    Persist via ``buildml.imitation_bundle.v1``. Honesty: supervised cloning
    from demonstration rows — not inverse RL, not DAgger by default, not a
    robotics stack.
    """

    task: str
    estimator: str
    columns: tuple[str, ...]
    action_column: str
    n_train_rows: int
    classes_: tuple[Any, ...] | None
    label_encoder_: Any = field(repr=False, default=None)
    estimator_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    train_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "imitation",
            "mode": "behavioral_cloning",
            "task": self.task,
            "estimator": self.estimator,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "n_train_rows": self.n_train_rows,
            "classes": None if self.classes_ is None else list(self.classes_),
            "train_score": self.train_score,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class ImitationFitResult:
    """Outcome of fitting a behavioral cloning policy on Session train."""

    task: str
    estimator: str
    n_train_rows: int
    columns: tuple[str, ...]
    action_column: str
    classes: tuple[Any, ...] | None = None
    train_score: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "estimator": self.estimator,
            "n_train_rows": self.n_train_rows,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "classes": None if self.classes is None else list(self.classes),
            "train_score": self.train_score,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ImitationEvalResult:
    """Holdout imitation metrics (action match vs demonstrations)."""

    partition: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ImitationPredictResult:
    """Predicted actions for a partition."""

    partition: str
    task: str
    n_rows: int
    actions: tuple[Any, ...]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_actions": len(self.actions),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RlPlan:
    """Train-fitted RL policy (contextual bandit or Gymnasium REINFORCE-lite).

    Persist via ``buildml.rl_bundle.v1``. Honesty: Session-shaped bandit /
    small-env RL — not MuJoCo, not multi-agent sims, not a robotics product.
    """

    mode: str
    algorithm: str
    columns: tuple[str, ...]
    action_column: str | None
    reward_column: str | None
    n_train_rows: int
    n_arms: int
    arms_: tuple[Any, ...]
    label_encoder_: Any = field(repr=False, default=None)
    policy_: Any = field(repr=False, default=None)
    propensity_model_: Any = field(repr=False, default=None)
    env_id: str | None = None
    obs_dim: int | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "rl",
            "mode": self.mode,
            "algorithm": self.algorithm,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "reward_column": self.reward_column,
            "n_train_rows": self.n_train_rows,
            "n_arms": self.n_arms,
            "arms": list(self.arms_),
            "env_id": self.env_id,
            "obs_dim": self.obs_dim,
            "train_metrics": dict(self.train_metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class RlFitResult:
    """Outcome of fitting an RL policy."""

    mode: str
    algorithm: str
    n_train_rows: int
    n_arms: int
    columns: tuple[str, ...]
    action_column: str | None = None
    reward_column: str | None = None
    env_id: str | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "algorithm": self.algorithm,
            "n_train_rows": self.n_train_rows,
            "n_arms": self.n_arms,
            "columns": list(self.columns),
            "action_column": self.action_column,
            "reward_column": self.reward_column,
            "env_id": self.env_id,
            "train_metrics": dict(self.train_metrics),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RlEvalResult:
    """Holdout / episode evaluation for RL policies."""

    partition: str | None
    mode: str
    n_rows: int | None
    metrics: dict[str, float]
    offline: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "mode": self.mode,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "offline": self.offline,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RlActResult:
    """Chosen actions (and optional scores) for contexts / observations."""

    partition: str | None
    mode: str
    n_rows: int
    actions: tuple[Any, ...]
    scores: tuple[tuple[float, ...], ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "mode": self.mode,
            "n_rows": self.n_rows,
            "n_actions": len(self.actions),
            "n_score_rows": len(self.scores),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
