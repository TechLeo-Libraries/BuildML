"""Configuration types for imitation learning and reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ImitationTask = Literal["classification", "regression"]

ImitationEstimator = Literal[
    "logistic_regression",
    "hist_gradient_boosting",
    "ridge",
    "hist_gradient_boosting_regressor",
]

RlMode = Literal["contextual_bandit", "gym_reinforce"]

BanditAlgorithm = Literal["linucb", "epsilon_greedy", "softmax"]


@dataclass(slots=True)
class ImitationConfig:
    """User-facing behavioral cloning knobs (serializable summary)."""

    task: ImitationTask = "classification"
    estimator: ImitationEstimator = "logistic_regression"
    columns: tuple[str, ...] | None = None
    action_column: str | None = None
    random_state: int | None = 0
    prefer_reduce_components: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "estimator": self.estimator,
            "columns": None if self.columns is None else list(self.columns),
            "action_column": self.action_column,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
        }


@dataclass(slots=True)
class RlConfig:
    """User-facing RL knobs (bandit + optional Gymnasium)."""

    mode: RlMode = "contextual_bandit"
    algorithm: BanditAlgorithm = "linucb"
    columns: tuple[str, ...] | None = None
    action_column: str | None = None
    reward_column: str | None = None
    alpha: float = 1.0
    epsilon: float = 0.1
    temperature: float = 1.0
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    # Gymnasium REINFORCE-lite
    env_id: str = "CartPole-v1"
    n_episodes: int = 200
    max_steps: int = 500
    learning_rate: float = 0.01
    gamma: float = 0.99
    hidden_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "algorithm": self.algorithm,
            "columns": None if self.columns is None else list(self.columns),
            "action_column": self.action_column,
            "reward_column": self.reward_column,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "env_id": self.env_id,
            "n_episodes": self.n_episodes,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "hidden_seed": self.hidden_seed,
        }
