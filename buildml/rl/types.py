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

ImitationBackend = Literal["sklearn", "industry"]

ImitationMethod = Literal["bc_mlp", "gail_lite"]

RlMode = Literal["contextual_bandit", "gym_reinforce", "gym_sb3"]

RlBackend = Literal["sklearn", "native", "industry"]

BanditAlgorithm = Literal["linucb", "epsilon_greedy", "softmax"]

Sb3Algorithm = Literal["ppo", "dqn", "a2c"]


@dataclass(slots=True)
class ImitationConfig:
    """User-facing behavioral cloning knobs (serializable summary)."""

    task: ImitationTask = "classification"
    backend: ImitationBackend = "sklearn"
    estimator: ImitationEstimator = "logistic_regression"
    method: ImitationMethod | None = None
    columns: tuple[str, ...] | None = None
    action_column: str | None = None
    env_id: str | None = None
    n_epochs: int = 40
    random_state: int | None = 0
    prefer_reduce_components: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "backend": self.backend,
            "estimator": self.estimator,
            "method": self.method,
            "columns": None if self.columns is None else list(self.columns),
            "action_column": self.action_column,
            "env_id": self.env_id,
            "n_epochs": self.n_epochs,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
        }


@dataclass(slots=True)
class RlConfig:
    """User-facing RL knobs (bandit + optional Gymnasium)."""

    mode: RlMode = "contextual_bandit"
    backend: RlBackend = "sklearn"
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
    total_timesteps: int = 20_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
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
            "total_timesteps": self.total_timesteps,
        }
