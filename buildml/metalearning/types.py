"""Configuration types for Session-facing meta-learning (few-shot / episodic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MetaLearningBackend = Literal["sklearn", "torch", "industry"]

MetaLearningMethod = Literal[
    "prototypical",
    "warm_start",
    "prototypical_torch",
    "maml",
    "reptile",
]

MetaLearningBaseEstimator = Literal[
    "logistic_regression",
    "sgd_classifier",
]


@dataclass(slots=True)
class MetaLearningConfig:
    """User-facing meta-learning knobs (serializable summary)."""

    backend: MetaLearningBackend = "sklearn"
    method: MetaLearningMethod = "prototypical"
    task_column: str | None = None
    columns: tuple[str, ...] | None = None
    n_way: int | None = None
    k_shot: int = 5
    n_query: int = 10
    n_episodes: int = 20
    base_estimator: MetaLearningBaseEstimator = "logistic_regression"
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    task_holdout_fraction: float = 0.25
    meta_epochs: int = 40
    inner_lr: float = 0.05
    inner_steps: int = 5
    meta_lr: float = 1e-3
    embed_dim: int = 32
    hidden_dim: int = 64
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize user-facing meta-learning configuration knobs.

        Suitable for plan ``config`` payloads and teaching overlays.

        Returns
        -------
        dict[str, Any]
            Backend, method, episodic protocol, and training hyperparameters.
        """
        return {
            "backend": self.backend,
            "method": self.method,
            "task_column": self.task_column,
            "columns": None if self.columns is None else list(self.columns),
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "n_query": self.n_query,
            "n_episodes": self.n_episodes,
            "base_estimator": self.base_estimator,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "task_holdout_fraction": self.task_holdout_fraction,
            "meta_epochs": self.meta_epochs,
            "inner_lr": self.inner_lr,
            "inner_steps": self.inner_steps,
            "meta_lr": self.meta_lr,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
        }
