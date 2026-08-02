"""Configuration types for the self-supervised Session path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SelfSupervisedMethod = Literal["masked_tabular"]
SSLHeadEstimator = Literal["logistic_regression", "hist_gradient_boosting"]


@dataclass(slots=True)
class SelfSupervisedConfig:
    """User-facing self-supervised knobs (serializable summary)."""

    method: SelfSupervisedMethod = "masked_tabular"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    latent_dim: int = 16
    hidden: tuple[int, ...] = (64,)
    mask_ratio: float = 0.15
    n_mask_views: int = 3
    max_iter: int = 200
    prefer_reduce_components: bool = True
    representation_prefix: str = "ssl_emb"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "latent_dim": self.latent_dim,
            "hidden": list(self.hidden),
            "mask_ratio": self.mask_ratio,
            "n_mask_views": self.n_mask_views,
            "max_iter": self.max_iter,
            "prefer_reduce_components": self.prefer_reduce_components,
            "representation_prefix": self.representation_prefix,
        }
