"""Configuration types for the self-supervised Session path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SelfSupervisedMethod = Literal[
    "simclr_tabular",
    "byol_tabular",
    "vicreg_tabular",
    "mae_tabular",
    "vae_tabular",
    "masked_tabular",
    "hf_text_ssl",
    "vision_ssl",
]

SSLHeadEstimator = Literal["logistic_regression", "hist_gradient_boosting"]

Modality = Literal["tabular", "text", "vision"]


@dataclass(slots=True)
class SelfSupervisedConfig:
    """User-facing self-supervised knobs (serializable summary)."""

    method: SelfSupervisedMethod = "simclr_tabular"
    modality: Modality = "tabular"
    columns: tuple[str, ...] | None = None
    text_column: str | None = None
    image_column: str | None = None
    random_state: int | None = 0
    latent_dim: int = 16
    hidden: tuple[int, ...] = (64,)
    mask_ratio: float = 0.15
    n_mask_views: int = 3
    max_iter: int = 200
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    temperature: float = 0.5
    projector_dim: int = 32
    projector_hidden: tuple[int, ...] = (64,)
    prefer_reduce_components: bool = True
    representation_prefix: str = "ssl_emb"
    backbone: str = "resnet18"
    weight_mode: str = "mock"
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "modality": self.modality,
            "columns": None if self.columns is None else list(self.columns),
            "text_column": self.text_column,
            "image_column": self.image_column,
            "random_state": self.random_state,
            "latent_dim": self.latent_dim,
            "hidden": list(self.hidden),
            "mask_ratio": self.mask_ratio,
            "n_mask_views": self.n_mask_views,
            "max_iter": self.max_iter,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "projector_dim": self.projector_dim,
            "projector_hidden": list(self.projector_hidden),
            "prefer_reduce_components": self.prefer_reduce_components,
            "representation_prefix": self.representation_prefix,
            "backbone": self.backbone,
            "weight_mode": self.weight_mode,
            "hf_model_name": self.hf_model_name,
            "device": self.device,
            "extra": dict(self.extra),
        }
