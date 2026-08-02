"""Vision SSL via torchvision backbone + projector (Session image column)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch
from buildml.dl.image import decode_image_cell
from buildml.dl.zoo import PretrainedBackbone, load_pretrained_backbone
from buildml.selfsupervised.torch import losses, models


class VisionSSLEncoder:
    """Image SSL: pretrained vision backbone + optional SimCLR projector finetune."""

    def __init__(
        self,
        *,
        architecture: str = "resnet18",
        weight_mode: str = "mock",
        latent_dim: int = 128,
        projector_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        temperature: float = 0.5,
        image_size: tuple[int, int] = (32, 32),
        random_state: int | None = 0,
        device: str = "cpu",
    ) -> None:
        self.architecture = architecture
        self.weight_mode = weight_mode
        self.latent_dim = int(latent_dim)
        self.projector_dim = int(projector_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.image_size = tuple(int(v) for v in image_size)
        self.random_state = random_state
        self.device = device
        self._backbone: PretrainedBackbone | None = None
        self._projector: Any = None
        self.pretext_loss_: float | None = None
        self.reconstruction_mae_: float | None = None

    def fit(self, images: list[Any] | np.ndarray, y: Any = None) -> VisionSSLEncoder:
        del y
        torch = require_torch(feature="Vision SSL")
        cells = list(images)
        if len(cells) < 4:
            raise ValidationError("Vision SSL requires at least 4 image samples.")
        self._backbone = load_pretrained_backbone(
            "vision",
            self.architecture,
            weights=self.weight_mode,  # type: ignore[arg-type]
            freeze=False,
        )
        backbone = self._backbone.module
        device = torch.device(self.device)
        backbone = backbone.to(device)
        feat_dim = int(self._backbone.feature_dim)
        self._projector = models.build_projector(
            feat_dim, hidden=(256,), out_dim=self.projector_dim
        ).to(device)
        tensors = self._decode_batch(cells)
        opt = torch.optim.AdamW(
            list(backbone.parameters()) + list(self._projector.parameters()),
            lr=self.learning_rate,
        )
        rng = np.random.default_rng(self.random_state)
        epoch_losses: list[float] = []
        for _ in range(self.epochs):
            indices = rng.permutation(len(tensors))
            batch_loss = 0.0
            n_batches = 0
            for start in range(0, len(indices), self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) < 2:
                    continue
                batch = torch.stack([tensors[i] for i in idx]).to(device)
                v1 = _vision_augment(batch, rng)
                v2 = _vision_augment(batch, rng)
                z1 = _forward_features(backbone, v1)
                z2 = _forward_features(backbone, v2)
                p1 = self._projector(z1)
                p2 = self._projector(z2)
                loss = losses.nt_xent_loss(p1, p2, temperature=self.temperature)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                batch_loss += float(loss.detach().cpu())
                n_batches += 1
            epoch_losses.append(batch_loss / max(n_batches, 1))
        self.pretext_loss_ = float(epoch_losses[-1]) if epoch_losses else None
        self.n_features_in_ = 1
        self.latent_dim = feat_dim
        return self

    def transform(self, images: list[Any] | np.ndarray) -> np.ndarray:
        if self._backbone is None:
            raise ValidationError("VisionSSLEncoder is not fitted.")
        torch = require_torch(feature="Vision SSL transform")
        device = torch.device(self.device)
        backbone = self._backbone.module.to(device)
        backbone.eval()
        tensors = self._decode_batch(list(images))
        with torch.no_grad():
            feats = _forward_features(backbone, torch.stack(tensors).to(device))
            return feats.cpu().numpy()

    def _decode_batch(self, cells: list[Any]) -> list[Any]:
        torch = require_torch(feature="Vision SSL decode")
        out: list[Any] = []
        for cell in cells:
            arr = decode_image_cell(cell, size=self.image_size, channels=3)
            out.append(torch.as_tensor(arr, dtype=torch.float32))
        return out

    def state_dict(self) -> dict[str, Any]:
        if self._backbone is None or self._projector is None:
            raise ValidationError("VisionSSLEncoder is not fitted.")
        return {
            "method": "vision_ssl",
            "architecture": self.architecture,
            "weight_mode": self.weight_mode,
            "latent_dim": self.latent_dim,
            "projector_dim": self.projector_dim,
            "image_size": list(self.image_size),
            "backbone": self._backbone.module.state_dict(),
            "projector": self._projector.state_dict(),
        }


def _forward_features(backbone: Any, batch: Any) -> Any:
    """Extract penultimate features from torchvision-style backbones."""
    torch = require_torch(feature="Vision SSL pool")
    if hasattr(backbone, "forward_features"):
        out = backbone.forward_features(batch)
    else:
        out = backbone(batch)
        if isinstance(out, tuple):
            out = out[0]
    if out.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)
    return out


def _vision_augment(batch: Any, rng: np.random.Generator) -> Any:
    torch = require_torch(feature="Vision SSL augment")
    out = batch.clone()
    if rng.random() > 0.5:
        out = torch.flip(out, dims=[3])
    noise = torch.randn_like(out) * 0.05
    return (out + noise).clamp(0.0, 1.0)
