"""Learned metric encoder for torch CBR backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.cbr.extras import require_torch_cbr
from buildml.core.errors import ValidationError


@dataclass
class TorchMetricEncoder:
    """Lite supervised metric encoder: MLP trunk + kNN in embedding space."""

    hidden_dim: int = 64
    embed_dim: int = 32
    epochs: int = 40
    learning_rate: float = 1e-3
    device: str = "cpu"
    random_state: int | None = 0
    n_features_: int = 0
    n_classes_: int = 0
    task_: str = "classification"
    module_: Any = field(default=None, repr=False)

    def encode(self, x: np.ndarray) -> np.ndarray:
        torch = require_torch_cbr(feature="CBR learned-metric encoding")
        if self.module_ is None:
            raise ValidationError("TorchMetricEncoder is not fitted.")
        self.module_.eval()
        device = torch.device(self.device)
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=device)
            emb = self.module_.encode(xt)
        return emb.cpu().numpy()


def build_torch_encoder(
    n_features: int,
    *,
    n_classes: int,
    task: str,
    hidden_dim: int = 64,
    embed_dim: int = 32,
    device: str = "cpu",
) -> Any:
    """Build a small MLP encoder + task head."""
    torch = require_torch_cbr(feature="CBR learned-metric encoder")
    task_key = str(task).lower()

    class _Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, embed_dim),
                torch.nn.ReLU(),
            )
            out_dim = 1 if task_key == "regression" else max(n_classes, 2)
            self.head = torch.nn.Linear(embed_dim, out_dim)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.trunk(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(x))

    return _Encoder().to(torch.device(device))


def fit_torch_encoder(
    encoder: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    task: str,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    random_state: int | None = 0,
) -> Any:
    """Supervised pretrain encoder; embedding layer used for kNN retrieval."""
    torch = require_torch_cbr(feature="CBR learned-metric training")
    if int(random_state) is not None:
        torch.manual_seed(int(random_state))
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train)
    dev = torch.device(device)
    encoder = encoder.to(dev)
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=float(learning_rate))
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    task_key = str(task).lower()
    if task_key == "regression":
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
        for _ in range(int(epochs)):
            opt.zero_grad()
            pred = encoder(xt).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, yt)
            loss.backward()
            opt.step()
    else:
        yt = torch.as_tensor(y, dtype=torch.long, device=dev)
        for _ in range(int(epochs)):
            opt.zero_grad()
            logits = encoder(xt)
            loss = torch.nn.functional.cross_entropy(logits, yt)
            loss.backward()
            opt.step()
    encoder.eval()
    return encoder


def encode_with_torch(encoder: Any, x: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    torch = require_torch_cbr(feature="CBR learned-metric encoding")
    encoder.eval()
    dev = torch.device(device)
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float32, device=dev)
        emb = encoder.encode(xt)
    return emb.cpu().numpy()
