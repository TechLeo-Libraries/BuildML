"""Train-only tabular autoencoder reconstruction-error anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from buildml.anomaly.extras import require_torch_anomaly
from buildml.core.errors import ValidationError


def _build_mlp_autoencoder(input_dim: int, latent_dim: int) -> Any:
    torch = require_torch_anomaly(feature="Torch autoencoder anomaly detector")
    import torch.nn as nn

    class TabularAnomalyAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = max(latent_dim * 2, 32)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, input_dim),
            )

        def forward(self, x: Any) -> tuple[Any, Any]:
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon

    return TabularAnomalyAutoencoder()


@dataclass
class TorchAnomalyAutoencoder:
    """Frozen Torch autoencoder for reconstruction-error anomaly scoring."""

    model: Any
    input_dim: int
    latent_dim: int
    epochs: int
    batch_size: int
    train_mse_: float

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Perform reconstruction error for the Session-facing workflow step.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.
        """
        torch = require_torch_anomaly()
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32)
            _, recon = self.model(tensor)
            err = torch.mean((tensor - recon) ** 2, dim=1)
            return err.cpu().numpy().astype(float)


def build_torch_autoencoder(
    x_fit: np.ndarray,
    *,
    latent_dim: int = 8,
    epochs: int = 40,
    batch_size: int = 64,
    random_state: int | None = 0,
) -> TorchAnomalyAutoencoder:
    """Fit a train-only autoencoder; scores are per-row MSE reconstruction error.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
x_fit:
    x fit (np.ndarray).
latent_dim:
    latent dim (int).
epochs:
    Training epochs for torch-backed estimators.
batch_size:
    Number of rows to select per query or training minibatch.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
TorchAnomalyAutoencoder
    Return value (TorchAnomalyAutoencoder) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    torch = require_torch_anomaly(feature="Torch autoencoder anomaly detector")
    import torch.nn as nn

    if x_fit.ndim != 2 or x_fit.shape[0] < 5:
        raise ValidationError("Torch autoencoder requires at least 5 train rows.")
    input_dim = int(x_fit.shape[1])
    latent_dim = max(2, min(int(latent_dim), input_dim))
    rng = np.random.default_rng(random_state)
    model = _build_mlp_autoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    n = x_fit.shape[0]
    batch_size = max(8, min(int(batch_size), n))
    indices = np.arange(n)
    for _ in range(int(epochs)):
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = torch.as_tensor(x_fit[batch_idx], dtype=torch.float32)
            optimizer.zero_grad()
            _, recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
    train_err = build_torch_autoencoder_score_helper(model, x_fit)
    train_mse = float(np.mean(train_err))
    return TorchAnomalyAutoencoder(
        model=model,
        input_dim=input_dim,
        latent_dim=latent_dim,
        epochs=int(epochs),
        batch_size=batch_size,
        train_mse_=train_mse,
    )


def build_torch_autoencoder_score_helper(model: Any, x: np.ndarray) -> np.ndarray:
    """Construct a torch autoencoder score helper ready for fit or scoring.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
model:
    model (Any).
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.
    """
    torch = require_torch_anomaly()
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32)
        _, recon = model(tensor)
        err = torch.mean((tensor - recon) ** 2, dim=1)
        return err.cpu().numpy().astype(float)


def torch_ae_anomaly_scores(estimator: TorchAnomalyAutoencoder, *, x: np.ndarray) -> np.ndarray:
    """Perform torch ae anomaly scores for the Session-facing workflow step.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
estimator:
    Fitted model object used for scoring or prediction.
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.
    """
    return estimator.reconstruction_error(x)
