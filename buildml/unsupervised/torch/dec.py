"""Deep Embedded Clustering (DEC) and Improved DEC (IDEC) for tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from buildml.dl.extras import require_torch
from buildml.unsupervised.backends import FitOutcome


def _centroids_from_labels(
    x: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray | None, tuple[int, ...]]:
    ids = sorted({int(v) for v in labels if int(v) >= 0})
    if not ids:
        return None, ()
    centers = [x[np.asarray(labels) == label].mean(axis=0) for label in ids]
    return np.asarray(centers, dtype=float), tuple(ids)


def _build_autoencoder(input_dim: int, latent_dim: int) -> Any:
    torch = require_torch(feature="Deep clustering (DEC/IDEC)")
    import torch.nn as nn

    class TabularAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = max(latent_dim * 2, 32)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, h),
                nn.ReLU(),
                nn.Linear(h, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, h),
                nn.ReLU(),
                nn.Linear(h, input_dim),
            )

        def forward(self, x: Any) -> tuple[Any, Any]:
            z = self.encoder(x)
            return z, self.decoder(z)

    return TabularAutoencoder()


def _student_t_q(z: Any, centers: Any, alpha: float = 1.0) -> Any:
    torch = require_torch()
    dist = torch.sum((z.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
    q = (1.0 + dist / alpha) ** (-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True)


def _target_distribution(q: np.ndarray) -> np.ndarray:
    weight = (q**2) / np.maximum(q.sum(axis=0, keepdims=True), 1e-12)
    p = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return p.astype(np.float64)


@dataclass
class DECModel:
    """Torch DEC/IDEC bundle for predict and serialization."""

    autoencoder: Any
    cluster_centers: np.ndarray
    method: str
    n_clusters: int
    latent_dim: int
    input_dim: int

    def predict_latent(self, x: np.ndarray) -> np.ndarray:
        """Perform predict latent for the Session-facing workflow step.

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
        torch = require_torch()
        self.autoencoder.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32)
            z, _ = self.autoencoder(tensor)
            return z.cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run predict on input data using the fitted internal state.

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
        torch = require_torch()
        self.autoencoder.eval()
        with torch.no_grad():
            z = torch.as_tensor(self.predict_latent(x), dtype=torch.float32)
            centers = torch.as_tensor(self.cluster_centers, dtype=torch.float32)
            q = _student_t_q(z, centers)
            return q.argmax(dim=1).cpu().numpy().astype(int)


def fit_dec_idec(
    x: np.ndarray,
    *,
    method: Literal["dec", "idec"],
    n_clusters: int,
    latent_dim: int,
    pretrain_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int | None,
) -> FitOutcome:
    """Fit dec idec on the train partition using the recorded contract.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
x:
    Feature matrix input rows.
method:
    Method or strategy identifier for the resolved backend.
n_clusters:
    Target number of clusters for partitioning.
latent_dim:
    latent dim (int).
pretrain_epochs:
    pretrain epochs (int).
finetune_epochs:
    finetune epochs (int).
batch_size:
    Number of rows to select per query or training minibatch.
learning_rate:
    Optimizer learning rate for torch training.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
FitOutcome
    Return value (FitOutcome) produced by this operation.
    """
    torch = require_torch(feature=f"Deep clustering ({method.upper()})")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(random_state)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    n_samples, input_dim = x.shape
    device = torch.device("cpu")

    ae = _build_autoencoder(input_dim, latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    ds = TensorDataset(torch.as_tensor(x, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=min(batch_size, n_samples), shuffle=True)

    # Phase 1: autoencoder pretrain
    ae.train()
    for _ in range(pretrain_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            z, recon = ae(batch)
            loss = mse(recon, batch)
            loss.backward()
            opt.step()

    # Init cluster centers with k-means on latent
    ae.eval()
    with torch.no_grad():
        z_all = ae.encoder(torch.as_tensor(x, dtype=torch.float32).to(device)).cpu().numpy()
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    init_labels = km.fit_predict(z_all)
    centers = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=device)

    # Phase 2: DEC / IDEC clustering
    ae.train()
    for _ in range(finetune_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            z, recon = ae(batch)
            q = _student_t_q(z, centers)
            p = torch.as_tensor(
                _target_distribution(q.detach().cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )
            kl = torch.mean(torch.sum(p * torch.log((p + 1e-10) / (q + 1e-10)), dim=1))
            if method == "idec":
                loss = kl + mse(recon, batch)
            else:
                loss = kl
            loss.backward()
            opt.step()

    model = DECModel(
        autoencoder=ae.cpu(),
        cluster_centers=centers.cpu().numpy(),
        method=method,
        n_clusters=n_clusters,
        latent_dim=latent_dim,
        input_dim=input_dim,
    )
    labels = model.predict(x)
    centroids, cids = _centroids_from_labels(x, labels)
    disclosures = [
        f"{method.upper()} deep clustering via Torch autoencoder + soft KMeans init.",
        "Holdout assign uses native encoder + student-t soft assignment (train-fitted centers).",
        "Requires buildml[torch]; not interchangeable with sklearn ClusterPlan estimators.",
    ]
    return FitOutcome(
        labels=labels,
        estimator=model,
        n_clusters=n_clusters,
        centroids=centroids,
        centroid_ids=cids,
        core_idx=(),
        assign_strategy="native",
        inertia=None,
        warnings=[],
        disclosures=disclosures,
        extra={"latent_dim": latent_dim, "method": method},
    )


def predict_dec_idec(estimator: DECModel, x: np.ndarray) -> np.ndarray:
    """Perform predict dec idec for the Session-facing workflow step.

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
    return estimator.predict(x)
