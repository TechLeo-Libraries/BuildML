"""Pure-Torch sparse/dense GCN for Session Graph ML (no PyTorch Geometric).

Justification
-------------
PyTorch Geometric is intentionally **not** required. It couples tightly to
specific Torch/CUDA builds and pulls a heavy stack. A 1–2 layer GCN with
symmetric normalized adjacency (Kipf & Welling) is implementable with core
``torch`` matmul on the dense normalized adjacency already built for the
Session size limit (≤5000 nodes). That keeps ``import buildml`` light and
makes ``buildml[torch]`` sufficient for the GNN path; classical features
remain behind ``buildml[graph]`` (NetworkX).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch


class GCNClassifier:
    """Small supervised GCN for node classification (train-mask loss)."""

    def __init__(
        self,
        *,
        in_dim: int,
        n_classes: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        epochs: int = 80,
        random_state: int | None = 0,
    ) -> None:
        if in_dim < 1:
            raise ValidationError("GCN in_dim must be >= 1.")
        if n_classes < 2:
            raise ValidationError("GCN requires at least 2 classes.")
        if n_layers not in {1, 2}:
            raise ValidationError("This surface supports n_layers in {1, 2} only.")
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.random_state = random_state
        self._module: Any = None
        self._torch = None
        self.train_losses_: list[float] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        adj_norm: np.ndarray,
        train_mask: np.ndarray,
        class_to_index: dict[Any, int],
    ) -> GCNClassifier:
        torch = require_torch(feature="Graph GCN node classification")
        self._torch = torch
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        device = torch.device("cpu")
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(adj_norm, dtype=torch.float32, device=device)
        y_idx = np.asarray([class_to_index[v] for v in y.tolist()], dtype=np.int64)
        y_t = torch.as_tensor(y_idx, dtype=torch.long, device=device)
        mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
        if int(mask.sum().item()) < 2:
            raise ValidationError("GCN fit needs at least 2 labeled train nodes.")

        module = _GCNModule(
            self.in_dim,
            self.hidden_dim,
            self.n_classes,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(device)
        opt = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        self.train_losses_ = []
        module.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = module(x_t, a_t)
            loss = loss_fn(logits[mask], y_t[mask])
            loss.backward()
            opt.step()
            self.train_losses_.append(float(loss.detach().cpu().item()))
        module.eval()
        self._module = module
        return self

    def predict_proba(self, x: np.ndarray, adj_norm: np.ndarray) -> np.ndarray:
        torch = self._torch or require_torch(feature="Graph GCN node classification")
        if self._module is None:
            raise ValidationError("GCNClassifier is not fitted.")
        device = next(self._module.parameters()).device
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(adj_norm, dtype=torch.float32, device=device)
        self._module.eval()
        with torch.no_grad():
            logits = self._module(x_t, a_t)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return np.asarray(proba, dtype=np.float64)

    def predict(self, x: np.ndarray, adj_norm: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x, adj_norm)
        return proba.argmax(axis=1)

    def to_state(self) -> dict[str, Any]:
        torch = self._torch or require_torch(feature="Graph GCN node classification")
        if self._module is None:
            raise ValidationError("GCNClassifier is not fitted.")
        return {
            "in_dim": self.in_dim,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "random_state": self.random_state,
            "state_dict": {k: v.detach().cpu() for k, v in self._module.state_dict().items()},
            "train_losses": list(self.train_losses_),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> GCNClassifier:
        torch = require_torch(feature="Graph GCN node classification")
        obj = cls(
            in_dim=int(state["in_dim"]),
            n_classes=int(state["n_classes"]),
            hidden_dim=int(state["hidden_dim"]),
            n_layers=int(state["n_layers"]),
            dropout=float(state["dropout"]),
            learning_rate=float(state["learning_rate"]),
            weight_decay=float(state["weight_decay"]),
            epochs=int(state["epochs"]),
            random_state=state.get("random_state"),
        )
        module = _GCNModule(
            obj.in_dim,
            obj.hidden_dim,
            obj.n_classes,
            n_layers=obj.n_layers,
            dropout=obj.dropout,
        )
        module.load_state_dict(state["state_dict"])
        module.eval()
        obj._module = module
        obj._torch = torch
        obj.train_losses_ = list(state.get("train_losses") or [])
        return obj


def _GCNModule(  # noqa: N802 — factory matching nn.Module usage
    in_dim: int,
    hidden_dim: int,
    n_classes: int,
    *,
    n_layers: int,
    dropout: float,
) -> Any:
    torch = require_torch(feature="Graph GCN node classification")

    class _Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_layers = n_layers
            self.dropout = torch.nn.Dropout(dropout)
            if n_layers == 1:
                self.lin1 = torch.nn.Linear(in_dim, n_classes)
                self.lin2 = None
            else:
                self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
                self.lin2 = torch.nn.Linear(hidden_dim, n_classes)

        def forward(self, x: Any, adj_norm: Any) -> Any:
            h = adj_norm @ x
            h = self.lin1(h)
            if self.lin2 is None:
                return h
            h = torch.relu(h)
            h = self.dropout(h)
            h = adj_norm @ h
            return self.lin2(h)

    return _Mod()
