"""Torch listwise-lite (ListNet-style) adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.ranking.extras import require_torch_ranking


@dataclass(slots=True)
class ListwiseLiteRanker:
    """Small MLP trained with per-query softmax cross-entropy on relevance."""

    hidden_dim: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    random_state: int | None = 0
    device: str = "cpu"
    n_features: int = 0
    model_: Any = field(default=None, repr=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = require_torch_ranking()
        if self.model_ is None:
            raise ValidationError("ListwiseLiteRanker is not fitted.")
        if X.size == 0:
            return np.zeros(0, dtype=float)
        self.model_.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            scores = self.model_(xt).squeeze(-1)
            return scores.detach().cpu().numpy().astype(float)


def _query_slices(groups: np.ndarray) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    if len(groups) == 0:
        return slices
    start = 0
    current = groups[0]
    for idx in range(1, len(groups)):
        if groups[idx] != current:
            slices.append((start, idx))
            start = idx
            current = groups[idx]
    slices.append((start, len(groups)))
    return slices


def fit_listwise_lite(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    hidden_dim: int = 64,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    random_state: int | None = 0,
    device: str = "cpu",
) -> ListwiseLiteRanker:
    """Fit a listwise-lite MLP with per-query softmax loss on graded relevance."""
    torch = require_torch_ranking()
    nn = torch.nn
    F = torch.nn.functional

    if len(X) < 4:
        raise ValidationError("listwise_lite needs ≥4 train rows.")
    n_features = int(X.shape[1])
    ranker = ListwiseLiteRanker(
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        random_state=random_state,
        device=str(device),
        n_features=n_features,
    )

    torch.manual_seed(0 if random_state is None else int(random_state))
    model = nn.Sequential(
        nn.Linear(n_features, int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    order = np.argsort(groups, kind="mergesort")
    X_ord = np.asarray(X[order], dtype=float)
    y_ord = np.asarray(y[order], dtype=float)
    g_ord = groups[order]
    slices = _query_slices(g_ord)

    for _ in range(int(epochs)):
        model.train()
        total_loss = 0.0
        n_groups = 0
        for start, end in slices:
            if end - start < 2:
                continue
            rel = y_ord[start:end]
            if float(np.max(rel)) <= 0.0:
                continue
            xt = torch.as_tensor(
                X_ord[start:end], dtype=torch.float32, device=device
            )
            scores = model(xt).squeeze(-1)
            target = torch.as_tensor(rel, dtype=torch.float32, device=device)
            target = target / target.sum().clamp_min(1e-8)
            log_probs = F.log_softmax(scores, dim=0)
            loss = -(target * log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_groups += 1
        if n_groups == 0:
            raise ValidationError(
                "listwise_lite needs queries with ≥2 items and positive relevance."
            )

    ranker.model_ = model
    return ranker


def score_listwise_lite(ranker: ListwiseLiteRanker, X: np.ndarray) -> np.ndarray:
    return ranker.predict(X)
