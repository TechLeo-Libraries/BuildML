"""implicit library adapter (ALS / BPR) for implicit-feedback recommenders."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix

from buildml.core.errors import ValidationError
from buildml.recommenders.extras import require_implicit

ImplicitMethod = Literal["als", "bpr"]


def build_user_item_csr(matrix: np.ndarray) -> csr_matrix:
    """Convert dense user×item matrix to CSR (users × items)."""
    return csr_matrix(matrix.astype(np.float32))


def fit_implicit_model(
    user_item_csr: csr_matrix,
    *,
    method: ImplicitMethod,
    n_factors: int,
    random_state: int | None,
    n_iterations: int = 15,
) -> Any:
    """Fit an implicit ALS or BPR model on a sparse user×item matrix."""
    require_implicit(feature=f"implicit method '{method}'")
    if method == "als":
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=int(n_factors),
            random_state=random_state,
            iterations=int(n_iterations),
        )
    elif method == "bpr":
        from implicit.bpr import BayesianPersonalizedRanking

        model = BayesianPersonalizedRanking(
            factors=int(n_factors),
            random_state=random_state,
            iterations=int(n_iterations),
        )
    else:
        raise ValidationError(f"Unsupported implicit method '{method}'")

    # implicit expects confidence-weighted user×item CSR
    model.fit(user_item_csr)
    return model


def score_implicit_model(
    model: Any,
    user_idx: int,
    *,
    n_items: int,
    exclude_mask: np.ndarray,
) -> np.ndarray:
    """Score all catalog items for one user via latent factor dot product."""
    require_implicit(feature="implicit scoring")
    user_factors = np.asarray(model.user_factors, dtype=float)
    item_factors = np.asarray(model.item_factors, dtype=float)
    if user_idx < 0 or user_idx >= user_factors.shape[0]:
        return np.zeros(n_items, dtype=float)
    scores = user_factors[user_idx] @ item_factors.T
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size != n_items:
        padded = np.zeros(n_items, dtype=float)
        n = min(scores.size, n_items)
        padded[:n] = scores[:n]
        scores = padded
    scores[exclude_mask] = -np.inf
    return scores
