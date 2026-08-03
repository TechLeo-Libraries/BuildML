"""implicit library adapter (ALS / BPR) for implicit-feedback recommenders."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix

from buildml.core.errors import ValidationError
from buildml.recommenders.extras import require_implicit

ImplicitMethod = Literal["als", "bpr"]


def build_user_item_csr(matrix: np.ndarray) -> csr_matrix:
    """Convert a dense user×item matrix to scipy CSR layout.

    The implicit library expects a sparse confidence-weighted matrix with users
    as rows and items as columns. Call this after
    :func:`~buildml.recommenders.features.build_user_item_matrix` when routing
    to ALS or BPR backends.

    Parameters
    ----------
    matrix:
        Dense ``(n_users, n_items)`` interaction matrix from train data.

    Returns
    -------
    csr_matrix
        Float32 CSR matrix ready for ``implicit`` model ``fit``.
    """
    return csr_matrix(matrix.astype(np.float32))


def fit_implicit_model(
    user_item_csr: csr_matrix,
    *,
    method: ImplicitMethod,
    n_factors: int,
    random_state: int | None,
    n_iterations: int = 15,
) -> Any:
    """Fit an implicit ALS or BPR model on a sparse user×item matrix.

    Wraps the ``implicit`` library for implicit-feedback collaborative
    filtering. The fitted model exposes ``user_factors`` and ``item_factors``
    consumed by :func:`score_implicit_model`.

    Parameters
    ----------
    user_item_csr:
        Sparse CSR matrix with users as rows and items as columns.
    method:
        ``"als"`` for alternating least squares or ``"bpr"`` for Bayesian
        personalized ranking.
    n_factors:
        Latent dimensionality; higher values capture finer taste structure at
        greater compute cost.
    random_state:
        Seed for reproducible factor initialization; ``None`` uses library
        defaults.
    n_iterations:
        Number of training iterations per fit call.

    Returns
    -------
    model
        Fitted ``implicit`` model with ``user_factors`` and ``item_factors``.

    Raises
    ------
    MissingExtraError
        When ``buildml[recommenders-industry]`` is not installed.
    ValidationError
        When ``method`` is not ``"als"`` or ``"bpr"``.
    """
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
    """Score all catalog items for one user via latent factor dot product.

    Computes ``user_factors[user_idx] @ item_factors.T`` and masks excluded
    items with ``-inf`` so they never appear in top-K output.

    Parameters
    ----------
    model:
        Fitted ``implicit`` model from :func:`fit_implicit_model`.
    user_idx:
        Row index of the user in the train interaction matrix.
    n_items:
        Catalog width; scores are padded or truncated to this length.
    exclude_mask:
        Boolean mask over items to suppress (typically train history).

    Returns
    -------
    np.ndarray
        Per-item scores of length ``n_items``; excluded items are ``-inf``.
    """
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
