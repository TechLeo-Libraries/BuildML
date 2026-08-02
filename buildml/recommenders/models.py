"""Collaborative filtering and content scoring engines (numpy/sklearn)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from buildml.core.errors import ValidationError


def fit_item_similarity(matrix: np.ndarray) -> np.ndarray:
    """Item-item cosine similarity from user×item matrix."""
    # Items as rows: matrix.T is items × users
    item_vectors = matrix.T
    return cosine_similarity(item_vectors)


def fit_user_similarity(matrix: np.ndarray) -> np.ndarray:
    """User-user cosine similarity from user×item matrix."""
    return cosine_similarity(matrix)


def fit_svd_factors(
    matrix: np.ndarray, *, n_factors: int, random_state: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD factorization: approx R ≈ U Σ V^T → user/item factors."""
    n_users, n_items = matrix.shape
    n_comp = int(min(n_factors, max(1, min(n_users, n_items) - 1)))
    if n_comp < 1:
        raise ValidationError("SVD requires at least 2 users and 2 items.")
    # Center by global mean for stability on sparse explicit ratings
    mask = matrix != 0
    if mask.any():
        global_mean = float(matrix[mask].mean())
    else:
        global_mean = 0.0
    centered = matrix.copy()
    centered[mask] = centered[mask] - global_mean
    model = TruncatedSVD(n_components=n_comp, random_state=random_state)
    user_factors = model.fit_transform(centered)
    item_factors = model.components_.T  # items × factors
    return user_factors, item_factors


def fit_nmf_factors(
    matrix: np.ndarray, *, n_factors: int, random_state: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative MF on non-negative interaction matrix."""
    n_users, n_items = matrix.shape
    n_comp = int(min(n_factors, max(1, min(n_users, n_items) - 1)))
    if n_comp < 1:
        raise ValidationError("NMF requires at least 2 users and 2 items.")
    # NMF needs non-negative; shift if needed (implicit already ≥ 0)
    shifted = matrix.copy()
    min_val = float(shifted.min()) if shifted.size else 0.0
    if min_val < 0:
        shifted = shifted - min_val
    model = NMF(
        n_components=n_comp,
        init="nndsvda",
        random_state=random_state,
        max_iter=400,
    )
    user_factors = model.fit_transform(shifted)
    item_factors = model.components_.T
    return user_factors, item_factors


def score_item_knn(
    matrix: np.ndarray,
    similarity: np.ndarray,
    user_idx: int,
    *,
    n_neighbors: int,
    exclude_mask: np.ndarray,
) -> np.ndarray:
    """Score all items for one user via item-item kNN CF."""
    rated = matrix[user_idx]
    rated_idx = np.flatnonzero(rated != 0)
    scores = np.zeros(matrix.shape[1], dtype=float)
    if rated_idx.size == 0:
        return scores
    for item_j in range(matrix.shape[1]):
        if exclude_mask[item_j]:
            continue
        sims = similarity[item_j, rated_idx]
        if n_neighbors < rated_idx.size:
            keep = np.argpartition(-np.abs(sims), n_neighbors)[:n_neighbors]
            sims = sims[keep]
            neighbors = rated_idx[keep]
        else:
            neighbors = rated_idx
        weights = sims
        denom = np.abs(weights).sum()
        if denom < 1e-12:
            continue
        scores[item_j] = float(np.dot(weights, rated[neighbors]) / denom)
    return scores


def score_user_knn(
    matrix: np.ndarray,
    similarity: np.ndarray,
    user_idx: int,
    *,
    n_neighbors: int,
    exclude_mask: np.ndarray,
) -> np.ndarray:
    """Score all items for one user via user-user kNN CF."""
    sims = similarity[user_idx].copy()
    sims[user_idx] = 0.0
    if n_neighbors < len(sims):
        neighbor_idx = np.argpartition(-np.abs(sims), n_neighbors)[:n_neighbors]
    else:
        neighbor_idx = np.flatnonzero(sims != 0)
    if neighbor_idx.size == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    weights = sims[neighbor_idx]
    neighbor_ratings = matrix[neighbor_idx]  # neighbors × items
    # Only count neighbors who rated each item
    rated_mask = neighbor_ratings != 0
    numer = (weights[:, None] * neighbor_ratings).sum(axis=0)
    denom = (np.abs(weights)[:, None] * rated_mask).sum(axis=0)
    scores = np.zeros(matrix.shape[1], dtype=float)
    valid = denom > 1e-12
    scores[valid] = numer[valid] / denom[valid]
    scores[exclude_mask] = 0.0
    return scores


def score_factorization(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_idx: int,
    *,
    global_mean: float,
    exclude_mask: np.ndarray,
) -> np.ndarray:
    scores = global_mean + user_factors[user_idx] @ item_factors.T
    scores = np.asarray(scores, dtype=float)
    scores[exclude_mask] = -np.inf
    return scores


def score_content(
    matrix: np.ndarray,
    item_features: np.ndarray,
    user_idx: int,
    *,
    exclude_mask: np.ndarray,
) -> np.ndarray:
    """User profile = rating-weighted mean of interacted item features."""
    rated = matrix[user_idx]
    rated_idx = np.flatnonzero(rated != 0)
    if rated_idx.size == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    weights = rated[rated_idx]
    profile = np.average(item_features[rated_idx], axis=0, weights=np.abs(weights))
    profile_norm = np.linalg.norm(profile)
    if profile_norm < 1e-12:
        return np.zeros(matrix.shape[1], dtype=float)
    item_norms = np.linalg.norm(item_features, axis=1)
    item_norms = np.where(item_norms < 1e-12, 1.0, item_norms)
    scores = (item_features @ profile) / (item_norms * profile_norm)
    scores = np.asarray(scores, dtype=float)
    scores[exclude_mask] = -np.inf
    return scores


def popularity_scores(item_popularity: np.ndarray, exclude_mask: np.ndarray) -> np.ndarray:
    scores = item_popularity.astype(float).copy()
    scores[exclude_mask] = -np.inf
    return scores


def top_k_from_scores(
    scores: np.ndarray,
    item_ids: tuple,
    k: int,
) -> tuple[tuple, tuple[float, ...]]:
    if k <= 0 or scores.size == 0:
        return (), ()
    finite = np.isfinite(scores)
    if not finite.any():
        return (), ()
    # Mask non-finite as very low so they never win
    work = scores.copy()
    work[~finite] = -np.inf
    k_eff = min(int(k), int(finite.sum()))
    if k_eff <= 0:
        return (), ()
    if k_eff < len(work):
        idx = np.argpartition(-work, k_eff)[:k_eff]
        idx = idx[np.argsort(-work[idx])]
    else:
        idx = np.argsort(-work)[:k_eff]
    rec_items = tuple(item_ids[i] for i in idx)
    rec_scores = tuple(float(work[i]) for i in idx)
    return rec_items, rec_scores
