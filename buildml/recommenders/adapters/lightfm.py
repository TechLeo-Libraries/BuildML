"""LightFM hybrid recommender adapter."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from buildml.core.errors import ValidationError
from buildml.recommenders.extras import require_lightfm


def _feature_csr(
    frame: pd.DataFrame,
    id_column: str,
    ids: tuple[Any, ...],
    feature_columns: Sequence[str] | None,
) -> csr_matrix | None:
    if not feature_columns:
        return None
    cols = [str(c) for c in feature_columns]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValidationError(f"LightFM feature columns missing: {missing}")
    for col in cols:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValidationError(
                f"LightFM requires numeric side features; {col!r} is not numeric."
            )
    rows: list[np.ndarray] = []
    for entity in ids:
        sub = frame.loc[frame[id_column] == entity, cols]
        if sub.empty:
            rows.append(np.zeros(len(cols), dtype=float))
        else:
            rows.append(sub.to_numpy(dtype=float).mean(axis=0))
    feats = np.vstack(rows) if rows else np.zeros((0, len(cols)))
    mean = np.mean(feats, axis=0) if len(feats) else np.zeros(len(cols))
    scale = np.std(feats, axis=0) if len(feats) else np.ones(len(cols))
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (feats - mean) / scale
    return csr_matrix(standardized.astype(np.float32))


def fit_lightfm_model(
    interactions_csr: csr_matrix,
    *,
    n_factors: int,
    random_state: int | None,
    feedback: str,
    user_features: csr_matrix | None = None,
    item_features: csr_matrix | None = None,
    epochs: int = 10,
) -> Any:
    """Fit a LightFM hybrid model on sparse interactions and side features.

    Trains a WARP-loss factorization that can incorporate optional user and
    item side features. The fitted model is scored via
    :func:`score_lightfm_model`.

    Parameters
    ----------
    interactions_csr:
        Sparse CSR user×item interaction matrix from train data.
    n_factors:
        Latent embedding width for users and items.
    random_state:
        Seed for reproducible initialization; ``None`` uses library defaults.
    feedback:
        ``"implicit"`` or ``"explicit"``; selects the LightFM loss configuration.
    user_features:
        Optional CSR user side-feature matrix aligned to train user order.
    item_features:
        Optional CSR item side-feature matrix aligned to train item order.
    epochs:
        Number of full passes over the interaction matrix during training.

    Returns
    -------
    model
        Fitted ``lightfm.LightFM`` instance ready for ``predict``.
    """
    require_lightfm(feature="LightFM hybrid recommender")
    from lightfm import LightFM

    loss = "warp" if feedback == "implicit" else "warp"
    if feedback == "explicit":
        loss = "warp"
    model = LightFM(
        no_components=int(n_factors),
        random_state=random_state,
        loss=loss,
    )
    model.fit(
        interactions_csr,
        user_features=user_features,
        item_features=item_features,
        epochs=int(epochs),
        num_threads=1,
    )
    return model


def score_lightfm_model(
    model: Any,
    user_idx: int,
    *,
    n_items: int,
    exclude_mask: np.ndarray,
    user_features: csr_matrix | None = None,
    item_features: csr_matrix | None = None,
) -> np.ndarray:
    """Score all catalog items for one user via LightFM predict.

    Calls ``model.predict`` for every item id with the same user index and
    masks excluded items with ``-inf``.

    Parameters
    ----------
    model:
        Fitted LightFM model from :func:`fit_lightfm_model`.
    user_idx:
        Row index of the user in the train interaction matrix.
    n_items:
        Catalog width; one score is produced per item index ``0 .. n_items-1``.
    exclude_mask:
        Boolean mask over items to suppress (typically train history).
    user_features:
        Optional CSR user side features stored on the plan.
    item_features:
        Optional CSR item side features stored on the plan.

    Returns
    -------
    np.ndarray
        Per-item scores of length ``n_items``; excluded items are ``-inf``.
    """
    require_lightfm(feature="LightFM scoring")
    item_ids = np.arange(n_items, dtype=np.int32)
    user_ids = np.full(n_items, int(user_idx), dtype=np.int32)
    scores = model.predict(
        user_ids,
        item_ids,
        user_features=user_features,
        item_features=item_features,
    )
    scores = np.asarray(scores, dtype=float).reshape(-1)
    scores[exclude_mask] = -np.inf
    return scores
