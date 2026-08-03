"""CatBoost YetiRank adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.ranking.extras import require_catboost


def fit_yetirank_catboost(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int | None = 0,
    iterations: int = 120,
    learning_rate: float = 0.08,
    depth: int = 6,
) -> Any:
    """Fit CatBoost YetiRank on query-grouped judgment rows.

    Builds a CatBoost Pool with ``group_id`` and trains a ranker with the
    YetiRank loss on standardized train features.

    Parameters
    ----------
    X:
        Standardized train feature matrix.
    y:
        Graded relevance labels aligned with ``groups``.
    groups:
        Query id array with one entry per row.
    random_state:
        Seed passed to CatBoost training.
    iterations:
        Number of boosting iterations.
    learning_rate:
        CatBoost learning rate.
    depth:
        Maximum tree depth.

    Returns
    -------
    catboost.CatBoostRanker
        Fitted YetiRank model ready for :func:`score_catboost`.
    """
    catboost = require_catboost()
    pool = catboost.Pool(X, label=y, group_id=groups)
    model = catboost.CatBoostRanker(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="YetiRank",
        random_seed=0 if random_state is None else int(random_state),
        verbose=False,
    )
    model.fit(pool)
    return model


def score_catboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Score rows with a fitted CatBoost ranker.

    Wraps ``model.predict`` and returns a float score vector aligned with ``X``.

    Parameters
    ----------
    model:
        Fitted CatBoost ranker from :func:`fit_yetirank_catboost`.
    X:
        Standardized feature matrix to score.

    Returns
    -------
    numpy.ndarray
        Predicted ranking scores, one per row.
    """
    if X.size == 0:
        return np.zeros(0, dtype=float)
    return np.asarray(model.predict(X), dtype=float)
