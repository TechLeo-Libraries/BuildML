"""XGBoost rank:ndcg adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.ranking.extras import require_xgboost
from buildml.ranking.features import query_group_sizes


def fit_rank_ndcg_xgb(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int | None = 0,
    n_estimators: int = 120,
    learning_rate: float = 0.08,
    max_depth: int = 6,
) -> Any:
    """Fit XGBoost rank:ndcg on query-grouped judgment rows.

    Sorts rows by query, builds a grouped DMatrix, and trains with the
    ``rank:ndcg`` objective.

    Parameters
    ----------
    X:
        Standardized train feature matrix.
    y:
        Graded relevance labels aligned with ``groups``.
    groups:
        Query id array with one entry per row.
    random_state:
        Seed passed to XGBoost training.
    n_estimators:
        Number of boosting rounds.
    learning_rate:
        XGBoost ``eta`` learning rate.
    max_depth:
        Maximum tree depth.

    Returns
    -------
    xgboost.Booster
        Fitted rank:ndcg booster ready for :func:`score_xgb`.
    """
    xgb = require_xgboost()
    X_sorted, y_sorted, _, group_sizes = query_group_sizes(X, y, groups)
    dtrain = xgb.DMatrix(X_sorted, label=y_sorted)
    dtrain.set_group(group_sizes)
    params = {
        "objective": "rank:ndcg",
        "eta": float(learning_rate),
        "max_depth": int(max_depth),
        "seed": 0 if random_state is None else int(random_state),
        "verbosity": 0,
    }
    return xgb.train(params, dtrain, num_boost_round=int(n_estimators))


def score_xgb(model: Any, X: np.ndarray) -> np.ndarray:
    """Score rows with a fitted XGBoost ranker.

    Builds a DMatrix and wraps booster ``predict`` for a float score vector
    aligned with ``X``.

    Parameters
    ----------
    model:
        Fitted XGBoost booster from :func:`fit_rank_ndcg_xgb`.
    X:
        Standardized feature matrix to score.

    Returns
    -------
    numpy.ndarray
        Predicted ranking scores, one per row.
    """
    if X.size == 0:
        return np.zeros(0, dtype=float)
    xgb = require_xgboost()
    dmat = xgb.DMatrix(X)
    return np.asarray(model.predict(dmat), dtype=float)
