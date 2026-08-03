"""LightGBM LambdaRank adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.ranking.extras import require_lightgbm
from buildml.ranking.features import query_group_sizes


def fit_lambdarank_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int | None = 0,
    n_estimators: int = 120,
    learning_rate: float = 0.08,
    num_leaves: int = 31,
    ndcg_at: int = 10,
) -> Any:
    """Fit LightGBM LambdaRank on query-grouped judgment rows.

    Sorts rows by query, builds a grouped Dataset, and trains with the
    ``lambdarank`` objective and nDCG evaluation metric.

    Parameters
    ----------
    X:
        Standardized train feature matrix.
    y:
        Graded relevance labels aligned with ``groups``.
    groups:
        Query id array with one entry per row.
    random_state:
        Seed passed to LightGBM training.
    n_estimators:
        Number of boosting rounds.
    learning_rate:
        LightGBM learning rate.
    num_leaves:
        Maximum number of leaves per tree.
    ndcg_at:
        Cutoff list for the nDCG training metric.

    Returns
    -------
    lightgbm.Booster
        Fitted LambdaRank booster ready for :func:`score_lgbm`.
    """
    lgb = require_lightgbm()
    X_sorted, y_sorted, _, group_sizes = query_group_sizes(X, y, groups)
    train_data = lgb.Dataset(X_sorted, label=y_sorted, group=group_sizes)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [int(ndcg_at)],
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "verbosity": -1,
        "seed": 0 if random_state is None else int(random_state),
        "force_col_wise": True,
    }
    return lgb.train(
        params,
        train_data,
        num_boost_round=int(n_estimators),
    )


def score_lgbm(model: Any, X: np.ndarray) -> np.ndarray:
    """Score rows with a fitted LightGBM ranker.

    Wraps booster ``predict`` and returns a float score vector aligned with
    ``X``.

    Parameters
    ----------
    model:
        Fitted LightGBM booster from :func:`fit_lambdarank_lgbm`.
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
