"""Sklearn pointwise / pairwise ranking adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.ranking.models import (
    fit_pairwise_ranksvm,
    fit_pointwise,
    score_linear,
    score_pointwise,
)


@dataclass(slots=True)
class SklearnRankerState:
    """Frozen sklearn ranker state for RankerPlan."""

    method: str
    pointwise_estimator: str = "ridge"
    pairwise_estimator: str = "ranksvm"
    estimator_: Any = field(default=None, repr=False)
    coef_: np.ndarray | None = field(default=None, repr=False)
    intercept_: float = 0.0
    n_pairwise_examples: int | None = None


def build_sklearn_ranker(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    method: str,
    pointwise_estimator: str = "ridge",
    pairwise_estimator: str = "ranksvm",
    alpha: float = 1.0,
    C: float = 1.0,
    max_pairs_per_query: int = 80,
    random_state: int | None = 0,
) -> SklearnRankerState:
    """Fit sklearn pointwise or pairwise ranker on standardized features.

    Dispatches to Ridge/HGB pointwise regression or RankSVM-lite pairwise
    training and packages the result in a :class:`SklearnRankerState`.

    Parameters
    ----------
    X:
        Standardized train feature matrix.
    y:
        Graded relevance labels aligned with ``groups``.
    groups:
        Query id array with one entry per row (pairwise path only).
    method:
        ``pointwise`` or ``pairwise`` sklearn ranker mode.
    pointwise_estimator:
        ``ridge`` or ``hgb`` when ``method='pointwise'``.
    pairwise_estimator:
        ``ranksvm`` when ``method='pairwise'``.
    alpha:
        Ridge regularization for pointwise Ridge.
    C:
        LinearSVC regularization for pairwise RankSVM-lite.
    max_pairs_per_query:
        Cap on oriented pairs sampled per train query.
    random_state:
        Seed for pair sampling and stochastic estimators.

    Returns
    -------
    SklearnRankerState
        Frozen sklearn ranker state for persistence on :class:`RankerPlan`.
    """
    state = SklearnRankerState(
        method=method,
        pointwise_estimator=pointwise_estimator,
        pairwise_estimator=pairwise_estimator,
    )
    if method == "pointwise":
        state.estimator_ = fit_pointwise(
            X,
            y,
            estimator=pointwise_estimator,
            alpha=alpha,
            random_state=random_state,
        )
        if hasattr(state.estimator_, "coef_"):
            state.coef_ = np.asarray(state.estimator_.coef_, dtype=float).ravel()
            state.intercept_ = float(getattr(state.estimator_, "intercept_", 0.0))
            if np.ndim(state.estimator_.intercept_) > 0:
                state.intercept_ = float(
                    np.asarray(state.estimator_.intercept_).ravel()[0]
                )
        return state
    estimator, coef, intercept, n_pairs = fit_pairwise_ranksvm(
        X,
        y,
        groups,
        C=C,
        max_pairs_per_query=max_pairs_per_query,
        random_state=random_state,
    )
    state.estimator_ = estimator
    state.coef_ = coef
    state.intercept_ = intercept
    state.n_pairwise_examples = n_pairs
    return state


def score_sklearn_ranker(state: SklearnRankerState, X: np.ndarray) -> np.ndarray:
    """Score rows with a fitted sklearn ranker state.

    Dispatches to pointwise prediction or linear pairwise scoring depending on
    the method stored on ``state``.

    Parameters
    ----------
    state:
        Frozen sklearn ranker state from :func:`build_sklearn_ranker`.
    X:
        Standardized feature matrix to score.

    Returns
    -------
    numpy.ndarray
        Predicted ranking scores, one per row.

    Raises
    ------
    ValueError
        When required estimator or coefficient state is missing from ``state``.
    """
    if state.method == "pointwise":
        if state.estimator_ is None:
            raise ValueError("SklearnRankerState missing pointwise estimator.")
        return score_pointwise(state.estimator_, X)
    if state.coef_ is None:
        raise ValueError("SklearnRankerState missing pairwise coefficients.")
    return score_linear(state.coef_, state.intercept_, X)
