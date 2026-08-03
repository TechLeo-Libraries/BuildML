"""Pointwise and pairwise ranking estimators (sklearn / numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

from buildml.core.errors import ValidationError


def fit_pointwise(
    X: np.ndarray,
    y: np.ndarray,
    *,
    estimator: str = "ridge",
    alpha: float = 1.0,
    random_state: int | None = 0,
) -> Any:
    """Fit a pointwise relevance regressor on standardized features.

    Trains Ridge or HistGradientBoostingRegressor to predict graded relevance
    labels directly from feature vectors.

    Parameters
    ----------
    X:
        Standardized train feature matrix.
    y:
        Graded relevance labels aligned with ``X``.
    estimator:
        ``ridge`` or ``hgb`` pointwise backend.
    alpha:
        Ridge regularization strength when ``estimator='ridge'``.
    random_state:
        Seed passed to the underlying sklearn estimator.

    Returns
    -------
    sklearn estimator
        Fitted pointwise regressor ready for :func:`score_pointwise`.

    Raises
    ------
    ValidationError
        When ``estimator`` is not ``ridge`` or ``hgb``.
    """
    if estimator == "ridge":
        model = Ridge(alpha=float(alpha), random_state=random_state)
        model.fit(X, y)
        return model
    if estimator == "hgb":
        model = HistGradientBoostingRegressor(
            max_depth=4,
            max_iter=100,
            learning_rate=0.1,
            random_state=random_state,
        )
        model.fit(X, y)
        return model
    raise ValidationError(
        f"Unknown pointwise_estimator={estimator!r}; expected 'ridge' or 'hgb'."
    )


def score_pointwise(model: Any, X: np.ndarray) -> np.ndarray:
    """Score rows with a fitted pointwise relevance regressor.

    Calls ``model.predict`` and coerces the output to a float numpy vector
    aligned with ``X``.

    Parameters
    ----------
    model:
        Fitted sklearn regressor from :func:`fit_pointwise`.
    X:
        Standardized feature matrix to score.

    Returns
    -------
    numpy.ndarray
        Predicted relevance scores, one per row.
    """
    if X.size == 0:
        return np.zeros(0, dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _pairwise_examples(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    max_pairs_per_query: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build RankSVM-style difference features within each query group."""
    rng = np.random.default_rng(random_state)
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    n_pairs = 0
    for qid in np.unique(groups):
        idx = np.where(groups == qid)[0]
        if len(idx) < 2:
            continue
        rel = y[idx]
        # Prefer pairs with distinct relevance
        order = np.argsort(-rel)
        idx = idx[order]
        rel = rel[order]
        candidates: list[tuple[int, int]] = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if rel[a] == rel[b]:
                    continue
                candidates.append((int(idx[a]), int(idx[b])))
        if not candidates:
            continue
        if len(candidates) > max_pairs_per_query:
            pick = rng.choice(len(candidates), size=max_pairs_per_query, replace=False)
            candidates = [candidates[i] for i in pick]
        for i, j in candidates:
            # Always orient so label = +1 (higher relevance first)
            if y[i] >= y[j]:
                diffs.append(X[i] - X[j])
                labels.append(1)
            else:
                diffs.append(X[j] - X[i])
                labels.append(1)
            # Also add the swapped orientation for balanced LinearSVC
            diffs.append(diffs[-1] * -1.0)
            labels.append(-1)
            n_pairs += 1
    if not diffs:
        raise ValidationError(
            "pairwise RankSVM needs queries with ≥2 items and distinct "
            "relevance grades in train."
        )
    return np.vstack(diffs), np.asarray(labels, dtype=int), n_pairs


def fit_pairwise_ranksvm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    C: float = 1.0,
    max_pairs_per_query: int = 80,
    random_state: int | None = 0,
) -> tuple[LinearSVC, np.ndarray, float, int]:
    """Fit RankSVM-lite using LinearSVC on within-query feature differences.

    Builds oriented pair examples per query, caps pairs with
    ``max_pairs_per_query``, and returns the linear scoring weights.

    Parameters
    ----------
    X:
        Standardized train feature matrix.
    y:
        Graded relevance labels aligned with ``groups``.
    groups:
        Query id array with one entry per row.
    C:
        LinearSVC regularization inverse strength.
    max_pairs_per_query:
        Maximum oriented pairs sampled per train query.
    random_state:
        Seed for pair subsampling when queries exceed the pair cap.

    Returns
    -------
    tuple[LinearSVC, numpy.ndarray, float, int]
        Fitted LinearSVC, linear coefficient vector, intercept, and pair count.
    """
    X_diff, y_pair, n_pairs = _pairwise_examples(
        X,
        y,
        groups,
        max_pairs_per_query=max_pairs_per_query,
        random_state=random_state,
    )
    model = LinearSVC(
        C=float(C),
        dual="auto",
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(X_diff, y_pair)
    coef = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    return model, coef, intercept, n_pairs


def score_linear(coef: np.ndarray, intercept: float, X: np.ndarray) -> np.ndarray:
    """Score rows with a linear ranker defined by coefficients and intercept.

    Used by pairwise RankSVM-lite paths that store weights on
    :class:`~buildml.ranking.results.RankerPlan` instead of a sklearn object.

    Parameters
    ----------
    coef:
        Linear weight vector of length ``n_features``.
    intercept:
        Scalar bias added to each dot product.
    X:
        Standardized feature matrix to score.

    Returns
    -------
    numpy.ndarray
        Linear scores, one per row.
    """
    if X.size == 0:
        return np.zeros(0, dtype=float)
    return np.asarray(X @ coef + intercept, dtype=float)
