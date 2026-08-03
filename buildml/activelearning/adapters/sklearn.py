"""Sklearn active-learning query scoring (core fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError


def score_sklearn_pool(
    *,
    strategy: str,
    x_pool: np.ndarray,
    estimator: Any,
    committee: Any | None = None,
) -> np.ndarray:
    """Score unlabeled pool rows with sklearn-native query strategies.

    Higher scores indicate higher priority for human labeling. Supports
    uncertainty-based strategies and query-by-committee vote entropy.

    Parameters
    ----------
    strategy:
        Query strategy name (``least_confidence``, ``margin``, ``entropy``,
        ``committee``, or ``expected_model_change_lite``).
    x_pool:
        Feature matrix for unlabeled pool rows.
    estimator:
        Fitted sklearn-compatible classifier with ``predict_proba`` when required.
    committee:
        Optional fitted :class:`~sklearn.ensemble.BaggingClassifier` for
        ``committee`` strategy.

    Returns
    -------
    np.ndarray
        One non-negative score per pool row, same length as ``x_pool``.

    Raises
    ------
    ValidationError
        When the strategy is unsupported or required estimators are missing.
    """
    if strategy in {
        "least_confidence",
        "margin",
        "entropy",
        "expected_model_change_lite",
    }:
        if not hasattr(estimator, "predict_proba"):
            raise ValidationError(
                f"Strategy {strategy!r} requires predict_proba on the base estimator."
            )
        proba = np.asarray(estimator.predict_proba(x_pool), dtype=float)
        proba = np.clip(proba, 1e-12, 1.0)
        if strategy == "least_confidence":
            return 1.0 - proba.max(axis=1)
        if strategy == "margin":
            part = np.partition(proba, -2, axis=1)
            top2 = part[:, -2:]
            margin = top2.max(axis=1) - top2.min(axis=1)
            return -margin
        if strategy == "entropy":
            return -np.sum(proba * np.log(proba), axis=1)
        conf = proba.max(axis=1)
        norms = np.linalg.norm(x_pool, axis=1)
        return norms * (1.0 - conf)

    if strategy == "committee":
        if committee is None:
            raise ValidationError(
                "Committee strategy requires a fitted committee. "
                "Call fit_active_learner(strategy='committee')."
            )
        member_preds = []
        estimators = getattr(committee, "estimators_", None)
        if not estimators:
            raise ValidationError("Committee has no fitted estimators_.")
        for est in estimators:
            member_preds.append(np.asarray(est.predict(x_pool)))
        votes = np.vstack(member_preds)
        n_members = votes.shape[0]
        scores = np.zeros(votes.shape[1], dtype=float)
        for j in range(votes.shape[1]):
            _, counts = np.unique(votes[:, j], return_counts=True)
            p = counts.astype(float) / float(n_members)
            p = np.clip(p, 1e-12, 1.0)
            scores[j] = float(-np.sum(p * np.log(p)))
        return scores

    raise ValidationError(
        f"Unsupported sklearn active-learning strategy {strategy!r}. "
        f"Supported: least_confidence, margin, entropy, committee, "
        "expected_model_change_lite."
    )
