"""Native industry active-learning query scoring (no torch coupling).

Used as the honest fallback for ``buildml[activelearning-industry]`` and when
scikit-activeml cannot be imported cleanly on a host.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError


def score_industry_native_pool(
    *,
    strategy: str,
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    x_pool: np.ndarray,
    estimator: Any,
    committee: Any | None = None,
) -> np.ndarray:
    """Score pool rows with CoreSet / QBC industry query strategies.

    Uses native numpy/sklearn scoring — no torch coupling. CoreSet ranks pool
    points by distance to the labeled set; QBC strategies measure committee
    disagreement.

    Parameters
    ----------
    strategy:
        Industry strategy (``core_set``, ``qbc_kl``, or ``qbc_variation_ratios``).
    x_labeled:
        Feature matrix for currently labeled train rows.
    y_labeled:
        Encoded labels for labeled rows (unused by CoreSet scoring).
    x_pool:
        Feature matrix for unlabeled pool rows.
    estimator:
        Fitted base estimator (reserved for future hybrid strategies).
    committee:
        Fitted bagging committee required for QBC strategies.

    Returns
    -------
    np.ndarray
        One score per pool row; higher means higher labeling priority.

    Raises
    ------
    ValidationError
        When the strategy is unsupported or QBC is requested without a committee.
    """
    x_pool = np.asarray(x_pool, dtype=float)
    if x_pool.shape[0] == 0:
        return np.empty(0, dtype=float)

    if strategy == "core_set":
        return _core_set_scores(x_labeled, x_pool)

    if strategy in {"qbc_kl", "qbc_variation_ratios"}:
        if committee is None:
            raise ValidationError(
                f"Strategy {strategy!r} requires a fitted committee ensemble."
            )
        if strategy == "qbc_kl":
            return _qbc_kl_scores(committee, x_pool)
        return _qbc_variation_scores(committee, x_pool)

    raise ValidationError(
        f"Unsupported native industry strategy {strategy!r}. "
        "Supported: core_set, qbc_kl, qbc_variation_ratios."
    )


def _core_set_scores(x_labeled: np.ndarray, x_pool: np.ndarray) -> np.ndarray:
    """k-center greedy distances — higher score = farther from labeled set."""
    x_labeled = np.asarray(x_labeled, dtype=float)
    x_pool = np.asarray(x_pool, dtype=float)
    if x_labeled.shape[0] == 0:
        norms = np.linalg.norm(x_pool, axis=1)
        return norms / (norms.max() + 1e-12)
    # Min distance from each pool point to any labeled point.
    dists = np.linalg.norm(x_pool[:, None, :] - x_labeled[None, :, :], axis=2)
    return dists.min(axis=1)


def _member_probas(committee: Any, x_pool: np.ndarray) -> np.ndarray:
    estimators = getattr(committee, "estimators_", None)
    if not estimators:
        raise ValidationError("Committee has no fitted estimators_.")
    probas = []
    for est in estimators:
        if not hasattr(est, "predict_proba"):
            raise ValidationError("QBC industry strategies require predict_proba members.")
        probas.append(np.asarray(est.predict_proba(x_pool), dtype=float))
    return np.stack(probas, axis=0)


def _qbc_kl_scores(committee: Any, x_pool: np.ndarray) -> np.ndarray:
    """Mean pairwise KL divergence among committee member distributions."""
    stack = _member_probas(committee, x_pool)
    stack = np.clip(stack, 1e-12, 1.0)
    n_members = stack.shape[0]
    if n_members < 2:
        return np.zeros(stack.shape[1], dtype=float)
    kl_sum = np.zeros(stack.shape[1], dtype=float)
    pairs = 0
    for i in range(n_members):
        for j in range(i + 1, n_members):
            p = stack[i]
            q = stack[j]
            kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
            kl_sum += kl
            pairs += 1
    return kl_sum / float(pairs)


def _qbc_variation_scores(committee: Any, x_pool: np.ndarray) -> np.ndarray:
    """Variation ratio: 1 - max vote fraction (higher = more disagreement)."""
    estimators = getattr(committee, "estimators_", None)
    if not estimators:
        raise ValidationError("Committee has no fitted estimators_.")
    votes = np.vstack([np.asarray(est.predict(x_pool)) for est in estimators])
    n_members = votes.shape[0]
    scores = np.zeros(votes.shape[1], dtype=float)
    for j in range(votes.shape[1]):
        _, counts = np.unique(votes[:, j], return_counts=True)
        scores[j] = 1.0 - float(counts.max()) / float(n_members)
    return scores
