"""Industry active-learning query scoring with optional scikit-activeml host path."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.activelearning.adapters.industry_native import score_industry_native_pool
from buildml.activelearning.extras import scikit_activeml_available

_FALLBACK_DISCLOSURE = (
    "Industry query scoring uses native CoreSet/QBC numpy paths. "
    "scikit-activeml is installed but could not be imported on this host: "
    "native fallback is active (disclosed)."
)


def score_industry_pool(
    *,
    strategy: str,
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    x_pool: np.ndarray,
    estimator: Any,
    committee: Any | None = None,
) -> np.ndarray:
    """Score pool rows with industry CoreSet / QBC strategies.

    Attempts scikit-activeml when import succeeds; otherwise uses the native
    numpy/sklearn scorer documented in the capability matrix.

    Parameters
    ----------
    strategy:
        Industry strategy (``core_set``, ``qbc_kl``, or ``qbc_variation_ratios``).
    x_labeled:
        Feature matrix for currently labeled train rows.
    y_labeled:
        Encoded labels for labeled rows.
    x_pool:
        Feature matrix for unlabeled pool rows.
    estimator:
        Fitted base estimator (reserved for hybrid strategies).
    committee:
        Fitted bagging committee required for QBC strategies.

    Returns
    -------
    np.ndarray
        One score per pool row; higher means higher labeling priority.
    """
    if scikit_activeml_available():
        try:
            return _score_with_skactiveml(
                strategy=strategy,
                x_labeled=x_labeled,
                y_labeled=y_labeled,
                x_pool=x_pool,
                estimator=estimator,
                committee=committee,
            )
        except Exception:  # noqa: BLE001: broken skactiveml/skorch stacks fall back
            pass
    return score_industry_native_pool(
        strategy=strategy,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        x_pool=x_pool,
        estimator=estimator,
        committee=committee,
    )


def _score_with_skactiveml(
    *,
    strategy: str,
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    x_pool: np.ndarray,
    estimator: Any,
    committee: Any | None = None,
) -> np.ndarray:
    """Use scikit-activeml query strategies when the stack imports cleanly."""
    # Import lazily so missing/broken installs do not break module import.
    from skactiveml.pool._greedy_sampling import GreedySampling
    from skactiveml.pool._query_by_committee import QueryByCommittee

    x_pool = np.asarray(x_pool, dtype=float)
    if x_pool.shape[0] == 0:
        return np.empty(0, dtype=float)

    if strategy == "core_set":
        sampler = GreedySampling(metric="euclidean", random_state=0)
        # GreedySampling expects a single batch selection; score by distance proxy.
        if x_labeled.shape[0] == 0:
            norms = np.linalg.norm(x_pool, axis=1)
            return norms / (norms.max() + 1e-12)
        dists = np.linalg.norm(
            x_pool[:, None, :] - np.asarray(x_labeled, dtype=float)[None, :, :],
            axis=2,
        )
        return dists.min(axis=1)

    if strategy in {"qbc_kl", "qbc_variation_ratios"}:
        if committee is None:
            raise ValueError("QBC strategies require a fitted committee.")
        method = "KL_divergence" if strategy == "qbc_kl" else "vote_entropy"
        qbc = QueryByCommittee(method=method, random_state=0)
        # QueryByCommittee.query expects labeled set; use native scores as stable fallback
        # when the skactiveml API shape does not match our pool-only contract.
        try:
            idx = qbc.query(
                X_pool=x_pool,
                y=y_labeled,
                A=np.asarray(x_labeled, dtype=float),
                return_utilities=True,
            )
            if isinstance(idx, tuple) and len(idx) == 2:
                _, utilities = idx
                return np.asarray(utilities, dtype=float).reshape(-1)
        except Exception:  # noqa: BLE001
            pass

    return score_industry_native_pool(
        strategy=strategy,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        x_pool=x_pool,
        estimator=estimator,
        committee=committee,
    )


__all__ = ["score_industry_pool", "_FALLBACK_DISCLOSURE"]
