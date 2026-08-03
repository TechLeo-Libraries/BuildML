"""Industry active-learning query scoring with optional scikit-activeml host path.

Native CoreSet/QBC scoring is always available. When ``skactiveml`` is on the
import path, this adapter attempts the real ``GreedySamplingX`` /
``QueryByCommittee`` APIs. Any import or API failure falls back to native
scoring and **always** attaches a disclosure string so callers never confuse
the host path with silent numpy scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.activelearning.adapters.industry_native import score_industry_native_pool
from buildml.activelearning.extras import (
    scikit_activeml_importable,
    scikit_activeml_spec_present,
)

_FALLBACK_DISCLOSURE = (
    "Industry query scoring uses native CoreSet/QBC numpy paths. "
    "scikit-activeml was present or attempted but could not be used on this host: "
    "native fallback is active (disclosed)."
)

_SKACTIVEML_SUCCESS_DISCLOSURE = (
    "Industry query scoring used the scikit-activeml host path "
    "(GreedySamplingX / QueryByCommittee)."
)

_NATIVE_DEFAULT_DISCLOSURE = (
    "Industry query scoring uses native CoreSet/QBC numpy paths "
    "(scikit-activeml not installed; native path is the documented default)."
)


def score_industry_pool(
    *,
    strategy: str,
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    x_pool: np.ndarray,
    estimator: Any,
    committee: Any | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Score pool rows with industry CoreSet / QBC strategies.

    Attempts scikit-activeml when the package is on ``sys.path`` and imports
    cleanly. On any failure, scores with the native numpy/sklearn scorer and
    returns an explicit fallback disclosure.

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
    tuple[np.ndarray, list[str]]
        ``(scores, disclosures)`` — one score per pool row (higher = higher
        labeling priority) plus honesty notes for the query result.
    """
    disclosures: list[str] = []
    # Subprocess probe first: in-process import can hard-crash on broken
    # torch/skorch hosts (Windows access violation), which try/except cannot catch.
    if scikit_activeml_spec_present() and scikit_activeml_importable():
        try:
            scores = _score_with_skactiveml(
                strategy=strategy,
                x_labeled=x_labeled,
                y_labeled=y_labeled,
                x_pool=x_pool,
                estimator=estimator,
                committee=committee,
            )
            disclosures.append(_SKACTIVEML_SUCCESS_DISCLOSURE)
            return scores, disclosures
        except Exception as exc:  # noqa: BLE001: residual API mismatches fall back
            disclosures.append(
                f"{_FALLBACK_DISCLOSURE} Reason: {type(exc).__name__}: {exc}"
            )
    elif scikit_activeml_spec_present():
        disclosures.append(
            f"{_FALLBACK_DISCLOSURE} Reason: subprocess import probe failed "
            "(broken skactiveml/torch/skorch stack on this host)."
        )
    else:
        disclosures.append(_NATIVE_DEFAULT_DISCLOSURE)

    scores = score_industry_native_pool(
        strategy=strategy,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        x_pool=x_pool,
        estimator=estimator,
        committee=committee,
    )
    return scores, disclosures


def _score_with_skactiveml(
    *,
    strategy: str,
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    x_pool: np.ndarray,
    estimator: Any,
    committee: Any | None = None,
) -> np.ndarray:
    """Use scikit-activeml query strategies; raise on failure (caller discloses)."""
    del estimator  # reserved for future hybrid strategies
    # Import private modules to avoid skactiveml.pool package __init__ side imports
    # when possible; still may pull skorch via skactiveml.base.
    from skactiveml.pool._greedy_sampling import GreedySamplingX
    from skactiveml.pool._query_by_committee import QueryByCommittee
    from skactiveml.utils import MISSING_LABEL

    x_pool = np.asarray(x_pool, dtype=float)
    x_labeled = np.asarray(x_labeled, dtype=float)
    y_labeled = np.asarray(y_labeled)
    n_pool = int(x_pool.shape[0])
    if n_pool == 0:
        return np.empty(0, dtype=float)

    n_labeled = int(x_labeled.shape[0])
    if n_labeled > 0:
        x_all = np.vstack([x_labeled, x_pool])
        y_all = np.concatenate(
            [y_labeled.astype(object), np.full(n_pool, MISSING_LABEL, dtype=object)]
        )
        candidates = np.arange(n_labeled, n_labeled + n_pool)
    else:
        x_all = x_pool
        y_all = np.full(n_pool, MISSING_LABEL, dtype=object)
        candidates = np.arange(n_pool)

    if strategy == "core_set":
        sampler = GreedySamplingX(metric="euclidean", random_state=0)
        result = sampler.query(
            x_all,
            y_all,
            candidates=candidates,
            batch_size=n_pool,
            return_utilities=True,
        )
        return _utilities_to_scores(result, n_pool)

    if strategy in {"qbc_kl", "qbc_variation_ratios"}:
        if committee is None:
            raise ValueError("QBC strategies require a fitted committee.")
        method = "KL_divergence" if strategy == "qbc_kl" else "variation_ratios"
        qbc = QueryByCommittee(method=method, random_state=0)
        result = qbc.query(
            x_all,
            y_all,
            ensemble=committee,
            fit_ensemble=False,
            candidates=candidates,
            batch_size=n_pool,
            return_utilities=True,
        )
        return _utilities_to_scores(result, n_pool)

    raise ValueError(
        f"Unsupported skactiveml industry strategy {strategy!r}. "
        "Supported: core_set, qbc_kl, qbc_variation_ratios."
    )


def _utilities_to_scores(result: Any, n_pool: int) -> np.ndarray:
    """Extract per-candidate utilities from a skactiveml ``query`` return value."""
    if not (isinstance(result, tuple) and len(result) == 2):
        raise TypeError(
            "scikit-activeml query did not return (indices, utilities); "
            f"got {type(result)!r}."
        )
    _, utilities = result
    arr = np.asarray(utilities, dtype=float)
    if arr.ndim == 2:
        # Typically (batch_size, n_candidates); first row ranks the full pool.
        arr = arr[0]
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.shape[0] != n_pool:
        raise ValueError(
            f"scikit-activeml utilities length {arr.shape[0]} != pool size {n_pool}."
        )
    # Replace non-finite utilities with a low priority so ranking stays defined.
    if not np.all(np.isfinite(arr)):
        finite = arr[np.isfinite(arr)]
        fill = float(finite.min() - 1.0) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr


__all__ = [
    "score_industry_pool",
    "_FALLBACK_DISCLOSURE",
    "_SKACTIVEML_SUCCESS_DISCLOSURE",
    "_NATIVE_DEFAULT_DISCLOSURE",
]
