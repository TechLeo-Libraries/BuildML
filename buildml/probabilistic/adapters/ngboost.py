"""NGBoost natural gradient boosting with predictive distributions."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.probabilistic.extras import require_ngboost


def build_ngboost_estimator(
    name: str,
    *,
    random_state: int | None = 0,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
) -> Any:
    """Construct an NGBRegressor or NGBClassifier for the industry backend.

    Requires ``buildml[probabilistic-industry]`` and returns an unfitted NGBoost
    model configured for Normal or Bernoulli predictive distributions.

    Parameters
    ----------
    name:
        ``ngboost_regressor`` or ``ngboost_classifier``.
    random_state:
        Seed forwarded to NGBoost.
    n_estimators:
        Boosting rounds.
    learning_rate:
        NGBoost learning rate.

    Returns
    -------
    NGBRegressor or NGBClassifier
        Unfitted distribution-boosting estimator.

    Raises
    ------
    ValidationError
        When ``name`` is not a supported NGBoost key.
    MissingExtraError
        When ngboost is not installed.
    """
    require_ngboost(feature=f"NGBoost {name}")
    if name == "ngboost_regressor":
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from ngboost.learners import default_tree_learner

        return NGBRegressor(
            Dist=Normal,
            Base=default_tree_learner,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            random_state=random_state,
            verbose=False,
        )
    if name == "ngboost_classifier":
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        from ngboost.learners import default_tree_learner

        return NGBClassifier(
            Dist=Bernoulli,
            Base=default_tree_learner,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            random_state=random_state,
            verbose=False,
        )
    raise ValidationError(f"Unsupported NGBoost estimator '{name}'")


def ngboost_predict_std(estimator_obj: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return predictive mean and standard deviation from an NGBRegressor.

    Reads the Normal predictive distribution via ``pred_dist`` for regression
    interval and CRPS evaluation paths.

    Parameters
    ----------
    estimator_obj:
        Fitted ``NGBRegressor``.
    x:
        Feature matrix.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Mean and standard deviation arrays aligned with ``x`` rows.
    """
    dist = estimator_obj.pred_dist(x)
    mean = np.asarray(dist.loc, dtype=float)
    scale = getattr(dist, "scale", None)
    if scale is None:
        std = np.sqrt(np.maximum(np.asarray(dist.var(), dtype=float), 1e-12))
    else:
        std = np.asarray(scale, dtype=float)
    return mean, std


def ngboost_predict_proba(estimator_obj: Any, x: np.ndarray) -> np.ndarray:
    """Return Bernoulli predictive probabilities from an NGBClassifier.

    Used by classification evaluate and conformal paths on the NGBoost backend.

    Parameters
    ----------
    estimator_obj:
        Fitted ``NGBClassifier``.
    x:
        Feature matrix.

    Returns
    -------
    numpy.ndarray
        Class probability matrix from ``predict_proba``.

    Raises
    ------
    ValidationError
        When the estimator lacks ``predict_proba``.
    """
    if not hasattr(estimator_obj, "predict_proba"):
        raise ValidationError("NGBClassifier lacks predict_proba.")
    return np.asarray(estimator_obj.predict_proba(x), dtype=float)


def ngboost_crps_regression(estimator_obj: Any, x: np.ndarray, y: np.ndarray) -> float:
    """Compute average CRPS for a Normal NGBoost predictive distribution.

    Used by :func:`buildml.probabilistic.evaluate.evaluate_probabilistic` when
    scoring regression holdouts with distribution-aware metrics.

    Parameters
    ----------
    estimator_obj:
        Fitted ``NGBRegressor``.
    x:
        Holdout features.
    y:
        Holdout targets aligned with ``x``.

    Returns
    -------
    float
        Mean continuous ranked probability score across rows.
    """
    dist = estimator_obj.pred_dist(x)
    yy = np.asarray(y, dtype=float)
    try:
        crps_vals = dist.crps(yy)
        return float(np.mean(crps_vals))
    except Exception:  # noqa: BLE001
        mean, std = ngboost_predict_std(estimator_obj, x)
        return float(_gaussian_crps(yy, mean, std))


def _gaussian_crps(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian predictive (regression fallback)."""
    from scipy.stats import norm

    s = np.maximum(np.asarray(std, dtype=float), 1e-12)
    m = np.asarray(mean, dtype=float)
    yy = np.asarray(y, dtype=float)
    z = (yy - m) / s
    crps = s * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))
