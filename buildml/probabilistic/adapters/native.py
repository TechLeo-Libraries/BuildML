"""Native sklearn Bayesian / GP / NB adapter with in-tree split conformal."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB

from buildml.core.errors import ValidationError
from buildml.probabilistic.conformal import (
    absolute_residual_scores,
    classification_nonconformity,
    conformal_quantile,
)

_RETURN_STD = {"bayesian_ridge", "gaussian_process_regressor"}
_CLASSIFIERS = {"gaussian_process_classifier", "gaussian_nb"}


def build_native_estimator(
    name: str,
    *,
    random_state: int | None,
    n_restarts_optimizer: int,
) -> Any:
    """Construct a native sklearn probabilistic estimator by catalog name.

    Dispatches BayesianRidge, GaussianProcess*, and GaussianNB keys from
    :func:`buildml.probabilistic.catalog.list_probabilistic_estimators`.

    Parameters
    ----------
    name:
        Native estimator key such as ``bayesian_ridge``.
    random_state:
        Seed for GP estimators.
    n_restarts_optimizer:
        GP kernel restart count.

    Returns
    -------
    sklearn estimator
        Unfitted native probabilistic model.

    Raises
    ------
    ValidationError
        When ``name`` is not a supported native estimator.
    """
    if name == "bayesian_ridge":
        return BayesianRidge()
    if name == "gaussian_process_regressor":
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        return GaussianProcessRegressor(
            kernel=kernel,
            random_state=random_state,
            n_restarts_optimizer=int(n_restarts_optimizer),
            normalize_y=True,
        )
    if name == "gaussian_process_classifier":
        kernel = RBF(length_scale=1.0)
        return GaussianProcessClassifier(
            kernel=kernel,
            random_state=random_state,
            n_restarts_optimizer=int(n_restarts_optimizer),
        )
    if name == "gaussian_nb":
        return GaussianNB()
    raise ValidationError(f"Unsupported native probabilistic estimator '{name}'")


def native_supports_return_std(estimator_name: str) -> bool:
    """Return whether the native estimator exposes posterior standard deviation.

    Catalog keys BayesianRidge and GaussianProcessRegressor support std output
    used by posterior-std interval paths.

    Parameters
    ----------
    estimator_name:
        Catalog estimator key.

    Returns
    -------
    bool
        ``True`` for BayesianRidge and GaussianProcessRegressor paths.
    """
    return estimator_name in _RETURN_STD


def native_supports_predict_proba(estimator_obj: Any) -> bool:
    """Return whether the fitted estimator supports ``predict_proba``.

    Classification conformal and evaluate paths require probability outputs
    from native GaussianProcessClassifier and GaussianNB models.

    Parameters
    ----------
    estimator_obj:
        Fitted sklearn estimator.

    Returns
    -------
    bool
        ``True`` when ``predict_proba`` is available on the object.
    """
    return hasattr(estimator_obj, "predict_proba")


def fit_native_conformal(
    estimator_obj: Any,
    *,
    task: str,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float,
    classes: tuple[Any, ...] | None = None,
) -> float:
    """Calibrate in-tree split conformal quantile on train-only carve rows.

    Computes nonconformity scores on the calibration carve and returns the
    finite-sample quantile used by predict and evaluate interval paths.

    Parameters
    ----------
    estimator_obj:
        Fitted native estimator (already trained on fit carve).
    task:
        ``regression`` or ``classification``.
    x_cal, y_cal:
        Calibration carve features and targets from Session train only.
    alpha:
        Miscoverage rate in ``(0, 1)``.
    classes:
        Unused placeholder kept for API symmetry with classification paths.

    Returns
    -------
    float
        Split-conformal quantile applied to interval construction.

    Raises
    ------
    ValidationError
        When classification conformal is requested but ``predict_proba`` is missing.
    """
    if task == "regression":
        pred_cal = np.asarray(estimator_obj.predict(x_cal), dtype=float)
        scores = absolute_residual_scores(y_cal, pred_cal)
        return conformal_quantile(scores, alpha)

    if not hasattr(estimator_obj, "predict_proba"):
        raise ValidationError("Classification conformal requires predict_proba.")
    proba = np.asarray(estimator_obj.predict_proba(x_cal), dtype=float)
    scores = classification_nonconformity(proba, y_cal)
    return conformal_quantile(scores, alpha)
