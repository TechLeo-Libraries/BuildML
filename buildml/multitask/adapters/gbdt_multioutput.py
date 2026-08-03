"""Industry GBDT multi-target multi-task adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.multioutput import MultiOutputClassifier

from buildml.core.errors import ValidationError
from buildml.multitask.extras import require_catboost, require_lightgbm, require_xgboost

IndustryMultiTaskMethod = Literal[
    "multi_output_xgb",
    "multi_output_lgbm",
    "multi_output_catboost",
]


@dataclass
class GBDTMultiTargetEstimator:
    """Honest GBDT multi-target estimator (native reg / MultiOutput cls)."""

    method: IndustryMultiTaskMethod = "multi_output_xgb"
    task: str = "classification"
    random_state: int | None = 0
    estimator_: Any = field(default=None, repr=False)

    def fit(self, x: np.ndarray, y: np.ndarray) -> GBDTMultiTargetEstimator:
        """Fit the GBDT multi-target estimator on design matrix ``x`` and targets ``y``.

        Builds the backend-specific estimator via :meth:`_build_estimator`, then
        fits on train-only data passed from :func:`fit_multitask`.

        Parameters
        ----------
        x:
            Float feature matrix of shape ``(n_samples, n_features)``.
        y:
            Target matrix of shape ``(n_samples, n_tasks)``.

        Returns
        -------
        GBDTMultiTargetEstimator
            Fitted estimator with ``estimator_`` populated (``self``).
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if self.task == "regression":
            y_fit = y_arr.astype(float)
        else:
            y_fit = y_arr.astype(int)
        estimator = self._build_estimator(n_outputs=int(y_fit.shape[1]))
        estimator.fit(x_arr, y_fit)
        self.estimator_ = estimator
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict multi-target outputs for rows in ``x``.

        Delegates to the fitted GBDT backend and rounds classification outputs
        to integer label codes.

        Parameters
        ----------
        x:
            Float feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(n_samples, n_tasks)``; classification outputs
            are rounded to integer codes.

        Raises
        ------
        ValidationError
            When the estimator has not been fitted.
        """
        if self.estimator_ is None:
            raise ValidationError("GBDTMultiTargetEstimator is not fitted.")
        x_arr = np.asarray(x, dtype=float)
        preds = np.asarray(self.estimator_.predict(x_arr))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if self.task == "classification":
            return np.rint(preds).astype(int)
        return preds.astype(float)

    def _build_estimator(self, *, n_outputs: int) -> Any:
        rs = self.random_state
        if self.method == "multi_output_xgb":
            xgb = require_xgboost()
            if self.task == "regression":
                return xgb.XGBRegressor(
                    n_estimators=160,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=rs,
                    tree_method="hist",
                    multi_strategy="multi_output_tree",
                )
            base = xgb.XGBClassifier(
                n_estimators=160,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=rs,
                eval_metric="mlogloss",
                tree_method="hist",
            )
            return MultiOutputClassifier(base)

        if self.method == "multi_output_lgbm":
            lgb = require_lightgbm()
            if self.task == "regression":
                return lgb.LGBMRegressor(
                    n_estimators=160,
                    learning_rate=0.08,
                    num_leaves=31,
                    random_state=rs,
                    verbose=-1,
                )
            base = lgb.LGBMClassifier(
                n_estimators=160,
                learning_rate=0.08,
                num_leaves=31,
                random_state=rs,
                verbose=-1,
            )
            return MultiOutputClassifier(base)

        if self.method == "multi_output_catboost":
            cb = require_catboost()
            if self.task == "regression":
                loss = "MultiRMSE" if n_outputs > 1 else "RMSE"
                return cb.CatBoostRegressor(
                    iterations=160,
                    depth=6,
                    learning_rate=0.08,
                    loss_function=loss,
                    random_seed=rs,
                    verbose=False,
                )
            base = cb.CatBoostClassifier(
                iterations=160,
                depth=6,
                learning_rate=0.08,
                random_seed=rs,
                verbose=False,
            )
            return MultiOutputClassifier(base)

        raise ValidationError(
            f"Unsupported industry multi-task method '{self.method}'"
        )


def build_gbdt_estimator(
    *,
    method: IndustryMultiTaskMethod,
    task: str,
    random_state: int | None,
) -> GBDTMultiTargetEstimator:
    """Construct a GBDT multi-target estimator for one industry method.

    Validates that ``task`` is same-type (classification or regression) before
    returning the adapter used by :func:`fit_multitask`.

    Parameters
    ----------
    method:
        Industry method: ``multi_output_xgb``, ``multi_output_lgbm``, or
        ``multi_output_catboost``.
    task:
        ``classification`` or ``regression`` (mixed targets are not supported).
    random_state:
        Seed passed to the underlying GBDT implementation.

    Returns
    -------
    GBDTMultiTargetEstimator
        Unfitted estimator ready for :meth:`GBDTMultiTargetEstimator.fit`.

    Raises
    ------
    ValidationError
        When ``task`` is mixed or the method is unsupported.
    """
    if task not in {"classification", "regression"}:
        raise ValidationError(
            "Industry GBDT multi-task supports same-type targets only. "
            "Use backend='torch', method='shared_trunk_multihead' for mixed "
            "classification+regression."
        )
    return GBDTMultiTargetEstimator(
        method=method,
        task=task,
        random_state=random_state,
    )
