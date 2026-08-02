"""Holdout blending estimators (meta-learner fit on an inner train holdout).

Leakage contract
----------------
Callers must pass **train-partition rows only**. The inner holdout is carved
from those rows; Session test / validation never enter meta-learner fitting.
After meta fit, base estimators are optionally refit on the full train matrix
for deployment (disclosed).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


def _as_estimator_list(estimators: list[tuple[str, Any]] | dict[str, Any]) -> list[tuple[str, Any]]:
    if isinstance(estimators, dict):
        return [(str(name), est) for name, est in estimators.items()]
    return [(str(name), est) for name, est in estimators]


def _meta_features_from_estimators(
    fitted: list[tuple[str, Any]],
    X: Any,
    *,
    task: Literal["classification", "regression"],
    blend_method: Literal["predict", "predict_proba"],
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for _, est in fitted:
        if (
            task == "classification"
            and blend_method == "predict_proba"
            and hasattr(est, "predict_proba")
        ):
            proba = np.asarray(est.predict_proba(X), dtype=float)
            if proba.ndim == 1:
                blocks.append(proba.reshape(-1, 1))
            elif proba.shape[1] == 2:
                # Binary: keep positive-class column to reduce collinearity.
                blocks.append(proba[:, 1:].reshape(-1, 1))
            else:
                blocks.append(proba)
        else:
            pred = np.asarray(est.predict(X), dtype=float).reshape(-1, 1)
            blocks.append(pred)
    return np.hstack(blocks)


def _fit_one(est: Any, X: Any, y: Any, sample_weight: Any | None) -> Any:
    model = clone(est)
    if sample_weight is not None:
        try:
            model.fit(X, y, sample_weight=sample_weight)
            return model
        except TypeError:
            pass
    model.fit(X, y)
    return model


def _split_rows(
    X: Any,
    y: Any,
    sample_weight: Any | None,
    *,
    holdout_fraction: float,
    random_state: int | None,
    stratify: Any | None,
) -> tuple[Any, Any, Any, Any, Any | None, Any | None]:
    """Split row-aligned design matrices while preserving pandas containers."""
    n = len(np.asarray(y))
    idx = np.arange(n)
    idx_base, idx_meta = train_test_split(
        idx,
        test_size=float(holdout_fraction),
        random_state=random_state,
        stratify=stratify,
    )
    if hasattr(X, "iloc"):
        x_base, x_meta = X.iloc[idx_base], X.iloc[idx_meta]
    else:
        X_arr = np.asarray(X)
        x_base, x_meta = X_arr[idx_base], X_arr[idx_meta]
    if hasattr(y, "iloc"):
        y_base, y_meta = y.iloc[idx_base], y.iloc[idx_meta]
    else:
        y_arr = np.asarray(y)
        y_base, y_meta = y_arr[idx_base], y_arr[idx_meta]
    sw_base = sw_meta = None
    if sample_weight is not None:
        if hasattr(sample_weight, "iloc"):
            sw_base, sw_meta = sample_weight.iloc[idx_base], sample_weight.iloc[idx_meta]
        else:
            sw = np.asarray(sample_weight)
            sw_base, sw_meta = sw[idx_base], sw[idx_meta]
    return x_base, x_meta, y_base, y_meta, sw_base, sw_meta


class HoldoutBlendClassifier(ClassifierMixin, BaseEstimator):
    """Classifier blending with an inner train holdout for the meta-learner."""

    def __init__(
        self,
        estimators: list[tuple[str, Any]] | dict[str, Any] | None = None,
        final_estimator: Any | None = None,
        *,
        holdout_fraction: float = 0.2,
        blend_method: Literal["predict", "predict_proba"] = "predict_proba",
        random_state: int | None = 0,
        refit_bases_on_full_train: bool = True,
        passthrough: bool = False,
    ) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_fraction = holdout_fraction
        self.blend_method = blend_method
        self.random_state = random_state
        self.refit_bases_on_full_train = refit_bases_on_full_train
        self.passthrough = passthrough

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> HoldoutBlendClassifier:
        if not (0.05 <= float(self.holdout_fraction) < 0.5):
            raise ValueError("holdout_fraction must be in [0.05, 0.5).")
        named = _as_estimator_list(self.estimators or [])
        if len(named) < 2:
            raise ValueError("Blending requires at least two base estimators.")

        y_arr = np.asarray(y)
        stratify = y_arr if len(np.unique(y_arr)) > 1 else None
        x_base, x_meta, y_base, y_meta, sw_base, sw_meta = _split_rows(
            X,
            y,
            sample_weight,
            holdout_fraction=self.holdout_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )

        fitted_bases = [
            (name, _fit_one(est, x_base, y_base, sw_base)) for name, est in named
        ]

        meta_x = _meta_features_from_estimators(
            fitted_bases,
            x_meta,
            task="classification",
            blend_method=self.blend_method,
        )
        if self.passthrough:
            meta_x = np.hstack([meta_x, np.asarray(x_meta, dtype=float)])

        final_proto = (
            self.final_estimator
            if self.final_estimator is not None
            else LogisticRegression(max_iter=1000)
        )
        final = _fit_one(final_proto, meta_x, y_meta, sw_meta)

        if self.refit_bases_on_full_train:
            self.estimators_ = [
                (name, _fit_one(est, X, y, sample_weight)) for name, est in named
            ]
        else:
            self.estimators_ = fitted_bases

        self.final_estimator_ = final
        self.classes_ = getattr(final, "classes_", np.unique(y_arr))
        self.named_estimators_ = {name: est for name, est in self.estimators_}
        self.n_features_in_ = int(np.asarray(X).shape[1])
        self.blend_holdout_rows_ = int(len(np.asarray(y_meta)))
        self.blend_base_rows_ = int(len(np.asarray(y_base)))
        return self

    def _transform_meta(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        meta = _meta_features_from_estimators(
            list(self.estimators_),
            X,
            task="classification",
            blend_method=self.blend_method,
        )
        if self.passthrough:
            meta = np.hstack([meta, np.asarray(X, dtype=float)])
        return meta

    def predict(self, X: Any) -> np.ndarray:
        return self.final_estimator_.predict(self._transform_meta(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self.final_estimator_, "predict_proba"):
            raise AttributeError(
                f"{type(self.final_estimator_).__name__} does not implement predict_proba"
            )
        return self.final_estimator_.predict_proba(self._transform_meta(X))


class HoldoutBlendRegressor(RegressorMixin, BaseEstimator):
    """Regressor blending with an inner train holdout for the meta-learner."""

    def __init__(
        self,
        estimators: list[tuple[str, Any]] | dict[str, Any] | None = None,
        final_estimator: Any | None = None,
        *,
        holdout_fraction: float = 0.2,
        blend_method: Literal["predict", "predict_proba"] = "predict",
        random_state: int | None = 0,
        refit_bases_on_full_train: bool = True,
        passthrough: bool = False,
    ) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_fraction = holdout_fraction
        self.blend_method = blend_method
        self.random_state = random_state
        self.refit_bases_on_full_train = refit_bases_on_full_train
        self.passthrough = passthrough

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> HoldoutBlendRegressor:
        if not (0.05 <= float(self.holdout_fraction) < 0.5):
            raise ValueError("holdout_fraction must be in [0.05, 0.5).")
        named = _as_estimator_list(self.estimators or [])
        if len(named) < 2:
            raise ValueError("Blending requires at least two base estimators.")

        x_base, x_meta, y_base, y_meta, sw_base, sw_meta = _split_rows(
            X,
            y,
            sample_weight,
            holdout_fraction=self.holdout_fraction,
            random_state=self.random_state,
            stratify=None,
        )

        fitted_bases = [
            (name, _fit_one(est, x_base, y_base, sw_base)) for name, est in named
        ]

        meta_x = _meta_features_from_estimators(
            fitted_bases,
            x_meta,
            task="regression",
            blend_method="predict",
        )
        if self.passthrough:
            meta_x = np.hstack([meta_x, np.asarray(x_meta, dtype=float)])

        final_proto = self.final_estimator if self.final_estimator is not None else Ridge()
        final = _fit_one(final_proto, meta_x, y_meta, sw_meta)

        if self.refit_bases_on_full_train:
            self.estimators_ = [
                (name, _fit_one(est, X, y, sample_weight)) for name, est in named
            ]
        else:
            self.estimators_ = fitted_bases

        self.final_estimator_ = final
        self.named_estimators_ = {name: est for name, est in self.estimators_}
        self.n_features_in_ = int(np.asarray(X).shape[1])
        self.blend_holdout_rows_ = int(len(np.asarray(y_meta)))
        self.blend_base_rows_ = int(len(np.asarray(y_base)))
        return self

    def _transform_meta(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        meta = _meta_features_from_estimators(
            list(self.estimators_),
            X,
            task="regression",
            blend_method="predict",
        )
        if self.passthrough:
            meta = np.hstack([meta, np.asarray(X, dtype=float)])
        return meta

    def predict(self, X: Any) -> np.ndarray:
        return self.final_estimator_.predict(self._transform_meta(X))
