"""Blend several models by learning, on held-out rows, how much to trust each.

scikit-learn ships stacking but not blending, so these two estimators fill the
gap. They follow the standard estimator protocol, which means they work
anywhere a scikit-learn model does — inside a pipeline, a grid search, or a
cross-validation.

The idea in one paragraph: split the data you are given, fit the base models on
the larger part, have them predict the smaller part, and fit a meta-learner on
those predictions. Because the meta-learner only ever sees predictions the bases
made about rows they had not been fitted on, it learns which model to trust
where, rather than which model memorises hardest.

Leakage contract
----------------
Callers must pass **train-partition rows only**. The inner holdout is carved out
of whatever it receives, so handing it the full frame silently blends on test
rows and produces a model whose evaluation means nothing. Session validation and
test never enter meta-learner fitting when the estimator is built through
:mod:`buildml.ensemble.fit`.

After the meta-learner is fitted, the bases are optionally refit on everything
they were given — the standard deployment pattern, disclosed because it means
the deployed bases differ slightly from those the meta-learner saw.

See Also
--------
buildml.ensemble.fit.fit_blending_ensemble : The Session-safe way to build these.
sklearn.ensemble.StackingClassifier : Out-of-fold meta-features instead.
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
    """Blend classifiers by fitting a meta-learner on held-out predictions.

    A scikit-learn-compatible classifier, so it can go anywhere an estimator
    goes. Fitting splits the data once, fits the bases on the larger part, has
    them predict the smaller part, and fits the meta-learner on those
    predictions.

    Two details are worth knowing because they affect results. The inner split
    is stratified whenever the target has more than one class, which keeps rare
    classes present in both halves — without it, a class with twenty examples
    can end up entirely on one side. And for binary problems only the
    positive-class probability column is kept, since the two columns sum to one
    and feeding both gives the meta-learner perfectly collinear inputs.

    Attributes
    ----------
    estimators_:
        The fitted bases. On the full data when
        ``refit_bases_on_full_train``, otherwise the blend-train fits.
    final_estimator_:
        The fitted meta-learner.
    classes_:
        The class labels.
    named_estimators_:
        The bases by name, for inspecting one of them.
    n_features_in_:
        Input width, for scikit-learn compatibility.
    blend_holdout_rows_:
        How many rows the meta-learner was fitted on. **The number to check.**
        Below a few dozen, the meta-learner is fitting noise.
    blend_base_rows_:
        How many rows the bases were fitted on for the blend.

    Notes
    -----
    **Pass train rows only.** The holdout is carved from whatever arrives.

    **``blend_holdout_rows_`` is the health check.** A small holdout produces a
    meta-learner that varies wildly with ``random_state``; if changing the seed
    changes your conclusions, that is what happened.

    See Also
    --------
    HoldoutBlendRegressor : The regression counterpart.
    """

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
        """Record the configuration, validating nothing until ``fit``.

        Parameters are stored unmodified, as scikit-learn requires for ``clone``
        and ``get_params`` to work — which is what lets this estimator be used
        inside a grid search. Validation therefore happens in :meth:`fit`.

        Parameters
        ----------
        estimators:
            Base estimators as ``(name, estimator)`` pairs or a mapping. At
            least two are needed; the check is deferred to ``fit``.
        final_estimator:
            The meta-learner. Defaults to logistic regression, a deliberately
            simple choice for a handful of prediction columns.
        holdout_fraction:
            Share reserved for the meta-learner, between 0.05 and 0.5.
        blend_method:
            ``'predict_proba'`` to blend probabilities, ``'predict'`` to blend
            labels. Probabilities carry more information and are the default;
            a base without them falls back to labels for that base alone.
        random_state:
            Seed for the inner split.
        refit_bases_on_full_train:
            Refit the bases on everything after the meta-learner is fitted.
        passthrough:
            Append the original features to the meta-learner's input.
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_fraction = holdout_fraction
        self.blend_method = blend_method
        self.random_state = random_state
        self.refit_bases_on_full_train = refit_bases_on_full_train
        self.passthrough = passthrough

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> HoldoutBlendClassifier:
        """Split, fit the bases, fit the meta-learner, then optionally refit.

        Everything happens inside the rows handed in. The split is stratified
        when the target has more than one class.

        Parameters
        ----------
        X:
            Training features. **Train-partition rows only** — the holdout comes
            out of these.
        y:
            Training labels.
        sample_weight:
            Per-row weights, split alongside the data and passed to any
            estimator whose ``fit`` accepts them. Estimators that do not are
            fitted unweighted rather than failing, since a blend should not be
            impossible because one base cannot weight.

        Returns
        -------
        HoldoutBlendClassifier
            Self, fitted.

        Raises
        ------
        ValueError
            If ``holdout_fraction`` is outside ``[0.05, 0.5)``, or if fewer than
            two base estimators were given. The upper bound matters: a holdout
            at or above half would leave the bases with less data than the
            meta-learner, which inverts the point of the method.

        Notes
        -----
        **Check ``blend_holdout_rows_`` afterwards.** It is the number that says
        whether the meta-learner had enough to learn from.

        **Weights are applied where accepted and dropped where not**, silently
        per estimator. Inspect ``named_estimators_`` if it matters which.
        """
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
        """Run every base, then let the meta-learner decide from their outputs.

        Two passes over the data: once through each base, then once through the
        meta-learner on the stacked predictions.

        Parameters
        ----------
        X:
            Features with the same columns, in the same order, as at fit time.

        Returns
        -------
        numpy.ndarray
            The predicted labels.

        Raises
        ------
        NotFittedError
            If called before ``fit``.

        Notes
        -----
        **Inference costs the sum of all the bases plus the meta-learner.** A
        blend of four models is roughly four times the latency of one, which is
        the trade a blend makes.
        """
        return self.final_estimator_.predict(self._transform_meta(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities, if the meta-learner can produce them.

        The probabilities come from the meta-learner alone. The bases'
        probabilities were its input features, not its output, so these are one
        model's opinion informed by several rather than an average of several.

        Parameters
        ----------
        X:
            Features matching the fit-time contract.

        Returns
        -------
        numpy.ndarray
            Probabilities, one column per class in ``classes_`` order.

        Raises
        ------
        AttributeError
            If the meta-learner has no ``predict_proba``. Choose one that does
            — the default logistic regression is such a model.
        NotFittedError
            If called before ``fit``.

        Notes
        -----
        **These are not necessarily well calibrated.** The meta-learner was
        fitted on a small holdout, so its confidence can be optimistic; see
        :func:`buildml.model.diagnostics.calibration_report` before treating
        them as probabilities rather than scores.
        """
        if not hasattr(self.final_estimator_, "predict_proba"):
            raise AttributeError(
                f"{type(self.final_estimator_).__name__} does not implement predict_proba"
            )
        return self.final_estimator_.predict_proba(self._transform_meta(X))


class HoldoutBlendRegressor(RegressorMixin, BaseEstimator):
    """Blend regressors by fitting a meta-learner on held-out predictions.

    The regression counterpart, and simpler than the classifier for two reasons.
    There is nothing to stratify on, so the inner split is plain random. And
    there are no probabilities, so the meta-features are always the bases'
    predicted values — one column per base.

    With a ridge meta-learner, the fitted coefficients read as weights: how much
    each base contributes to the final answer. That is unusually interpretable
    for an ensemble, and worth looking at — a base with a coefficient near zero
    is contributing nothing but latency, and a negative coefficient means the
    meta-learner is using that model as a correction rather than a prediction.

    Attributes
    ----------
    estimators_:
        The fitted bases.
    final_estimator_:
        The fitted meta-learner. With the default ridge, its ``coef_`` is the
        per-base weighting.
    named_estimators_:
        The bases by name.
    n_features_in_:
        Input width.
    blend_holdout_rows_:
        Rows the meta-learner saw. The number to sanity-check.
    blend_base_rows_:
        Rows the bases were fitted on for the blend.

    Notes
    -----
    **Pass train rows only.** The holdout is carved from whatever arrives.

    **The inner split is not stratified**, and cannot be. On a small dataset
    with a skewed target, the holdout may not represent the range the model has
    to cover.

    See Also
    --------
    HoldoutBlendClassifier : The classification counterpart.
    """

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
        """Record the configuration, validating nothing until ``fit``.

        Parameters are stored unmodified so ``clone`` and ``get_params`` behave,
        which is what allows this estimator inside a search.

        Parameters
        ----------
        estimators:
            Base estimators as ``(name, estimator)`` pairs or a mapping. At
            least two.
        final_estimator:
            The meta-learner. Defaults to ridge, whose coefficients then read as
            per-base weights.
        holdout_fraction:
            Share reserved for the meta-learner, between 0.05 and 0.5.
        blend_method:
            Accepted for symmetry with the classifier and effectively fixed at
            ``'predict'`` — there are no probabilities in regression.
        random_state:
            Seed for the inner split.
        refit_bases_on_full_train:
            Refit the bases on everything after the meta-learner is fitted.
        passthrough:
            Append the original features to the meta-learner's input.
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_fraction = holdout_fraction
        self.blend_method = blend_method
        self.random_state = random_state
        self.refit_bases_on_full_train = refit_bases_on_full_train
        self.passthrough = passthrough

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> HoldoutBlendRegressor:
        """Split, fit the bases, fit the meta-learner, then optionally refit.

        The split is plain random; there is nothing to stratify on.

        Parameters
        ----------
        X:
            Training features. **Train-partition rows only.**
        y:
            Training targets.
        sample_weight:
            Per-row weights, split alongside the data and passed to estimators
            that accept them. Others are fitted unweighted.

        Returns
        -------
        HoldoutBlendRegressor
            Self, fitted.

        Raises
        ------
        ValueError
            If ``holdout_fraction`` is outside ``[0.05, 0.5)``, or fewer than
            two bases were given.

        Notes
        -----
        **Inspect ``final_estimator_.coef_`` afterwards** when the meta-learner
        is a ridge. It shows how much each base ended up contributing, and a
        near-zero coefficient is a model you can drop.
        """
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
        """Run every base, then combine their values through the meta-learner.

        Two passes: each base predicts, and the meta-learner turns that row of
        predictions into the final number.

        Parameters
        ----------
        X:
            Features with the same columns, in the same order, as at fit time.

        Returns
        -------
        numpy.ndarray
            The predicted values.

        Raises
        ------
        NotFittedError
            If called before ``fit``.

        Notes
        -----
        **Predictions are not bounded by the bases' range.** A ridge
        meta-learner with an intercept and non-convex coefficients can output
        values outside what any base predicted, which matters when the target
        has a physical floor — a count, a price, a duration. Clip afterwards if
        the domain requires it.
        """
        return self.final_estimator_.predict(self._transform_meta(X))
