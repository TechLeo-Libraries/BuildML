"""Fit Bayesian / probabilistic estimators (train-only; optional conformal)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.probabilistic.conformal import (
    absolute_residual_scores,
    classification_nonconformity,
    conformal_quantile,
)
from buildml.probabilistic.features import (
    encode_classification_targets,
    matrix_from_frame,
    regression_targets,
    resolve_probabilistic_columns,
    split_train_for_conformal,
    train_partition_frame,
)
from buildml.probabilistic.results import ProbabilisticFitResult, ProbabilisticPlan
from buildml.probabilistic.types import (
    IntervalMethod,
    ProbabilisticConfig,
    ProbabilisticEstimator,
    ProbabilisticTask,
)

_REGRESSORS = {"bayesian_ridge", "gaussian_process_regressor"}
_CLASSIFIERS = {"gaussian_process_classifier", "gaussian_nb"}
_RETURN_STD = {"bayesian_ridge", "gaussian_process_regressor"}


def fit_probabilistic(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    estimator: ProbabilisticEstimator = "bayesian_ridge",
    task: ProbabilisticTask | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    alpha: float = 0.1,
    conformal: bool = True,
    conformal_calibration_fraction: float = 0.2,
    interval_method: IntervalMethod | None = None,
    prefer_reduce_components: bool = True,
    n_restarts_optimizer: int = 0,
    reduce_plan: Any | None = None,
) -> tuple[ProbabilisticPlan, ProbabilisticFitResult]:
    """Fit a probabilistic / Bayesian-leaning estimator on Session train.

    Honesty
    -------
    Uses sklearn ``BayesianRidge``, ``GaussianProcess*``, or ``GaussianNB``.
    Optional split conformal calibrates intervals/sets on a **train-only**
    carve — never Session validation/test. This is uncertainty quantification
    for tabular estimators, not a PyMC/Stan MCMC platform or Bayesian deep net.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    if not 0.0 < float(alpha) < 1.0:
        raise ValidationError(f"alpha must be in (0, 1); got {alpha}.")

    est_key = str(estimator).lower().replace("-", "_")
    resolved_task = _resolve_task(est_key, task)
    resolved_interval = _resolve_interval_method(
        est_key,
        resolved_task,
        conformal=conformal,
        interval_method=interval_method,
    )

    target = dataset.require_target()
    train = train_partition_frame(dataset, split_plan)
    cols, used_reduce, disclosures = resolve_probabilistic_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    n_train = int(len(split_plan.train_indices))
    full = dataset._ensure_pandas()
    warnings: list[str] = []

    use_conformal = bool(conformal) and resolved_interval in {
        "split_conformal",
        "both",
    }
    # Discover class vocabulary early (labels only) for stratified carve.
    class_vocab: tuple[Any, ...] | None = None
    if resolved_task == "classification":
        train_y = train[target]
        if train_y.isna().any():
            raise ValidationError(
                "Probabilistic classification requires non-null train targets."
            )
        class_vocab = tuple(sorted(train_y.astype(str).unique().tolist()))
        if len(class_vocab) < 2:
            raise ValidationError(
                "Probabilistic classification requires at least 2 classes."
            )
        disclosures.append(
            "Classifier classes_ discovered from the full train target "
            "vocabulary (labels only) before the optional conformal carve."
        )

    fit_indices: list[Any]
    calib_indices: list[Any]
    if use_conformal:
        stratify = None
        if resolved_task == "classification":
            stratify = train.loc[list(split_plan.train_indices), target].astype(str)
        fit_indices, calib_indices = split_train_for_conformal(
            split_plan.train_indices,
            calibration_fraction=conformal_calibration_fraction,
            random_state=random_state,
            stratify_labels=None if stratify is None else list(stratify),
        )
        disclosures.append(
            f"Split conformal calibration carved from train only: "
            f"n_fit={len(fit_indices)}, n_calib={len(calib_indices)} "
            f"(fraction≈{conformal_calibration_fraction}). "
            "Validation/test were never used for conformal calibration."
        )
    else:
        fit_indices = list(split_plan.train_indices)
        calib_indices = []
        if conformal and resolved_task == "classification" and est_key not in _CLASSIFIERS:
            warnings.append(
                "conformal=True ignored for this estimator/task combination."
            )

    fit_frame = full.loc[fit_indices]
    x_fit = matrix_from_frame(fit_frame, cols)

    label_encoder = None
    classes_tuple: tuple[Any, ...] | None = None
    y_fit: np.ndarray
    if resolved_task == "classification":
        y_fit, label_encoder, classes_tuple = encode_classification_targets(
            fit_frame[target],
            classes=class_vocab,
        )
    else:
        y_fit = regression_targets(fit_frame[target])

    estimator_obj = _build_estimator(
        est_key,
        random_state=random_state,
        n_restarts_optimizer=n_restarts_optimizer,
    )
    try:
        estimator_obj.fit(x_fit, y_fit)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Probabilistic fit failed for estimator={est_key!r}: {exc}"
        ) from exc

    conformal_q: float | None = None
    if use_conformal and calib_indices:
        calib_frame = full.loc[calib_indices]
        x_cal = matrix_from_frame(calib_frame, cols)
        if resolved_task == "regression":
            y_cal = regression_targets(calib_frame[target])
            pred_cal = np.asarray(estimator_obj.predict(x_cal), dtype=float)
            scores = absolute_residual_scores(y_cal, pred_cal)
            conformal_q = conformal_quantile(scores, alpha)
            disclosures.append(
                f"Regression split-conformal quantile q̂={conformal_q:.6g} "
                f"at alpha={alpha} (target coverage≈{1 - alpha:.0%})."
            )
        else:
            if not hasattr(estimator_obj, "predict_proba"):
                raise ValidationError(
                    "Classification conformal requires predict_proba."
                )
            y_cal, _, _ = encode_classification_targets(
                calib_frame[target],
                classes=classes_tuple,
            )
            proba = np.asarray(estimator_obj.predict_proba(x_cal), dtype=float)
            scores = classification_nonconformity(proba, y_cal)
            conformal_q = conformal_quantile(scores, alpha)
            disclosures.append(
                f"Classification split-conformal nonconformity quantile "
                f"q̂={conformal_q:.6g} at alpha={alpha}."
            )

    supports_std = est_key in _RETURN_STD
    supports_proba = hasattr(estimator_obj, "predict_proba")
    disclosures.extend(
        [
            "Probabilistic path uses sklearn BayesianRidge / GaussianProcess* / "
            "GaussianNB — not PyMC/Stan MCMC or Bayesian deep nets.",
            "Fit uses the Session train partition only "
            "(plus an optional train-only conformal carve).",
            "Validation/test are evaluation / interval scoring only.",
            f"interval_method={resolved_interval}; alpha={alpha}.",
        ]
    )
    if supports_std:
        disclosures.append(
            "Estimator supports predict(..., return_std=True) for posterior "
            "predictive std under the model’s Gaussian assumptions."
        )
    if supports_proba:
        disclosures.append(
            "Estimator supports predict_proba; evaluate reports NLL/Brier. "
            "Classical Session.calibration() remains available for classical "
            "fit(...) classifiers and is not overwritten by this path."
        )

    config = ProbabilisticConfig(
        estimator=est_key,  # type: ignore[arg-type]
        task=resolved_task,
        columns=tuple(cols),
        random_state=random_state,
        alpha=float(alpha),
        conformal=bool(conformal),
        conformal_calibration_fraction=float(conformal_calibration_fraction),
        interval_method=resolved_interval,
        prefer_reduce_components=prefer_reduce_components,
        n_restarts_optimizer=int(n_restarts_optimizer),
    )
    plan = ProbabilisticPlan(
        estimator_name=est_key,
        task=resolved_task,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        n_fit_rows=len(fit_indices),
        n_conformal_calib_rows=len(calib_indices),
        alpha=float(alpha),
        conformal=bool(use_conformal),
        interval_method=resolved_interval,
        classes_=classes_tuple,
        estimator_=estimator_obj,
        label_encoder_=label_encoder,
        conformal_quantile_=conformal_q,
        conformal_fit_indices_=tuple(fit_indices),
        conformal_calib_indices_=tuple(calib_indices),
        supports_return_std=supports_std,
        supports_predict_proba=supports_proba,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = ProbabilisticFitResult(
        estimator_name=est_key,
        task=resolved_task,
        n_train_rows=n_train,
        n_fit_rows=len(fit_indices),
        n_conformal_calib_rows=len(calib_indices),
        columns=tuple(cols),
        target_column=target,
        alpha=float(alpha),
        conformal=bool(use_conformal),
        interval_method=resolved_interval,
        classes=classes_tuple,
        conformal_quantile=conformal_q,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _resolve_task(estimator: str, task: ProbabilisticTask | None) -> ProbabilisticTask:
    if estimator in _CLASSIFIERS:
        if task == "regression":
            raise ValidationError(
                f"Estimator {estimator!r} is a classifier; task cannot be "
                "'regression'."
            )
        return "classification"
    if estimator in _REGRESSORS:
        if task == "classification":
            raise ValidationError(
                f"Estimator {estimator!r} is a regressor; task cannot be "
                "'classification'."
            )
        return "regression"
    raise ValidationError(
        f"Unknown probabilistic estimator={estimator!r}. "
        f"Supported: {sorted(_CLASSIFIERS | _REGRESSORS)}"
    )


def _resolve_interval_method(
    estimator: str,
    task: ProbabilisticTask,
    *,
    conformal: bool,
    interval_method: IntervalMethod | None,
) -> IntervalMethod:
    if interval_method is not None:
        method = interval_method
    elif task == "regression":
        if conformal and estimator in _RETURN_STD:
            method = "both"
        elif conformal:
            method = "split_conformal"
        elif estimator in _RETURN_STD:
            method = "posterior_std"
        else:
            method = "none"
    else:
        method = "split_conformal" if conformal else "none"

    if method == "posterior_std" and estimator not in _RETURN_STD:
        raise ValidationError(
            f"interval_method='posterior_std' requires bayesian_ridge or "
            f"gaussian_process_regressor; got {estimator!r}."
        )
    if method == "both" and estimator not in _RETURN_STD:
        raise ValidationError(
            "interval_method='both' requires an estimator with return_std."
        )
    if method in {"split_conformal", "both"} and not conformal and interval_method is not None:
        # Explicit method requesting conformal without conformal=True — enable.
        pass
    return method


def _build_estimator(
    name: str,
    *,
    random_state: int | None,
    n_restarts_optimizer: int,
) -> Any:
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
    raise ValidationError(f"Unsupported probabilistic estimator '{name}'")
