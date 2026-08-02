"""Fit Bayesian / probabilistic estimators (train-only; optional conformal)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.probabilistic.adapters.mapie import (
    fit_mapie,
    mapie_supports_predict_proba,
    mapie_supports_return_std,
)
from buildml.probabilistic.adapters.native import (
    build_native_estimator,
    fit_native_conformal,
    native_supports_predict_proba,
    native_supports_return_std,
)
from buildml.probabilistic.adapters.ngboost import build_ngboost_estimator
from buildml.probabilistic.catalog import resolve_backend_estimator
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
    ProbabilisticBackend,
    ProbabilisticConfig,
    ProbabilisticEstimator,
    ProbabilisticTask,
)

_RETURN_STD = {"bayesian_ridge", "gaussian_process_regressor"}
_CLASSIFIERS = {"gaussian_process_classifier", "gaussian_nb"}
_MAPIE_METHODS = {"split", "cv_plus", "jackknife_plus"}


def fit_probabilistic(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: ProbabilisticBackend | None = None,
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
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    reduce_plan: Any | None = None,
) -> tuple[ProbabilisticPlan, ProbabilisticFitResult]:
    """Fit a probabilistic / Bayesian-leaning estimator on Session train.

    Backends
    --------
    native (default):
        sklearn BayesianRidge / GaussianProcess* / GaussianNB + in-tree split
        conformal carved from train only.
    mapie (``buildml[probabilistic-industry]``):
        MAPIE conformal regression/classification — split, CV+, jackknife+.
    ngboost (``buildml[probabilistic-industry]``):
        NGBoost predictive distributions with optional in-tree conformal overlay.

    Honesty: uncertainty quantification for tabular estimators — not PyMC/Stan
    MCMC or Bayesian deep nets. Classical ``Session.calibration()`` unchanged.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    if not 0.0 < float(alpha) < 1.0:
        raise ValidationError(f"alpha must be in (0, 1); got {alpha}.")

    est_key = str(estimator).lower().replace("-", "_")
    resolved_backend, resolved_estimator, resolved_task = resolve_backend_estimator(
        backend=backend,
        estimator=est_key,
        task=task,
    )
    resolved_interval = _resolve_interval_method(
        resolved_backend,
        resolved_estimator,
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
    use_mapie = resolved_backend == "mapie"
    if use_mapie:
        use_conformal = False  # MAPIE owns conformal calibration

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
    if use_conformal or (use_mapie and resolved_estimator == "split"):
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
        if conformal and resolved_backend == "native" and resolved_task == "classification":
            if resolved_estimator not in _CLASSIFIERS:
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

    mapie_method: str | None = None
    supports_std = False
    supports_proba = False
    conformal_q: float | None = None

    if resolved_backend == "native":
        estimator_obj = build_native_estimator(
            resolved_estimator,
            random_state=random_state,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        try:
            estimator_obj.fit(x_fit, y_fit)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"Probabilistic fit failed for estimator={resolved_estimator!r}: {exc}"
            ) from exc

        if use_conformal and calib_indices:
            calib_frame = full.loc[calib_indices]
            x_cal = matrix_from_frame(calib_frame, cols)
            if resolved_task == "regression":
                y_cal = regression_targets(calib_frame[target])
            else:
                y_cal, _, _ = encode_classification_targets(
                    calib_frame[target],
                    classes=classes_tuple,
                )
            conformal_q = fit_native_conformal(
                estimator_obj,
                task=resolved_task,
                x_cal=x_cal,
                y_cal=y_cal,
                alpha=alpha,
                classes=classes_tuple,
            )
            disclosures.append(
                f"Native split-conformal quantile q̂={conformal_q:.6g} at alpha={alpha}."
            )

        supports_std = native_supports_return_std(resolved_estimator)
        supports_proba = native_supports_predict_proba(estimator_obj)
        disclosures.extend(_native_disclosures(resolved_estimator, supports_std, supports_proba))

    elif resolved_backend == "mapie":
        mapie_method = resolved_estimator
        x_cal = y_cal_arr = None
        if resolved_estimator == "split" and calib_indices:
            calib_frame = full.loc[calib_indices]
            x_cal = matrix_from_frame(calib_frame, cols)
            if resolved_task == "regression":
                y_cal_arr = regression_targets(calib_frame[target])
            else:
                y_cal_arr, _, _ = encode_classification_targets(
                    calib_frame[target],
                    classes=classes_tuple,
                )
        elif resolved_estimator == "split":
            raise ValidationError(
                "MAPIE split conformal requires a train-only calibration carve."
            )

        estimator_obj, mapie_disclosures = fit_mapie(
            method=resolved_estimator,  # type: ignore[arg-type]
            task=resolved_task,
            x_fit=x_fit,
            y_fit=y_fit,
            x_cal=x_cal,
            y_cal=y_cal_arr,
            random_state=random_state,
            alpha=float(alpha),
        )
        disclosures.extend(mapie_disclosures)
        supports_std = mapie_supports_return_std()
        supports_proba = mapie_supports_predict_proba(estimator_obj)
        resolved_interval = _mapie_interval_method(resolved_estimator)

    elif resolved_backend == "ngboost":
        estimator_obj = build_ngboost_estimator(
            resolved_estimator,
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        try:
            estimator_obj.fit(x_fit, y_fit)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"NGBoost fit failed for estimator={resolved_estimator!r}: {exc}"
            ) from exc

        if use_conformal and calib_indices:
            calib_frame = full.loc[calib_indices]
            x_cal = matrix_from_frame(calib_frame, cols)
            if resolved_task == "regression":
                y_cal = regression_targets(calib_frame[target])
                from buildml.probabilistic.adapters.ngboost import ngboost_predict_std

                mean, _ = ngboost_predict_std(estimator_obj, x_cal)
                from buildml.probabilistic.conformal import (
                    absolute_residual_scores,
                    conformal_quantile,
                )

                scores = absolute_residual_scores(y_cal, mean)
                conformal_q = conformal_quantile(scores, alpha)
            else:
                y_cal, _, _ = encode_classification_targets(
                    calib_frame[target],
                    classes=classes_tuple,
                )
                conformal_q = fit_native_conformal(
                    estimator_obj,
                    task=resolved_task,
                    x_cal=x_cal,
                    y_cal=y_cal,
                    alpha=alpha,
                    classes=classes_tuple,
                )
            disclosures.append(
                f"NGBoost + in-tree conformal overlay q̂={conformal_q:.6g} at alpha={alpha}."
            )

        supports_std = resolved_task == "regression"
        supports_proba = hasattr(estimator_obj, "predict_proba")
        disclosures.extend(
            [
                "NGBoost backend: natural-gradient boosting predictive distributions.",
                "Fit uses Session train only; validation/test are evaluation only.",
                f"interval_method={resolved_interval}; alpha={alpha}.",
            ]
        )
    else:
        raise ValidationError(f"Unknown backend={resolved_backend!r}.")

    disclosures.extend(
        [
            "Fit uses the Session train partition only "
            "(plus an optional train-only conformal carve).",
            "Validation/test are evaluation / interval scoring only.",
            f"backend={resolved_backend}, interval_method={resolved_interval}, alpha={alpha}.",
        ]
    )

    config = ProbabilisticConfig(
        backend=resolved_backend,
        estimator=resolved_estimator,  # type: ignore[arg-type]
        task=resolved_task,
        columns=tuple(cols),
        random_state=random_state,
        alpha=float(alpha),
        conformal=bool(conformal) if resolved_backend != "mapie" else True,
        conformal_calibration_fraction=float(conformal_calibration_fraction),
        interval_method=resolved_interval,
        prefer_reduce_components=prefer_reduce_components,
        n_restarts_optimizer=int(n_restarts_optimizer),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        mapie_method=mapie_method,  # type: ignore[arg-type]
    )
    plan = ProbabilisticPlan(
        backend=resolved_backend,
        estimator_name=resolved_estimator,
        task=resolved_task,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        n_fit_rows=len(fit_indices),
        n_conformal_calib_rows=len(calib_indices),
        alpha=float(alpha),
        conformal=bool(use_conformal or use_mapie),
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
        backend=resolved_backend,
        estimator_name=resolved_estimator,
        task=resolved_task,
        n_train_rows=n_train,
        n_fit_rows=len(fit_indices),
        n_conformal_calib_rows=len(calib_indices),
        columns=tuple(cols),
        target_column=target,
        alpha=float(alpha),
        conformal=bool(use_conformal or use_mapie),
        interval_method=resolved_interval,
        classes=classes_tuple,
        conformal_quantile=conformal_q,
        mapie_method=mapie_method,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _native_disclosures(
    estimator: str,
    supports_std: bool,
    supports_proba: bool,
) -> list[str]:
    out = [
        "Native backend: sklearn BayesianRidge / GaussianProcess* / GaussianNB "
        "+ optional in-tree split conformal — not PyMC/Stan MCMC.",
    ]
    if supports_std:
        out.append(
            "Estimator supports predict(..., return_std=True) for posterior "
            "predictive std under the model's Gaussian assumptions."
        )
    if supports_proba:
        out.append(
            "Estimator supports predict_proba; evaluate reports NLL/Brier. "
            "Classical Session.calibration() remains available for classical "
            "fit(...) classifiers and is not overwritten by this path."
        )
    return out


def _mapie_interval_method(method: str) -> IntervalMethod:
    if method == "split":
        return "mapie"
    if method == "cv_plus":
        return "mapie_cv_plus"
    if method == "jackknife_plus":
        return "mapie_jackknife_plus"
    return "mapie"


def _resolve_interval_method(
    backend: str,
    estimator: str,
    task: ProbabilisticTask,
    *,
    conformal: bool,
    interval_method: IntervalMethod | None,
) -> IntervalMethod:
    if backend == "mapie":
        if interval_method is not None and interval_method not in {
            "mapie",
            "mapie_cv_plus",
            "mapie_jackknife_plus",
            "split_conformal",
            "none",
        }:
            pass  # fall through to mapie default
        return _mapie_interval_method(estimator)

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
    return method
