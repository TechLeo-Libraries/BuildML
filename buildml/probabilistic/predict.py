"""Point predictions and predictive intervals for probabilistic plans."""

from __future__ import annotations

from typing import Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.probabilistic.adapters.mapie import mapie_predict_interval, mapie_predict_sets
from buildml.probabilistic.adapters.ngboost import ngboost_predict_std
from buildml.probabilistic.conformal import (
    classification_prediction_sets,
    regression_intervals,
)
from buildml.probabilistic.features import (
    decode_predictions,
    matrix_from_frame,
    norm_ppf,
)
from buildml.probabilistic.results import (
    ProbabilisticIntervalResult,
    ProbabilisticPlan,
    ProbabilisticPredictResult,
)

PartitionOrAll = PartitionName | Literal["all"]


def predict_probabilistic(
    dataset: Dataset,
    plan: ProbabilisticPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    return_std: bool = True,
    return_proba: bool = True,
) -> ProbabilisticPredictResult:
    """Predict with the fitted probabilistic estimator (no refit / no leakage)."""
    if plan is None:
        raise ValidationError("No ProbabilisticPlan. Call fit_probabilistic first.")

    frame, part_name = _resolve_frame(dataset, split_plan, partition)
    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"Missing feature columns for predict_probabilistic: {missing}"
        )

    x = matrix_from_frame(frame, list(plan.columns))
    disclosures = [
        "predict_probabilistic does not update the probabilistic plan.",
        f"Predictions from backend={plan.backend}, estimator={plan.estimator_name}.",
    ]
    warnings: list[str] = []
    std_out: tuple[float, ...] | None = None
    proba_out: tuple[tuple[float, ...], ...] | None = None

    if plan.backend == "mapie":
        if plan.task == "regression":
            point, _, _, _ = mapie_predict_interval(
                plan.estimator_, x, task="regression", alpha=plan.alpha
            )
            preds = point
        else:
            point, _ = mapie_predict_sets(
                plan.estimator_, x, alpha=plan.alpha, task="classification"
            )
            preds = tuple(decode_predictions(np.asarray(point), plan.label_encoder_))
        if return_proba and plan.supports_predict_proba:
            from buildml.probabilistic.adapters.mapie import MapieWrapper

            handle = (
                plan.estimator_
                if isinstance(plan.estimator_, MapieWrapper)
                else plan.estimator_
            )
            est = getattr(handle, "estimator", handle)
            base = getattr(est, "estimator", est)
            if hasattr(base, "predict_proba"):
                proba = np.asarray(base.predict_proba(x), dtype=float)
                proba_out = tuple(tuple(float(v) for v in row) for row in proba)
    elif plan.task == "regression":
        if plan.backend == "ngboost" and return_std:
            mean, std = ngboost_predict_std(plan.estimator_, x)
            preds = tuple(float(v) for v in mean)
            std_out = tuple(float(v) for v in std)
        elif return_std and plan.supports_return_std:
            mean, std = plan.estimator_.predict(x, return_std=True)
            preds = tuple(float(v) for v in mean)
            std_out = tuple(float(v) for v in std)
        else:
            raw = plan.estimator_.predict(x)
            preds = tuple(float(v) for v in raw)
            if return_std and not plan.supports_return_std:
                warnings.append(
                    "return_std requested but estimator does not support it."
                )
    else:
        raw = plan.estimator_.predict(x)
        preds = tuple(decode_predictions(raw, plan.label_encoder_))
        if return_proba and plan.supports_predict_proba:
            proba = np.asarray(plan.estimator_.predict_proba(x), dtype=float)
            proba_out = tuple(tuple(float(v) for v in row) for row in proba)
        elif return_proba and not plan.supports_predict_proba:
            warnings.append(
                "return_proba requested but estimator has no predict_proba."
            )

    return ProbabilisticPredictResult(
        partition=part_name,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=int(len(frame)),
        predictions=preds,
        std=std_out,
        probabilities=proba_out,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def predict_interval(
    dataset: Dataset,
    plan: ProbabilisticPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    alpha: float | None = None,
    method: str | None = None,
) -> ProbabilisticIntervalResult:
    """Build predictive intervals (regression) or prediction sets (classification).

    Methods
    -------
    ``posterior_std``
        Gaussian intervals from ``return_std`` (BayesianRidge / GPR / NGBoost).
    ``split_conformal``
        Distribution-free intervals/sets using the train-only conformal quantile.
    ``both``
        Prefer conformal bounds when available; still return posterior std.
    ``mapie`` / ``mapie_cv_plus`` / ``mapie_jackknife_plus``
        MAPIE-backed intervals/sets from the fitted wrapper.
    """
    if plan is None:
        raise ValidationError("No ProbabilisticPlan. Call fit_probabilistic first.")

    resolved_alpha = float(plan.alpha if alpha is None else alpha)
    if not 0.0 < resolved_alpha < 1.0:
        raise ValidationError(f"alpha must be in (0, 1); got {resolved_alpha}.")

    resolved_method = str(method or plan.interval_method)
    frame, part_name = _resolve_frame(dataset, split_plan, partition)
    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Missing feature columns for predict_interval: {missing}")

    x = matrix_from_frame(frame, list(plan.columns))
    disclosures = [
        f"predict_interval method={resolved_method}, alpha={resolved_alpha}.",
        "Intervals/sets do not use holdout rows for calibration "
        "(conformal quantile was fit on a train carve, if enabled).",
    ]
    warnings: list[str] = []

    if plan.backend == "mapie":
        return _mapie_intervals(
            plan,
            x,
            part_name=part_name,
            alpha=resolved_alpha,
            disclosures=disclosures,
            warnings=warnings,
        )

    if plan.task == "regression":
        return _regression_intervals(
            plan,
            x,
            part_name=part_name,
            alpha=resolved_alpha,
            method=resolved_method,
            disclosures=disclosures,
            warnings=warnings,
        )

    return _classification_sets(
        plan,
        x,
        part_name=part_name,
        alpha=resolved_alpha,
        method=resolved_method,
        disclosures=disclosures,
        warnings=warnings,
    )


def _mapie_intervals(
    plan: ProbabilisticPlan,
    x: np.ndarray,
    *,
    part_name: str,
    alpha: float,
    disclosures: list[str],
    warnings: list[str],
) -> ProbabilisticIntervalResult:
    if plan.task == "regression":
        point, lower, upper, used = mapie_predict_interval(
            plan.estimator_, x, task="regression", alpha=alpha
        )
        disclosures.append(f"MAPIE regression intervals via method={plan.estimator_name}.")
        return ProbabilisticIntervalResult(
            partition=part_name,
            estimator_name=plan.estimator_name,
            task=plan.task,
            n_rows=len(point),
            alpha=alpha,
            method=used,
            lower=lower,
            upper=upper,
            point=point,
            std=None,
            prediction_sets=None,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    point_raw, sets = mapie_predict_sets(
        plan.estimator_, x, alpha=alpha, task="classification"
    )
    point = tuple(decode_predictions(np.asarray(point_raw), plan.label_encoder_))
    disclosures.append(f"MAPIE classification prediction sets method={plan.estimator_name}.")
    return ProbabilisticIntervalResult(
        partition=part_name,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=len(point),
        alpha=alpha,
        method="mapie",
        lower=None,
        upper=None,
        point=point,
        std=None,
        prediction_sets=sets,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _regression_intervals(
    plan: ProbabilisticPlan,
    x: np.ndarray,
    *,
    part_name: str,
    alpha: float,
    method: str,
    disclosures: list[str],
    warnings: list[str],
) -> ProbabilisticIntervalResult:
    std_out: tuple[float, ...] | None = None
    if plan.backend == "ngboost" and plan.supports_return_std:
        mean, std = ngboost_predict_std(plan.estimator_, x)
        point = tuple(float(v) for v in mean)
        std_out = tuple(float(v) for v in std)
    elif plan.supports_return_std:
        mean, std = plan.estimator_.predict(x, return_std=True)
        point = tuple(float(v) for v in mean)
        std_out = tuple(float(v) for v in std)
    else:
        mean = np.asarray(plan.estimator_.predict(x), dtype=float)
        point = tuple(float(v) for v in mean)

    lower: tuple[float, ...] | None = None
    upper: tuple[float, ...] | None = None
    used = method

    want_conformal = method in {"split_conformal", "both"}
    want_std = method in {"posterior_std", "both"}

    if want_conformal:
        if plan.conformal_quantile_ is None:
            if method == "split_conformal":
                raise ValidationError(
                    "No conformal quantile on plan. Refit with conformal=True."
                )
            warnings.append(
                "interval_method includes conformal but no quantile is stored; "
                "falling back to posterior_std when available."
            )
            want_conformal = False
            used = "posterior_std"
        else:
            lo, hi = regression_intervals(np.asarray(point), plan.conformal_quantile_)
            if abs(alpha - plan.alpha) > 1e-12:
                warnings.append(
                    f"Requested alpha={alpha} differs from plan.alpha={plan.alpha}; "
                    "using the stored conformal quantile (re-fit to change alpha)."
                )
            lower = tuple(float(v) for v in lo)
            upper = tuple(float(v) for v in hi)
            disclosures.append(
                f"Split-conformal half-width q̂={plan.conformal_quantile_:.6g}."
            )

    if want_std and std_out is not None:
        z = norm_ppf(1.0 - alpha / 2.0)
        std_lo = tuple(p - z * s for p, s in zip(point, std_out, strict=True))
        std_hi = tuple(p + z * s for p, s in zip(point, std_out, strict=True))
        disclosures.append(
            f"Posterior-std Gaussian intervals use z={z:.4g} for alpha={alpha}."
        )
        if lower is None or upper is None:
            lower, upper = std_lo, std_hi
            used = "posterior_std"
        else:
            used = "both"
            disclosures.append(
                "Primary lower/upper are split-conformal; std is the "
                "model posterior predictive standard deviation."
            )
    elif want_std and std_out is None:
        warnings.append(
            "posterior_std requested but estimator lacks return_std support."
        )

    if lower is None or upper is None:
        raise ValidationError(
            f"Could not build regression intervals with method={method!r}. "
            "Enable conformal=True and/or use bayesian_ridge / "
            "gaussian_process_regressor / ngboost_regressor."
        )

    return ProbabilisticIntervalResult(
        partition=part_name,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=len(point),
        alpha=alpha,
        method=used,
        lower=lower,
        upper=upper,
        point=point,
        std=std_out,
        prediction_sets=None,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _classification_sets(
    plan: ProbabilisticPlan,
    x: np.ndarray,
    *,
    part_name: str,
    alpha: float,
    method: str,
    disclosures: list[str],
    warnings: list[str],
) -> ProbabilisticIntervalResult:
    if not plan.supports_predict_proba:
        raise ValidationError(
            "Classification prediction sets require predict_proba."
        )
    if plan.conformal_quantile_ is None:
        raise ValidationError(
            "No conformal quantile on plan. Refit with conformal=True "
            "for classification prediction sets."
        )
    if method not in {"split_conformal", "both", "none"} and method != plan.interval_method:
        warnings.append(
            f"Classification intervals use split_conformal sets; "
            f"requested method={method!r} treated as split_conformal."
        )
    if abs(alpha - plan.alpha) > 1e-12:
        warnings.append(
            f"Requested alpha={alpha} differs from plan.alpha={plan.alpha}; "
            "using the stored conformal quantile (re-fit to change alpha)."
        )

    proba = np.asarray(plan.estimator_.predict_proba(x), dtype=float)
    raw = plan.estimator_.predict(x)
    point = tuple(decode_predictions(raw, plan.label_encoder_))
    classes = plan.classes_ or ()
    sets = classification_prediction_sets(proba, plan.conformal_quantile_, classes)
    disclosures.append(
        f"Split-conformal prediction sets use q̂={plan.conformal_quantile_:.6g}."
    )
    return ProbabilisticIntervalResult(
        partition=part_name,
        estimator_name=plan.estimator_name,
        task=plan.task,
        n_rows=len(point),
        alpha=alpha,
        method="split_conformal",
        lower=None,
        upper=None,
        point=point,
        std=None,
        prediction_sets=tuple(sets),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _resolve_frame(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
):
    if partition == "all":
        return dataset._ensure_pandas(), "all"
    if split_plan is None:
        raise ValidationError(
            f"partition='{partition}' requires a SplitPlan. Call session.split(...)."
        )
    return frame_for_partition(dataset, split_plan, partition), str(partition)
