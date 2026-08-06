"""Horizon generation and one-step prediction helpers for ForecastPlan."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.forecasting.backends import industry_one_step, industry_predict
from buildml.forecasting.features import lag_feature_row, ordered_frame, target_series
from buildml.forecasting.results import ForecastGenerateResult, ForecastPlan


def _predict_next(
    plan: ForecastPlan,
    history: list[float],
    *,
    step: int,
    exog_row: np.ndarray | None = None,
) -> float:
    """Predict one step ahead from history (past values only)."""
    method = plan.method
    if plan.industry_estimator_ is not None:
        if step > 1 and plan.method not in {"prophet", "nbeats"}:
            # Multi-step industry forecast in one call for statsmodels
            preds = industry_predict(
                plan.industry_estimator_,
                steps=step,
                history=history,
                exog_future=None if exog_row is None else exog_row.reshape(1, -1),
            )
            return float(preds[step - 1])
        preds = industry_predict(
            plan.industry_estimator_,
            steps=1,
            history=history,
            exog_future=None if exog_row is None else exog_row.reshape(1, -1),
        )
        return float(preds[0])
    if method == "naive":
        assert plan.baseline_value_ is not None
        return float(plan.baseline_value_)
    if method == "mean":
        assert plan.baseline_value_ is not None
        return float(plan.baseline_value_)
    if method == "drift":
        assert plan.baseline_value_ is not None and plan.drift_slope_ is not None
        return float(plan.baseline_value_ + plan.drift_slope_ * step)
    if method == "seasonal_naive":
        hist = plan.seasonal_history_
        if not hist:
            raise ValidationError("seasonal_naive plan is missing seasonal_history_")
        # step is 1-indexed from origin
        return float(hist[(step - 1) % len(hist)])
    if method in {"lag_ridge", "lag_hgb"}:
        if plan.estimator_ is None:
            raise ValidationError(f"{method} plan is missing a fitted estimator")
        if plan.exog_columns and exog_row is None:
            raise ValidationError(
                "This ForecastPlan uses exogenous columns; provide future exog "
                "rows for horizon generation (or evaluate with rolling_one_step "
                "where holdout exog is available)."
            )
        feats = lag_feature_row(
            np.asarray(history, dtype=float),
            plan.lags,
            exog_row=exog_row,
        )
        return float(plan.estimator_.predict(feats.reshape(1, -1))[0])
    raise ValidationError(f"Unsupported forecast method '{method}'")


def generate_forecast(
    plan: ForecastPlan,
    *,
    horizon: int | None = None,
    history: list[float] | tuple[float, ...] | np.ndarray | None = None,
    future_exog: np.ndarray | pd.DataFrame | None = None,
    origin: str = "train_end",
) -> ForecastGenerateResult:
    """Generate an H-step forecast from a frozen plan without refit.

    Supports baseline, lag-tabular, and industry backends with optional future
    exogenous rows and recursive multi-step composition for lag models.

    Parameters
    ----------
    plan:
        Train-fitted :class:`ForecastPlan` to generate from.
    horizon:
        Steps ahead. Defaults to ``plan.horizon``.
    history:
        Optional target history ending at the forecast origin. Defaults to
        ``plan.last_train_values_`` (train-end origin).
    future_exog:
        Required when the plan has exogenous columns: shape ``(horizon, n_exog)``.
    origin:
        Label recorded in the result (e.g. ``train_end``, ``validation_end``).

    Returns
    -------
    ForecastGenerateResult
        Horizon predictions with method, origin, and disclosure strings.

    Raises
    ------
    ValidationError
        When horizon or history is invalid, exogenous inputs are missing or
        malformed, or the plan method is unsupported.
    """
    h = int(plan.horizon if horizon is None else horizon)
    if h < 1:
        raise ValidationError("horizon must be >= 1")
    hist = list(
        float(v)
        for v in (
            plan.last_train_values_
            if history is None
            else np.asarray(history, dtype=float).reshape(-1).tolist()
        )
    )
    if not hist:
        raise ValidationError("Forecast history is empty")

    warnings: list[str] = []

    if plan.industry_estimator_ is not None:
        exog_mat_ind = None
        if plan.exog_columns:
            if future_exog is None:
                raise ValidationError(
                    "Industry exog plan requires future_exog for horizon generation."
                )
            exog_mat_ind = np.asarray(
                future_exog if not isinstance(future_exog, pd.DataFrame)
                else future_exog.loc[:, list(plan.exog_columns)].to_numpy(),
                dtype=float,
            )
            if exog_mat_ind.ndim == 1:
                exog_mat_ind = exog_mat_ind.reshape(-1, 1)
        preds_tuple = industry_predict(
            plan.industry_estimator_,
            steps=h,
            history=hist,
            exog_future=exog_mat_ind,
        )
        disclosures = [
            f"Industry horizon generate (backend={plan.backend}, method={plan.method}); no refit.",
        ]
        if plan.method in {"prophet", "nbeats"}:
            disclosures.append(
                "Prophet/N-BEATS use model-native multi-step paths; "
                "not recursive lag composition."
            )
        return ForecastGenerateResult(
            method=plan.method,
            horizon=h,
            origin=origin,
            predictions=preds_tuple,
            timestamps=(),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    exog_mat: np.ndarray | None = None
    disclosures = [
        f"Horizon generate from origin={origin} with frozen ForecastPlan "
        f"(method={plan.method}); no refit.",
        "Recursive multi-step: each step may feed predicted values into later lags.",
    ]
    if plan.exog_columns:
        if future_exog is None:
            raise ValidationError(
                "Plan uses exogenous columns; pass future_exog with shape "
                f"(horizon={h}, n_exog={len(plan.exog_columns)})."
            )
        if isinstance(future_exog, pd.DataFrame):
            missing = [c for c in plan.exog_columns if c not in future_exog.columns]
            if missing:
                raise ValidationError(f"future_exog missing columns: {missing}")
            exog_mat = future_exog.loc[:, list(plan.exog_columns)].to_numpy(dtype=float)
        else:
            exog_mat = np.asarray(future_exog, dtype=float)
        if exog_mat.ndim == 1:
            exog_mat = exog_mat.reshape(-1, 1)
        if exog_mat.shape != (h, len(plan.exog_columns)):
            raise ValidationError(
                f"future_exog shape {exog_mat.shape} != "
                f"(horizon={h}, n_exog={len(plan.exog_columns)})"
            )
        if np.isnan(exog_mat).any():
            raise ValidationError("future_exog contains nulls")
        disclosures.append(
            "Exogenous horizon path uses caller-supplied future_exog "
            "(not inferred by BuildML)."
        )

    preds: list[float] = []
    working = list(hist)
    for step in range(1, h + 1):
        row = None if exog_mat is None else exog_mat[step - 1]
        yhat = _predict_next(plan, working, step=step, exog_row=row)
        preds.append(yhat)
        working.append(yhat)

    if plan.method in {"naive", "mean", "seasonal_naive", "drift"}:
        disclosures.append(
            "Baseline methods are intentionally simple reference forecasts; "
            "prefer lag_ridge/lag_hgb when lag structure matters."
        )

    return ForecastGenerateResult(
        method=plan.method,
        horizon=h,
        origin=origin,
        predictions=tuple(preds),
        timestamps=(),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def history_through_partition(
    dataset: Dataset,
    plan: ForecastPlan,
    split_plan: SplitPlan,
    *,
    through: str = "train",
) -> list[float]:
    """Collect chronologically ordered target history through a partition.

    Concatenates train, validation, and/or test targets in clock order for
    evaluation origins and rolling one-step protocols.

    Parameters
    ----------
    dataset:
        Session dataset containing all split partitions.
    plan:
        Train-fitted plan defining target and time columns.
    split_plan:
        Temporal split plan defining partition membership.
    through:
        Last partition to include: ``train``, ``validation``, or ``test``.

    Returns
    -------
    list[float]
        Chronologically ordered target values through the requested partition.

    Raises
    ------
    ValidationError
        When ``through`` is not ``train``, ``validation``, or ``test``.
    """
    if through == "train":
        parts = ["train"]
    elif through == "validation":
        parts = ["train"]
        if split_plan.validation_indices:
            parts.append("validation")
    elif through == "test":
        parts = ["train"]
        if split_plan.validation_indices:
            parts.append("validation")
        parts.append("test")
    else:
        raise ValidationError(
            f"through must be train|validation|test, got {through!r}"
        )

    values: list[float] = []
    for name in parts:
        frame = ordered_frame(
            dataset, split_plan, name, time_column=plan.time_column
        )
        values.extend(float(v) for v in target_series(frame, plan.target_column).tolist())
    return values


def rolling_one_step_predictions(
    plan: ForecastPlan,
    history: list[float],
    actuals: np.ndarray,
    *,
    exog: np.ndarray | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Walk holdout actuals with one-step predictions using prior actuals.

    At step *i*, predicts using ``history + actuals[:i]`` and never future
    holdout targets. Appends each actual after scoring so later lags stay honest.

    Parameters
    ----------
    plan:
        Train-fitted :class:`ForecastPlan` to score without refit.
    history:
        Target history ending immediately before the holdout partition.
    actuals:
        Holdout actual target values in chronological order.
    exog:
        Optional holdout exogenous matrix with one row per actual.

    Returns
    -------
    predictions : tuple[float, ...]
        One-step forecasts aligned with ``actuals``.
    actuals_out : tuple[float, ...]
        Holdout actual values used for scoring.

    Raises
    ------
    ValidationError
        When holdout actuals are empty, exog row count mismatches, or plan
        state required by the method is missing.
    """
    actuals = np.asarray(actuals, dtype=float).reshape(-1)
    if actuals.size == 0:
        raise ValidationError("No holdout actuals to evaluate")
    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.shape[0] != actuals.shape[0]:
            raise ValidationError("exog rows must match holdout actuals length")

    working = list(float(v) for v in history)
    preds: list[float] = []
    for i in range(int(actuals.shape[0])):
        row = None if exog is None else exog[i]
        if plan.method == "naive":
            yhat = float(working[-1])
        elif plan.method == "mean":
            if plan.baseline_value_ is None:
                raise ValidationError("mean plan is missing baseline_value_")
            yhat = float(plan.baseline_value_)
        elif plan.method == "drift":
            if plan.drift_slope_ is None:
                raise ValidationError("drift plan is missing drift_slope_")
            yhat = float(working[-1] + plan.drift_slope_)
        elif plan.method == "seasonal_naive":
            period = plan.seasonal_period or len(plan.seasonal_history_) or 1
            if len(working) < period:
                raise ValidationError(
                    f"Need at least seasonal_period={period} history rows for "
                    "rolling seasonal_naive"
                )
            yhat = float(working[-period])
        elif plan.method in {"lag_ridge", "lag_hgb"}:
            yhat = _predict_next(plan, working, step=1, exog_row=row)
        elif plan.industry_estimator_ is not None:
            yhat = industry_one_step(
                plan.industry_estimator_, working, exog_row=row
            )
        else:
            yhat = _predict_next(plan, working, step=1, exog_row=row)
        preds.append(float(yhat))
        working.append(float(actuals[i]))
    return tuple(preds), tuple(float(v) for v in actuals.tolist())


def origin_predictions(
    plan: ForecastPlan,
    history: list[float],
    n_points: int,
    *,
    future_exog: np.ndarray | None = None,
) -> tuple[tuple[float, ...], Any]:
    """Generate a fixed-origin multi-step forecast over holdout steps.

    Delegates to :func:`generate_forecast` with origin label ``eval_origin``
    so origin-strategy evaluation shares the same recursive path as generate.

    Parameters
    ----------
    plan:
        Train-fitted :class:`ForecastPlan` to score without refit.
    history:
        Target history ending at the evaluation origin.
    n_points:
        Number of holdout steps to forecast in one recursive pass.
    future_exog:
        Optional future exogenous rows for exog-aware plans.

    Returns
    -------
    predictions : tuple[float, ...]
        Multi-step forecasts covering ``n_points`` steps.
    generated :
        Full :class:`ForecastGenerateResult` from the underlying generate call.

    Raises
    ------
    ValidationError
        Propagated from :func:`generate_forecast` when inputs are invalid.
    """
    generated = generate_forecast(
        plan,
        horizon=n_points,
        history=history,
        future_exog=future_exog,
        origin="eval_origin",
    )
    return generated.predictions, generated
