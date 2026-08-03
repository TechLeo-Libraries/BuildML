"""Train-only classical forecaster fitting with temporal leakage guards."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.forecasting.backends import fit_industry_backend
from buildml.forecasting.catalog import (
    CORE_BASELINE_METHODS,
    EXOG_COMPATIBLE_METHODS,
    method_requires_extra,
    method_supports_exog,
    resolve_default_method,
)
from buildml.forecasting.features import (
    assert_partition_time_order,
    assert_temporal_split,
    build_lag_matrix,
    normalize_lags,
    ordered_frame,
    resolve_exog_columns,
    resolve_target_column,
    resolve_time_column,
    stamp_strings,
    target_series,
)
from buildml.forecasting.results import ForecastFitResult, ForecastPlan
from buildml.forecasting.types import ForecastConfig, ForecastMethod


def fit_forecaster(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: ForecastMethod = "auto",
    horizon: int = 1,
    lags: list[int] | tuple[int, ...] | None = None,
    seasonal_period: int | None = None,
    exog_columns: list[str] | None = None,
    target_column: str | None = None,
    time_column: str | None = None,
    random_state: int | None = 0,
    alpha: float = 1.0,
    max_iter: int = 100,
    max_depth: int | None = 3,
    learning_rate: float = 0.1,
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    nbeats_input_size: int = 24,
    nbeats_horizon: int | None = None,
) -> tuple[ForecastPlan, ForecastFitResult]:
    """Fit a classical forecaster on the train partition only.

    Resolves method defaults, enforces temporal split guards, and returns a
    frozen :class:`ForecastPlan` ready for generate and evaluate.

    Parameters
    ----------
    dataset:
        Session dataset with time and target columns assigned.
    split_plan:
        Temporal split plan defining the train partition.
    method:
        ``auto`` picks ETS when statsmodels installed else ``lag_ridge``.
        Baselines (``naive``, ``seasonal_naive``, ``drift``, ``mean``) or
        lag-tabular (``lag_ridge``, ``lag_hgb``). With ``buildml[timeseries]``:
        ``arima``, ``auto_arima``, ``ets``, ``sarimax``. Prophet / N-BEATS
        behind ``timeseries-prophet`` / ``timeseries-ml``.
    horizon:
        Default forecast horizon stored on the plan (generate may override).
    lags:
        Positive lag orders used by lag models (and as seasonal fallback).
    seasonal_period:
        Required for ``seasonal_naive``; optional otherwise.
    exog_columns:
        Optional numeric exogenous columns. Univariate when empty. Horizon
        generation with exog requires future exogenous values (disclosed).
    target_column:
        Optional explicit target column overriding dataset role resolution.
    time_column:
        Optional explicit time column overriding dataset role resolution.
    random_state:
        Seed for stochastic lag_hgb and N-BEATS estimators.
    alpha:
        Ridge regularisation strength for ``lag_ridge``.
    max_iter:
        Maximum iterations for ``lag_hgb`` or N-BEATS training.
    max_depth:
        Tree depth limit for ``lag_hgb``.
    learning_rate:
        Learning rate for ``lag_hgb``.
    order:
        ARIMA ``(p, d, q)`` order for statsmodels methods.
    seasonal_order:
        SARIMAX seasonal ``(P, D, Q, s)`` tuple.
    nbeats_input_size:
        N-BEATS lookback window length.
    nbeats_horizon:
        N-BEATS native forecast horizon; defaults to ``horizon`` when ``None``.

    Returns
    -------
    tuple[ForecastPlan, ForecastFitResult]
        Train-fitted plan and fit summary for history logs.

    Raises
    ------
    MissingExtraError
        When the requested method requires an optional extra that is not installed.
    ValidationError
        When temporal guards fail, hyperparameters are invalid, or the method is
        unsupported.

    Notes
    -----
    **Leakage:** Requires ``time_split`` (or chronologically ordered
    ``inject_split``). Random/stratified/group splits are refused. Features
    at time *t* use only past target values (and optional contemporaneous exog).
    """
    assert_fit_partition(split_plan, "train")
    assert_temporal_split(split_plan)
    assert split_plan is not None
    method = resolve_default_method(method)  # type: ignore[arg-type]
    extra_req = method_requires_extra(method)
    if extra_req is not None:
        raise MissingExtraError(extra_req, f"{method} forecasting")

    if horizon < 1:
        raise ValidationError("horizon must be >= 1")

    time_col = resolve_time_column(dataset, time_column)
    target_col = resolve_target_column(dataset, target_column)
    lag_tuple = normalize_lags(lags)
    exog = resolve_exog_columns(
        dataset, exog_columns, target_column=target_col, time_column=time_col
    )
    assert_partition_time_order(dataset, split_plan, time_column=time_col)

    train = ordered_frame(dataset, split_plan, "train", time_column=time_col)
    y = target_series(train, target_col)
    n_train = int(y.shape[0])
    train_end = stamp_strings([train[time_col].iloc[-1]])[0]
    disclosures: list[str] = [
        "Forecaster fitted on train partition only after chronological ordering.",
        "Lag features at time t use only y[t-lag] (no future target leakage).",
        "This is a classical lag/baseline forecast path: not a full econometrics "
        "suite, not ARIMA productization, and not a digital twin.",
    ]
    warnings: list[str] = []
    univariate = len(exog) == 0
    if univariate:
        disclosures.append("Univariate mode: target history only (no exogenous columns).")
    else:
        if not method_supports_exog(method):
            raise ValidationError(
                f"method='{method}' does not accept exogenous columns. "
                f"Use one of {sorted(EXOG_COMPATIBLE_METHODS)} or omit exog_columns."
            )
        disclosures.append(
            "Exogenous mode: contemporaneous numeric exog columns are included in "
            "lag-model features. Horizon generate requires future exog values; "
            "rolling one-step eval may use holdout exog at each step (disclosed)."
        )

    estimator: Any = None
    industry_estimator: Any = None
    backend = "sklearn"
    baseline: float | None = None
    drift_slope: float | None = None
    seasonal_history: tuple[float, ...] = ()
    last_train = tuple(float(v) for v in y.tolist())
    n_fit = n_train

    if method == "naive":
        baseline = float(y[-1])
        disclosures.append("Naive baseline: forecast equals the last train observation.")
    elif method == "mean":
        baseline = float(np.mean(y))
        disclosures.append("Mean baseline: forecast equals the train mean.")
    elif method == "drift":
        if n_train < 2:
            raise ValidationError("drift method requires at least 2 train rows")
        drift_slope = float((y[-1] - y[0]) / (n_train - 1))
        baseline = float(y[-1])
        disclosures.append(
            "Drift baseline: linear extrapolation from first to last train point."
        )
    elif method == "seasonal_naive":
        period = int(seasonal_period) if seasonal_period is not None else None
        if period is None:
            # Prefer the largest lag as a seasonal period when unspecified.
            period = int(max(lag_tuple))
            warnings.append(
                f"seasonal_naive: seasonal_period not set; using max(lags)={period}."
            )
        if period < 1:
            raise ValidationError("seasonal_period must be >= 1")
        if n_train < period:
            raise ValidationError(
                f"seasonal_naive needs n_train >= seasonal_period ({period}); "
                f"have {n_train}"
            )
        seasonal_period = period
        seasonal_history = tuple(float(v) for v in y[-period:].tolist())
        disclosures.append(
            f"Seasonal naive: repeats the last seasonal_period={period} train values."
        )
    elif method in {"lag_ridge", "lag_hgb"}:
        exog_mat = None
        if exog:
            exog_mat = train.loc[:, list(exog)].to_numpy(dtype=float)
            if np.isnan(exog_mat).any():
                raise ValidationError(
                    "Exogenous columns contain nulls; impute before fit_forecast"
                )
        x_mat, y_mat, _start = build_lag_matrix(y, lag_tuple, exog=exog_mat)
        n_fit = int(y_mat.shape[0])
        if method == "lag_ridge":
            if alpha < 0:
                raise ValidationError("alpha must be >= 0 for lag_ridge")
            estimator = Ridge(alpha=float(alpha))
            estimator.fit(x_mat, y_mat)
            disclosures.append(
                "lag_ridge: Ridge on lag (and optional exog) features; "
                "multi-step generate uses recursive one-step predictions."
            )
        else:
            estimator = HistGradientBoostingRegressor(
                max_iter=int(max_iter),
                max_depth=max_depth,
                learning_rate=float(learning_rate),
                random_state=random_state,
            )
            estimator.fit(x_mat, y_mat)
            disclosures.append(
                "lag_hgb: HistGradientBoostingRegressor on lag/exog features; "
                "recursive multi-step generate; not a sequence neural net."
            )
    elif method not in CORE_BASELINE_METHODS:
        exog_mat = None
        if exog:
            exog_mat = train.loc[:, list(exog)].to_numpy(dtype=float)
            if np.isnan(exog_mat).any():
                raise ValidationError(
                    "Exogenous columns contain nulls; impute before fit_forecast"
                )
        nb_h = int(nbeats_horizon if nbeats_horizon is not None else horizon)
        outcome = fit_industry_backend(
            y,
            method=method,
            seasonal_period=seasonal_period,
            exog=exog_mat,
            order=order,
            seasonal_order=seasonal_order,
            random_state=random_state,
            nbeats_input_size=nbeats_input_size,
            nbeats_horizon=nb_h,
            max_iter=max_iter,
        )
        industry_estimator = outcome.estimator
        backend = outcome.backend
        disclosures.extend(outcome.disclosures)
        warnings.extend(outcome.warnings)
        n_fit = n_train
        if method == "nbeats":
            disclosures.append(
                "N-BEATS generate uses the neuralforecast model directly; "
                "rolling eval may refit slices (disclosed in evaluate)."
            )
    else:
        raise ValidationError(f"Unsupported forecast method '{method}'")

    config = ForecastConfig(
        method=method,
        horizon=int(horizon),
        lags=lag_tuple,
        seasonal_period=seasonal_period,
        exog_columns=exog,
        target_column=target_col,
        time_column=time_col,
        random_state=random_state,
        alpha=alpha,
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        order=order,
        seasonal_order=seasonal_order,
        nbeats_input_size=nbeats_input_size,
        nbeats_horizon=int(nbeats_horizon if nbeats_horizon is not None else horizon),
    )
    plan = ForecastPlan(
        method=method,
        target_column=target_col,
        time_column=time_col,
        horizon=int(horizon),
        lags=lag_tuple,
        seasonal_period=seasonal_period,
        exog_columns=exog,
        n_train_rows=n_train,
        n_fit_rows=n_fit,
        train_end_stamp=train_end,
        estimator_=estimator,
        baseline_value_=baseline,
        drift_slope_=drift_slope,
        seasonal_history_=seasonal_history,
        last_train_values_=last_train,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
        univariate=univariate,
        backend=backend,
        industry_estimator_=industry_estimator,
    )
    result = ForecastFitResult(
        method=method,
        target_column=target_col,
        time_column=time_col,
        n_train_rows=n_train,
        n_fit_rows=n_fit,
        horizon=int(horizon),
        lags=lag_tuple,
        univariate=univariate,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        train_end_stamp=train_end,
    )
    return plan, result
