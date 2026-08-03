"""Thin Session facades over buildml.forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.forecasting.checkpoint import load_forecast_bundle, save_forecast_bundle
from buildml.forecasting.evaluate import evaluate_forecast
from buildml.forecasting.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    generate_result_summary,
)
from buildml.forecasting.fit import fit_forecaster
from buildml.forecasting.predict import generate_forecast, history_through_partition
from buildml.forecasting.types import ForecastEvalStrategy, ForecastMethod


def fit_forecast(
    session,
    *,
    method: ForecastMethod = "lag_ridge",
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
) -> Any:
    """Fit a classical forecaster on the train partition only.

    Delegates to :func:`buildml.forecasting.fit.fit_forecaster`, stores the
    :class:`~buildml.forecasting.results.ForecastPlan` on Session, and records
    the fit. Follow with :func:`generate_forecast_op` or
    :func:`evaluate_forecast_op`.

    Parameters
    ----------
    session:
        Active Session with a chronologically ordered dataset and split plan.
    method:
        Forecasting method (``lag_ridge``, ``arima``, ``nbeats``, etc.).
    horizon:
        Default forecast horizon in steps.
    lags:
        Explicit lag indices for lag-based methods.
    seasonal_period:
        Seasonal period for seasonal methods.
    exog_columns:
        Optional exogenous regressor columns.
    target_column:
        Target series column; defaults to target role.
    time_column:
        Timestamp column; inferred when omitted.
    random_state:
        Seed for stochastic estimators.
    alpha:
        Regularization strength for ridge-style methods.
    max_iter:
        Maximum iterations for iterative solvers.
    max_depth:
        Tree depth for tree-based forecasters.
    learning_rate:
        Learning rate for gradient boosting forecasters.
    order:
        ARIMA ``(p, d, q)`` order tuple.
    seasonal_order:
        Seasonal ARIMA ``(P, D, Q, s)`` order tuple.
    nbeats_input_size:
        Input window size for N-BEATS backend.
    nbeats_horizon:
        N-BEATS forecast horizon override.

    Returns
    -------
    ForecastFitResult
        Serializable fit summary including method and horizon disclosures.

    Notes
    -----
    **Leakage:** Requires ``time_split`` (or chronologically ordered
    ``inject_split``). Random/stratified/group splits are refused. Lag features
    use only past target values.
    """
    session.assert_can_fit("train")
    plan, result = fit_forecaster(
        session.dataset,
        session._split_plan,
        method=method,
        horizon=horizon,
        lags=lags,
        seasonal_period=seasonal_period,
        exog_columns=exog_columns,
        target_column=target_column,
        time_column=time_column,
        random_state=random_state,
        alpha=alpha,
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        order=order,
        seasonal_order=seasonal_order,
        nbeats_input_size=nbeats_input_size,
        nbeats_horizon=nbeats_horizon,
    )
    session._forecast_plan = plan
    session._forecast_fit_result = result
    session._forecast_generate_result = None
    session._forecast_eval_result = None
    session._record(
        "fit_forecast",
        {
            "method": method,
            "horizon": horizon,
            "lags": None if lags is None else list(lags),
            "seasonal_period": seasonal_period,
            "exog_columns": exog_columns,
            "target_column": target_column,
            "time_column": time_column,
            "alpha": alpha,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def generate_forecast_op(
    session,
    *,
    horizon: int | None = None,
    origin: str = "train_end",
    future_exog: np.ndarray | pd.DataFrame | list[list[float]] | None = None,
) -> Any:
    """Generate an H-step forecast from the train-fitted ForecastPlan.

    Delegates to :func:`buildml.forecasting.predict.generate_forecast` without
    refitting. History is taken from train end or extended through validation/test
    when ``origin`` requests it.

    Parameters
    ----------
    session:
        Active Session with a ForecastPlan from :func:`fit_forecast`.
    horizon:
        Forecast steps; defaults to the plan horizon when ``None``.
    origin:
        History cutoff (``train_end``, ``validation_end``, or ``test_end``).
    future_exog:
        Optional exogenous values for the forecast horizon.

    Returns
    -------
    ForecastGenerateResult
        Point forecasts and optional intervals for the requested horizon.

    Raises
    ------
    ValidationError
        When no forecast plan exists or ``origin`` requires a missing split.
    """
    plan = getattr(session, "_forecast_plan", None)
    if plan is None:
        raise ValidationError("No forecast plan. Call fit_forecast(...) first.")
    history = None
    if origin == "train_end":
        history = list(plan.last_train_values_)
    elif origin in {"validation_end", "test_end"}:
        through = "validation" if origin == "validation_end" else "test"
        if session._split_plan is None:
            raise ValidationError("origin beyond train_end requires a SplitPlan")
        history = history_through_partition(
            session.dataset, plan, session._split_plan, through=through
        )
    elif origin != "train_end":
        raise ValidationError(
            "origin must be one of: train_end, validation_end, test_end"
        )

    exog = future_exog
    if isinstance(future_exog, list):
        exog = np.asarray(future_exog, dtype=float)

    result = generate_forecast(
        plan,
        horizon=horizon,
        history=history,
        future_exog=exog,
        origin=origin,
    )
    session._forecast_generate_result = result
    session._record(
        "generate_forecast",
        {"horizon": horizon, "origin": origin, "has_future_exog": future_exog is not None},
        warnings=tuple(result.warnings),
        result_summary=generate_result_summary(result),
    )
    return result


def evaluate_forecast_op(
    session,
    *,
    partition: PartitionName = "test",
    strategy: ForecastEvalStrategy = "rolling_one_step",
) -> Any:
    """Evaluate the train-fitted ForecastPlan on a holdout partition.

    Delegates to :func:`buildml.forecasting.evaluate.evaluate_forecast` using
    rolling or static evaluation strategies. Falls back to ``test`` when no
    validation partition exists.

    Parameters
    ----------
    session:
        Active Session with a ForecastPlan from :func:`fit_forecast`.
    partition:
        Holdout partition (default ``test``).
    strategy:
        Evaluation strategy (``rolling_one_step`` or ``static_multi_step``).

    Returns
    -------
    ForecastEvalResult
        Holdout error metrics for the frozen forecast plan.

    Raises
    ------
    ValidationError
        When no forecast plan exists on the Session.
    """
    plan = getattr(session, "_forecast_plan", None)
    if plan is None:
        raise ValidationError("No forecast plan. Call fit_forecast(...) first.")
    resolved: PartitionName = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_forecast(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        strategy=strategy,
    )
    session._forecast_eval_result = result
    session._record(
        "evaluate_forecast",
        {"partition": resolved, "strategy": strategy},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_forecast_bundle_op(session, path: str | Path) -> Path:
    """Persist the active ForecastPlan as ``buildml.forecast_bundle.v2``.

    Delegates to :func:`buildml.forecasting.checkpoint.save_forecast_bundle`.
    Reload with :func:`load_forecast_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a ForecastPlan from :func:`fit_forecast`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no forecast plan exists on the Session.
    """
    plan = getattr(session, "_forecast_plan", None)
    if plan is None:
        raise ValidationError("No forecast plan. Call fit_forecast(...) first.")
    out = save_forecast_bundle(
        path,
        plan,
        fit_result=getattr(session, "_forecast_fit_result", None),
        eval_result=getattr(session, "_forecast_eval_result", None),
        generate_result=getattr(session, "_forecast_generate_result", None),
    )
    session._record(
        "save_forecast_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "method": plan.method,
            "horizon": plan.horizon,
        },
    )
    return out


def load_forecast_bundle_op(session, path: str | Path) -> Any:
    """Load a forecast bundle into this Session.

    Delegates to :func:`buildml.forecasting.checkpoint.load_forecast_bundle`
    and clears prior generate/eval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded ForecastPlan.
    path:
        Path to a ``buildml.forecast_bundle.v2`` directory.

    Returns
    -------
    Session
        ``session`` with ForecastPlan attached for chaining.
    """
    plan = load_forecast_bundle(path)
    session._forecast_plan = plan
    session._forecast_fit_result = None
    session._forecast_generate_result = None
    session._forecast_eval_result = None
    session._record(
        "load_forecast_bundle",
        {"path": str(path), "method": plan.method, "horizon": plan.horizon},
        result_summary=plan.to_dict(),
    )
    return session
