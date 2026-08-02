"""Leakage-safe forecast evaluation metrics on holdout partitions."""

from __future__ import annotations

from typing import Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan
from buildml.forecasting.features import ordered_frame, target_series
from buildml.forecasting.predict import (
    history_through_partition,
    origin_predictions,
    rolling_one_step_predictions,
)
from buildml.forecasting.results import ForecastEvalResult, ForecastPlan
from buildml.forecasting.types import ForecastEvalStrategy


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValidationError("actuals and predictions length mismatch")
    if y_true.size == 0:
        raise ValidationError("No points to score")
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    metrics = {"mae": mae, "rmse": rmse}
    nonzero = np.abs(y_true) > 1e-12
    if int(nonzero.sum()) == 0:
        metrics["mape"] = float("nan")
    else:
        mape = float(
            np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100.0
        )
        metrics["mape"] = mape
    return metrics


def evaluate_forecast(
    dataset: Dataset,
    plan: ForecastPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionName = "test",
    strategy: ForecastEvalStrategy | Literal["rolling_one_step", "origin"] = (
        "rolling_one_step"
    ),
) -> ForecastEvalResult:
    """Score a train-fitted ForecastPlan on a holdout partition.

    Strategies
    ----------
    rolling_one_step:
        Chronological walk: at each holdout step, predict one step using all
        prior *actual* targets (train + earlier holdout). Never uses future
        holdout targets as features.
    origin:
        Fixed origin at the end of prior partitions; recursive multi-step
        forecast of length ``len(partition)`` compared to holdout actuals.
        Harder and often worse than rolling — disclosed.

    Metrics
    -------
    MAE, RMSE, and MAPE (MAPE undefined / NaN when all actuals are ~0).
    MAPE is scale-sensitive and unstable near zero — disclosed limitation.
    """
    if split_plan is None:
        raise ValidationError(
            f"partition='{partition}' requires a SplitPlan. Call session.time_split(...)."
        )
    if partition not in {"validation", "test"}:
        raise ValidationError(
            "evaluate_forecast partition must be 'validation' or 'test' "
            "(train metrics are optimistic and refused as the default eval API)."
        )
    indices = split_plan.indices_for(partition)
    if not indices:
        raise ValidationError(f"Partition '{partition}' is empty")

    holdout = ordered_frame(
        dataset, split_plan, partition, time_column=plan.time_column
    )
    actuals = target_series(holdout, plan.target_column)
    # Validation scores from train end; test may include validation actuals in history.
    through = "train"
    if partition == "test" and split_plan.validation_indices:
        through = "validation"
    history = history_through_partition(
        dataset, plan, split_plan, through=through
    )

    disclosures = [
        f"Evaluated on partition={partition} with strategy={strategy}.",
        "Metrics use holdout actuals vs forecasts from a train-fitted plan.",
        "MAPE is undefined or unstable near zero targets; prefer MAE/RMSE then.",
        "Not a full econometric residual diagnostic suite.",
    ]
    warnings: list[str] = []
    recommendations: list[str] = []

    exog = None
    if plan.exog_columns:
        exog = holdout.loc[:, list(plan.exog_columns)].to_numpy(dtype=float)
        if np.isnan(exog).any():
            raise ValidationError(
                "Holdout exogenous columns contain nulls; impute before evaluate_forecast"
            )
        disclosures.append(
            "Exog-aware eval uses holdout exogenous values at each scored timestamp "
            "(known at that clock time in this offline evaluation setting)."
        )

    if strategy == "rolling_one_step":
        preds, acts = rolling_one_step_predictions(
            plan, history, actuals, exog=exog
        )
        disclosures.append(
            "rolling_one_step appends holdout *actuals* into history after each "
            "prediction (standard expanding one-step protocol; no future leakage)."
        )
    elif strategy == "origin":
        preds, _gen = origin_predictions(
            plan, history, int(actuals.shape[0]), future_exog=exog
        )
        acts = tuple(float(v) for v in actuals.tolist())
        disclosures.append(
            "origin strategy is a fixed multi-step recursive forecast from the "
            "end of prior partitions; error compounds with horizon."
        )
        recommendations.append(
            "Compare origin vs rolling_one_step; large gaps often mean weak "
            "multi-step recursion rather than useless one-step skill."
        )
    else:
        raise ValidationError(
            f"Unsupported evaluate strategy '{strategy}'. "
            "Use 'rolling_one_step' or 'origin'."
        )

    y_true = np.asarray(acts, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    metrics = _metrics(y_true, y_pred)
    if np.isnan(metrics.get("mape", 0.0)):
        warnings.append("MAPE is NaN because all |actual| values are ~0.")

    if metrics["mae"] > 0 and plan.method in {"lag_ridge", "lag_hgb"}:
        recommendations.append(
            "Compare against naive / seasonal_naive baselines on the same split "
            "before claiming lag-model value."
        )
    recommendations.append(
        "Refuse shuffled splits for forecasting; keep time_split for production claims."
    )

    return ForecastEvalResult(
        partition=str(partition),
        method=plan.method,
        strategy=str(strategy),
        n_points=int(y_true.shape[0]),
        metrics=metrics,
        predictions=tuple(float(v) for v in y_pred.tolist()),
        actuals=tuple(float(v) for v in y_true.tolist()),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        recommendations=tuple(recommendations),
    )
