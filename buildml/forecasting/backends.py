"""Industry forecasting backend fitters (statsmodels, Prophet, neuralforecast)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.forecasting.extras import (
    neuralforecast_available,
    prophet_available,
    require_neuralforecast,
    require_prophet,
    require_statsmodels,
    statsmodels_available,
)


@dataclass(slots=True)
class IndustryFitOutcome:
    estimator: Any
    backend: str
    method: str
    disclosures: list[str]
    warnings: list[str]
    extra: dict[str, Any]


def fit_industry_backend(
    y: np.ndarray,
    *,
    method: str,
    seasonal_period: int | None = None,
    exog: np.ndarray | None = None,
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    random_state: int | None = 0,
    nbeats_input_size: int = 24,
    nbeats_horizon: int = 7,
    max_iter: int = 50,
) -> IndustryFitOutcome:
    """Fit an industry backend on train target history."""
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    disclosures: list[str] = []
    warnings: list[str] = []
    extra: dict[str, Any] = {}

    if method in {"arima", "auto_arima", "ets", "sarimax"}:
        if not statsmodels_available():
            raise MissingExtraError("timeseries", f"{method} forecasting")
        require_statsmodels(feature=f"{method} forecasting")
        return _fit_statsmodels(
            y,
            method=method,
            seasonal_period=seasonal_period,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
        )

    if method == "prophet":
        if not prophet_available():
            raise MissingExtraError("timeseries-prophet", "Prophet forecasting")
        return _fit_prophet(y, seasonal_period=seasonal_period)

    if method == "nbeats":
        if not neuralforecast_available():
            raise MissingExtraError("timeseries-ml", "N-BEATS forecasting")
        return _fit_nbeats(
            y,
            input_size=nbeats_input_size,
            horizon=nbeats_horizon,
            max_steps=max_iter,
            random_state=random_state,
        )

    raise ValidationError(f"Unknown industry forecast method '{method}'")


def _fit_statsmodels(
    y: np.ndarray,
    *,
    method: str,
    seasonal_period: int | None,
    exog: np.ndarray | None,
    order: tuple[int, int, int] | None,
    seasonal_order: tuple[int, int, int, int] | None,
) -> IndustryFitOutcome:
    disclosures: list[str] = []
    warnings: list[str] = []
    n = len(y)

    if method == "ets":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        period = seasonal_period or 7
        seasonal = "add" if n >= 2 * period else None
        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=period if seasonal else None,
        )
        fitted = model.fit(optimized=True, use_brute=False)
        disclosures.append(
            f"ETS (Holt-Winters) via statsmodels; trend=add, "
            f"seasonal={seasonal}, period={period if seasonal else None}."
        )
        return IndustryFitOutcome(
            estimator={"kind": "ets", "result": fitted, "seasonal_period": period},
            backend="statsmodels",
            method="ets",
            disclosures=disclosures,
            warnings=warnings,
            extra={"seasonal_period": period},
        )

    if method == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        ord_ = order or (1, 1, 1)
        model = ARIMA(y, order=ord_, exog=exog)
        fitted = model.fit()
        disclosures.append(f"ARIMA{ord_} via statsmodels.tsa.arima.model.ARIMA.")
        return IndustryFitOutcome(
            estimator={"kind": "arima", "result": fitted, "order": ord_},
            backend="statsmodels",
            method="arima",
            disclosures=disclosures,
            warnings=warnings,
            extra={"order": ord_},
        )

    if method == "auto_arima":
        ord_ = order or _auto_arima_order(y)
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(y, order=ord_)
        fitted = model.fit()
        disclosures.append(
            f"auto_arima: selected ARIMA{ord_} via AIC grid search on low orders."
        )
        warnings.append(
            "auto_arima is a lightweight AIC search — not pmdarima; "
            "pass explicit order= for production control."
        )
        return IndustryFitOutcome(
            estimator={"kind": "arima", "result": fitted, "order": ord_},
            backend="statsmodels",
            method="auto_arima",
            disclosures=disclosures,
            warnings=warnings,
            extra={"order": ord_},
        )

    if method == "sarimax":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        period = seasonal_period or 7
        ord_ = order or (1, 1, 1)
        sord = seasonal_order or (1, 0, 1, period)
        model = SARIMAX(y, exog=exog, order=ord_, seasonal_order=sord)
        fitted = model.fit(disp=False)
        disclosures.append(
            f"SARIMAX order={ord_}, seasonal_order={sord} via statsmodels."
        )
        return IndustryFitOutcome(
            estimator={
                "kind": "sarimax",
                "result": fitted,
                "order": ord_,
                "seasonal_order": sord,
            },
            backend="statsmodels",
            method="sarimax",
            disclosures=disclosures,
            warnings=warnings,
            extra={"order": ord_, "seasonal_order": sord},
        )

    raise ValidationError(f"Unsupported statsmodels method '{method}'")


def _auto_arima_order(y: np.ndarray) -> tuple[int, int, int]:
    """Small AIC grid for (p,d,q) with p,q in 0..2, d in 0..1."""
    from statsmodels.tsa.arima.model import ARIMA

    best: tuple[int, int, int] | None = None
    best_aic = float("inf")
    for p in range(3):
        for d in range(2):
            for q in range(3):
                if p == d == q == 0:
                    continue
                try:
                    res = ARIMA(y, order=(p, d, q)).fit()
                    aic = float(res.aic)
                    if aic < best_aic:
                        best_aic = aic
                        best = (p, d, q)
                except Exception:  # noqa: BLE001
                    continue
    return best or (1, 1, 1)


def _fit_prophet(
    y: np.ndarray,
    *,
    seasonal_period: int | None,
) -> IndustryFitOutcome:
    require_prophet()
    from prophet import Prophet

    n = len(y)
    ds = pd.date_range("2000-01-01", periods=n, freq="D")
    frame = pd.DataFrame({"ds": ds, "y": y})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=(seasonal_period == 7) if seasonal_period else True,
        yearly_seasonality=n >= 365,
    )
    model.fit(frame)
    disclosures = [
        "Prophet fit on synthetic daily ds index (train row order preserved).",
        "For production, ensure time_column maps to meaningful calendar ds.",
    ]
    warnings = [
        "Prophet uses internal calendar ds — multi-series / irregular timestamps "
        "need explicit ds alignment (disclosed limitation)."
    ]
    return IndustryFitOutcome(
        estimator={"kind": "prophet", "result": model, "n_train": n},
        backend="prophet",
        method="prophet",
        disclosures=disclosures,
        warnings=warnings,
        extra={"seasonal_period": seasonal_period},
    )


def _fit_nbeats(
    y: np.ndarray,
    *,
    input_size: int,
    horizon: int,
    max_steps: int,
    random_state: int | None,
) -> IndustryFitOutcome:
    require_neuralforecast()
    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS

    n = len(y)
    if n < input_size + horizon:
        raise ValidationError(
            f"N-BEATS needs n >= input_size + horizon ({input_size + horizon}); have {n}"
        )
    ds = pd.DataFrame(
        {
            "unique_id": ["series"] * n,
            "ds": pd.date_range("2000-01-01", periods=n, freq="D"),
            "y": y,
        }
    )
    model = NBEATS(
        input_size=int(input_size),
        h=int(horizon),
        max_steps=int(max_steps),
        random_seed=random_state or 0,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=ds)
    disclosures = [
        f"N-BEATS via neuralforecast (input_size={input_size}, h={horizon}, "
        f"max_steps={max_steps}).",
        "Neural forecast uses synthetic daily ds; irregular timestamps need alignment.",
    ]
    warnings = [
        "N-BEATS is a lightweight default — tune input_size/h/max_steps for production."
    ]
    return IndustryFitOutcome(
        estimator={
            "kind": "nbeats",
            "result": nf,
            "input_size": input_size,
            "horizon": horizon,
        },
        backend="neuralforecast",
        method="nbeats",
        disclosures=disclosures,
        warnings=warnings,
        extra={"input_size": input_size, "horizon": horizon},
    )


def industry_predict(
    estimator: Any,
    *,
    steps: int,
    history: list[float] | None = None,
    exog_future: np.ndarray | None = None,
) -> tuple[float, ...]:
    """Multi-step forecast from a fitted industry estimator wrapper."""
    if not isinstance(estimator, dict) or "kind" not in estimator:
        raise ValidationError("Invalid industry estimator payload")
    kind = estimator["kind"]
    result = estimator["result"]

    if kind == "ets":
        fc = result.forecast(steps)
        return tuple(float(v) for v in np.asarray(fc, dtype=float).tolist())

    if kind in {"arima", "sarimax"}:
        fc = result.forecast(steps, exog=exog_future)
        return tuple(float(v) for v in np.asarray(fc, dtype=float).tolist())

    if kind == "prophet":
        future = result.make_future_dataframe(periods=steps, freq="D", include_history=False)
        pred = result.predict(future)
        return tuple(float(v) for v in pred["yhat"].tolist()[-steps:])

    if kind == "nbeats":
        import pandas as pd

        nf = result
        h = int(estimator.get("horizon", steps))
        if history is None:
            raise ValidationError("N-BEATS predict requires history for rolling context")
        n = len(history)
        ds = pd.DataFrame(
            {
                "unique_id": ["series"] * n,
                "ds": pd.date_range("2000-01-01", periods=n, freq="D"),
                "y": history,
            }
        )
        pred = nf.predict(df=ds)
        vals = pred["NBEATS"].tolist()[-steps:]
        return tuple(float(v) for v in vals)

    raise ValidationError(f"Unsupported industry estimator kind '{kind}'")


def industry_one_step(
    estimator: Any,
    history: list[float],
    *,
    exog_row: np.ndarray | None = None,
) -> float:
    """One-step ahead prediction for rolling evaluation."""
    preds = industry_predict(estimator, steps=1, history=history, exog_future=exog_row)
    return float(preds[0])
