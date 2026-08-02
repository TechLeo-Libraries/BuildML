#!/usr/bin/env python3
"""Subset M4-style rolling-origin forecast benchmark (synthetic daily series)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from buildml import Session
from buildml.forecasting.extras import statsmodels_available


def _series(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2019-01-01", periods=n, freq="D")
    y = 20 + 0.05 * np.arange(n) + 3 * np.sin(2 * np.pi * np.arange(n) / 7)
    y += rng.normal(0, 0.3, n)
    return pd.DataFrame({"ds": t, "y": y})


def _run_method(method: str, frame: pd.DataFrame) -> dict[str, float | str | None]:
    session = (
        Session.ingest(frame)
        .set_roles({"ds": "time", "y": "target"})
        .time_split(test_size=0.2, validation_size=0.1)
    )
    try:
        session.fit_forecast(method=method, horizon=7, seasonal_period=7, lags=[1, 2, 3, 7])
    except Exception as exc:  # noqa: BLE001
        return {"method": method, "error": str(exc)}
    rolling = session.evaluate_forecast(partition="test", strategy="rolling_one_step")
    origin = session.evaluate_forecast(partition="test", strategy="rolling_origin")
    return {
        "method": method,
        "rolling_mae": rolling.metrics.get("mae"),
        "rolling_rmse": rolling.metrics.get("rmse"),
        "origin_mae": origin.metrics.get("mae"),
        "origin_rmse": origin.metrics.get("rmse"),
    }


def main() -> int:
    frame = _series(42)
    methods = ["naive", "seasonal_naive", "lag_ridge"]
    if statsmodels_available():
        methods.extend(["ets", "auto_arima"])
    rows = [_run_method(m, frame) for m in methods]
    payload = {"methods": rows, "statsmodels": statsmodels_available()}
    out = Path(__file__).resolve().parent / "results" / "forecast_m4_subset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
