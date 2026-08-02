"""Forecasting alpha-gate smoke: time_split → fit → eval → generate → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_forecasting_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 110
    t = pd.date_range("2024-03-01", periods=n, freq="D")
    y = 12 + 0.02 * np.arange(n) + np.sin(np.arange(n) / 6.0) + rng.normal(0, 0.2, n)
    frame = pd.DataFrame({"ts": t, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )

    fit = session.fit_forecast(
        method="lag_ridge",
        horizon=7,
        lags=[1, 2, 3, 7],
        alpha=0.5,
    )
    assert fit.univariate is True
    assert session.forecast_plan is not None
    assert session.forecast_fit_result is not None

    val = session.evaluate_forecast(partition="validation", strategy="rolling_one_step")
    assert val.partition == "validation"
    assert "mae" in val.metrics
    assert session.forecast_eval_result is not None

    gen = session.generate_forecast(horizon=7, origin="train_end")
    assert len(gen.predictions) == 7

    before = session.explain("fit_forecast", moment="before")
    assert before.operation == "fit_forecast"
    assert before.prerequisite_status.get("split") is True

    bundle = session.save_forecast_bundle(tmp_path / "forecast_bundle")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "forecast_plan.joblib").is_file()

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )
    restored.load_forecast_bundle(bundle)
    again = restored.generate_forecast(horizon=7)
    assert again.predictions == gen.predictions

    # Baseline comparison path still works on the same temporal split recipe
    baseline = (
        Session.ingest(frame)
        .set_roles({"ts": "time", "y": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )
    baseline.fit_forecast(method="seasonal_naive", seasonal_period=7, horizon=7)
    base_metrics = baseline.evaluate_forecast(partition="test")
    assert "rmse" in base_metrics.metrics
