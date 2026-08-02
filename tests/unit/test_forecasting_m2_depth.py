"""Deeper forecasting coverage: protocols, walkthrough, AI allowlist, low-level API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.forecasting.evaluate import evaluate_forecast
from buildml.forecasting.fit import fit_forecaster
from buildml.forecasting.predict import generate_forecast


def _frame(n: int = 90, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-06-01", periods=n, freq="D")
    y = 5 + 0.03 * np.arange(n) + np.cos(np.arange(n) / 5.0) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"clock": t, "sales": y})


def test_low_level_fit_generate_evaluate(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"clock": "time", "sales": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )
    plan, fit = fit_forecaster(
        session.dataset,
        session.split_plan,
        method="lag_hgb",
        lags=[1, 2, 3],
        horizon=4,
        max_iter=40,
        max_depth=2,
        random_state=0,
    )
    assert fit.method == "lag_hgb"
    gen = generate_forecast(plan, horizon=4)
    assert len(gen.predictions) == 4
    ev = evaluate_forecast(
        session.dataset, plan, session.split_plan, partition="validation"
    )
    assert ev.metrics["rmse"] >= 0.0

    from buildml.forecasting.checkpoint import save_forecast_bundle

    out = save_forecast_bundle(tmp_path / "direct", plan, fit_result=fit, eval_result=ev)
    assert (out / "meta.json").is_file()


def test_injected_split_ok_when_chronological() -> None:
    frame = _frame(n=60)
    session = Session.ingest(frame).set_roles({"clock": "time", "sales": "target"})
    # Chronological inject: early train, late test
    train_idx = list(range(0, 40))
    test_idx = list(range(40, 60))
    session.inject_split(train_indices=train_idx, test_indices=test_idx)
    fit = session.fit_forecast(method="naive", horizon=3)
    assert fit.n_train_rows == 40
    metrics = session.evaluate_forecast(partition="test", strategy="rolling_one_step")
    assert metrics.n_points == 20


def test_injected_split_refuses_time_overlap() -> None:
    frame = _frame(n=40)
    # Shuffle indices so test can start before train ends in clock time
    session = Session.ingest(frame).set_roles({"clock": "time", "sales": "target"})
    session.inject_split(
        train_indices=list(range(10, 30)),
        test_indices=list(range(0, 10)),
    )
    with pytest.raises(LeakageError, match="Temporal leakage"):
        session.fit_forecast(method="mean")


def test_walkthrough_and_ai_tools_include_forecast() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"clock": "time", "sales": "target"})
        .time_split(test_size=0.25)
    )
    session.fit_forecast(method="lag_ridge", lags=[1, 2], horizon=3)
    session.evaluate_forecast(partition="test")
    report = session.walkthrough()
    payload = report.to_dict()
    status = payload["forecasting_status"]
    assert status["enabled"] is True
    assert status["method"] == "lag_ridge"
    assert any("ForecastPlan" in d or "forecast" in d.lower() for d in status["disclosures"])

    registry = build_default_registry()
    for name in (
        "fit_forecast",
        "generate_forecast",
        "evaluate_forecast",
        "save_forecast_bundle",
        "load_forecast_bundle",
    ):
        assert registry.get(name) is not None


def test_short_series_raises() -> None:
    frame = _frame(n=12)
    session = (
        Session.ingest(frame)
        .set_roles({"clock": "time", "sales": "target"})
        .time_split(test_size=3)
    )
    with pytest.raises(ValidationError, match="max\\(lags\\)"):
        session.fit_forecast(method="lag_ridge", lags=[1, 2, 3, 7, 14])
