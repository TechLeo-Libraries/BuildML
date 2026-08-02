"""Phase R3 forecasting upgrade: industry backends, bundle v2, rolling_origin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import MissingExtraError
from buildml.forecasting.catalog import list_forecast_methods, resolve_default_method
from buildml.forecasting.checkpoint import BUNDLE_FORMAT_V2, load_forecast_bundle, save_forecast_bundle
from buildml.forecasting.extras import statsmodels_available
from buildml.forecasting.fit import fit_forecaster


def _frame(n: int = 120, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-01", periods=n, freq="D")
    y = 8 + 0.03 * np.arange(n) + np.sin(np.arange(n) / 6.0) + rng.normal(0, 0.15, n)
    return pd.DataFrame({"clock": t, "sales": y})


def test_auto_method_resolves() -> None:
    resolved = resolve_default_method("auto")
    if statsmodels_available():
        assert resolved == "ets"
    else:
        assert resolved == "lag_ridge"


def test_lag_ridge_still_works(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"clock": "time", "sales": "target"})
        .time_split(test_size=0.2, validation_size=0.2)
    )
    fit = session.fit_forecast(method="lag_ridge", lags=[1, 2, 3], horizon=5)
    assert fit.method == "lag_ridge"
    gen = session.generate_forecast(horizon=5)
    assert len(gen.predictions) == 5
    ev = session.evaluate_forecast(partition="validation", strategy="rolling_origin")
    assert ev.metrics["rmse"] >= 0.0

    plan = session.forecast_plan
    assert plan is not None
    out = save_forecast_bundle(tmp_path / "fc", plan, fit_result=fit, eval_result=ev)
    meta = (out / "meta.json").read_text(encoding="utf-8")
    assert BUNDLE_FORMAT_V2 in meta
    loaded = load_forecast_bundle(out)
    assert loaded.method == "lag_ridge"


@pytest.mark.skipif(not statsmodels_available(), reason="statsmodels not installed")
def test_ets_forecast_when_statsmodels() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"clock": "time", "sales": "target"})
        .time_split(test_size=0.2)
    )
    fit = session.fit_forecast(method="ets", horizon=7, seasonal_period=7)
    assert fit.method == "ets"
    plan = session.forecast_plan
    assert plan is not None
    assert plan.backend == "statsmodels"
    gen = session.generate_forecast(horizon=7)
    assert len(gen.predictions) == 7
    metrics = session.evaluate_forecast(partition="test", strategy="rolling_one_step")
    assert metrics.n_points > 0


def test_industry_method_raises_without_extra() -> None:
    if statsmodels_available():
        pytest.skip("statsmodels installed")
    session = (
        Session.ingest(_frame())
        .set_roles({"clock": "time", "sales": "target"})
        .time_split(test_size=0.2)
    )
    with pytest.raises(MissingExtraError):
        session.fit_forecast(method="ets")


def test_catalog_and_ai_tools() -> None:
    methods = list_forecast_methods()
    names = {row["method"] for row in methods}
    assert "lag_ridge" in names
    assert "ets" in names
    registry = build_default_registry()
    spec = registry.get("fit_forecast")
    assert spec is not None
    enum = spec.parameters["properties"]["method"]["enum"]
    assert "auto" in enum
    assert "ets" in enum
    ev = registry.get("evaluate_forecast")
    assert ev is not None
    assert "rolling_origin" in ev.parameters["properties"]["strategy"]["enum"]


def test_low_level_auto_fit() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"clock": "time", "sales": "target"})
        .time_split(test_size=0.2)
    )
    plan, fit = fit_forecaster(
        session.dataset,
        session.split_plan,
        method="auto",
        horizon=3,
    )
    assert plan.horizon == 3
    assert fit.n_train_rows > 0
