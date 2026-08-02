"""Unit coverage for the classical forecasting thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.forecasting.checkpoint import BUNDLE_FORMAT, load_forecast_bundle


def _series_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    y = 8.0 + 0.04 * np.arange(n) + np.sin(np.arange(n) / 7.0) + rng.normal(0, 0.25, n)
    promo = (np.arange(n) % 14 < 3).astype(float)
    return pd.DataFrame({"ts": t, "y": y, "promo": promo})


def _ready_session(*, validation: bool = True, n: int = 100) -> Session:
    kwargs: dict[str, object] = {"test_size": 0.2}
    if validation:
        kwargs["validation_size"] = 0.2
    return (
        Session.ingest(_series_frame(n=n))
        .set_roles({"ts": "time", "y": "target", "promo": "feature"})
        .time_split(**kwargs)
    )


def test_core_import_does_not_require_extra() -> None:
    import buildml.forecasting as fc

    assert hasattr(Session, "fit_forecast")
    assert hasattr(Session, "evaluate_forecast")
    assert hasattr(fc, "fit_forecaster")


def test_catalog_covers_forecast_operations() -> None:
    for name in (
        "fit_forecast",
        "generate_forecast",
        "evaluate_forecast",
        "save_forecast_bundle",
        "load_forecast_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert "forecast-temporal-leakage" in OPERATION_CATALOG["fit_forecast"].concept_links
    assert "forecast-lag-features" in OPERATION_CATALOG["fit_forecast"].concept_links
    assert "forecast-eval-protocols" in OPERATION_CATALOG["evaluate_forecast"].concept_links
    assert (
        "forecast-bundle-boundary"
        in OPERATION_CATALOG["save_forecast_bundle"].concept_links
    )


def test_fit_requires_split() -> None:
    session = Session.ingest(_series_frame()).set_roles(
        {"ts": "time", "y": "target", "promo": "ignore"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_forecast(method="naive")


def test_refuses_random_split() -> None:
    session = (
        Session.ingest(_series_frame())
        .set_roles({"ts": "time", "y": "target", "promo": "ignore"})
        .split(test_size=0.2, random_state=0)
    )
    with pytest.raises(LeakageError, match="refuses split kind"):
        session.fit_forecast(method="lag_ridge", lags=[1, 2, 3])


def test_lag_ridge_fit_generate_evaluate_and_bundle(tmp_path: Path) -> None:
    session = _ready_session()
    fit = session.fit_forecast(
        method="lag_ridge",
        horizon=5,
        lags=[1, 2, 3, 7],
        alpha=1.0,
    )
    assert fit.method == "lag_ridge"
    assert fit.univariate is True
    assert fit.n_fit_rows < fit.n_train_rows
    assert session.forecast_plan is not None

    gen = session.generate_forecast(horizon=5)
    assert len(gen.predictions) == 5

    metrics = session.evaluate_forecast(
        partition="test", strategy="rolling_one_step"
    )
    assert metrics.n_points > 0
    assert "mae" in metrics.metrics
    assert "rmse" in metrics.metrics
    assert "mape" in metrics.metrics
    assert metrics.metrics["mae"] >= 0.0

    path = session.save_forecast_bundle(tmp_path / "fc")
    assert (path / "meta.json").is_file()
    assert (path / "forecast_plan.joblib").is_file()
    plan = load_forecast_bundle(path)
    assert plan.method == "lag_ridge"
    assert plan.lags == session.forecast_plan.lags

    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"ts": "time", "y": "target", "promo": "feature"})
        .time_split(test_size=0.2, validation_size=0.2)
    )
    restored.load_forecast_bundle(path)
    again = restored.generate_forecast(horizon=5)
    assert again.predictions == gen.predictions

    with pytest.raises(ValidationError, match=BUNDLE_FORMAT):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "meta.json").write_text(
            '{"format": "buildml.rag_bundle.v1"}', encoding="utf-8"
        )
        (bad / "forecast_plan.joblib").write_bytes(b"x")
        load_forecast_bundle(bad)


def test_seasonal_naive_and_baselines() -> None:
    session = _ready_session()
    fit = session.fit_forecast(
        method="seasonal_naive", seasonal_period=7, horizon=7
    )
    assert fit.method == "seasonal_naive"
    gen = session.generate_forecast(horizon=7)
    assert len(gen.predictions) == 7
    # Seasonal naive repeats the last season block.
    assert gen.predictions[0] == pytest.approx(session.forecast_plan.seasonal_history_[0])
    metrics = session.evaluate_forecast(partition="validation")
    assert "mae" in metrics.metrics

    for method in ("naive", "mean", "drift"):
        s = _ready_session()
        s.fit_forecast(method=method, horizon=3)  # type: ignore[arg-type]
        out = s.generate_forecast(horizon=3)
        assert len(out.predictions) == 3


def test_exog_requires_future_for_generate() -> None:
    session = _ready_session()
    session.fit_forecast(
        method="lag_ridge",
        lags=[1, 2, 3],
        exog_columns=["promo"],
        horizon=4,
    )
    assert session.forecast_plan is not None
    assert session.forecast_plan.univariate is False
    with pytest.raises(ValidationError, match="future_exog"):
        session.generate_forecast(horizon=4)
    future = np.zeros((4, 1), dtype=float)
    gen = session.generate_forecast(horizon=4, future_exog=future)
    assert len(gen.predictions) == 4
    # rolling eval can use holdout exog
    metrics = session.evaluate_forecast(partition="test", strategy="rolling_one_step")
    assert metrics.n_points > 0


def test_origin_strategy_and_explain() -> None:
    session = _ready_session()
    session.fit_forecast(method="lag_ridge", lags=[1, 2, 3], horizon=5)
    origin = session.evaluate_forecast(partition="test", strategy="origin")
    assert origin.strategy == "origin"
    assert origin.n_points == len(origin.predictions)
    before = session.explain("fit_forecast", moment="before")
    assert before.operation == "fit_forecast"
    assert before.prerequisite_status.get("split") is True
