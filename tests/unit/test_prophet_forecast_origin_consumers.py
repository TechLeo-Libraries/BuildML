"""Real Prophet forecasts advance their clock while fitted parameters stay fixed."""

import copy
import hashlib
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.forecasting.backends import industry_one_step, industry_predict
from buildml.forecasting.predict import generate_forecast

pytest.importorskip("prophet", reason="real Prophet backend not installed")

FAMILIES = ("linear", "weekly", "constant", "yearly")
BACKINGS = (
    "pandas",
    "pandas-parquet",
    "polars-memory",
    "polars-lazy",
    "duckdb-memory",
    "duckdb-lazy",
)


def _signal(family, positions):
    positions = np.asarray(positions)
    if family == "constant":
        return np.full(len(positions), 8.0)
    if family == "weekly":
        return 10.0 + 0.2 * positions + 2 * np.sin(2 * np.pi * positions / 7)
    return 4.0 + 0.7 * positions


def _frame(family):
    n_train = 370 if family == "yearly" else 48
    n = n_train + 24
    frame = pd.DataFrame(
        {
            "clock": pd.date_range("2024-01-01", periods=n),
            "target": _signal(family, np.arange(n)),
            "row_id": [f"record-{i}" for i in range(n)],
        }
    )
    frame.index = [f"row-{i // 2}" for i in range(n)]
    return frame, n_train


def _session(frame, n_train, tmp_path, backing="pandas"):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith(("parquet", "lazy")):
        path = tmp_path / "rows.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    else:
        session = Session.ingest(frame, engine=engine)
    session.set_roles({"clock": "time", "target": "target", "row_id": "id"})
    return session.inject_split(
        train_indices=list(range(n_train)),
        validation_indices=list(range(n_train, n_train + 12)),
        test_indices=list(range(n_train + 12, n_train + 24)),
    )


@pytest.fixture(scope="module", params=FAMILIES)
def fitted(request, tmp_path_factory):
    family = request.param
    frame, n_train = _frame(family)
    directory = tmp_path_factory.mktemp("prophet_" + family)
    session = _session(frame, n_train, directory)
    seasonal_period = 7 if family in {"linear", "weekly"} else 30
    session.forecast.fit(method="prophet", horizon=5, seasonal_period=seasonal_period)
    assert session.forecast_plan.backend == "prophet"
    assert session.forecast_plan.n_train_rows == n_train
    assert len(session.forecast_plan.industry_estimator_["result"].history) == n_train
    bundle = session.forecast.save_bundle(directory / "bundle")
    return family, frame, n_train, bundle


def _reference(plan, start, count):
    # Explicit clock positions supplied directly to the real fitted consumer.
    # No BuildML forecast-generation helper determines these dates.
    dates = [datetime(2000, 1, 1) + timedelta(days=start + i) for i in range(count)]
    return (
        plan.industry_estimator_["result"].predict(pd.DataFrame({"ds": dates}))["yhat"].to_numpy()
    )


def _snapshot(plan):
    model = plan.industry_estimator_["result"]
    return (
        copy.deepcopy(plan.to_dict()),
        copy.deepcopy(model.params),
        model.history.copy(deep=True),
        model.history_dates.copy(deep=True),
        model.changepoints_t.copy(),
    )


def _unchanged(plan, before):
    metadata, params, history, dates, changepoints = before
    model = plan.industry_estimator_["result"]
    assert plan.to_dict() == metadata
    for name, value in params.items():
        np.testing.assert_array_equal(model.params[name], value)
    pd.testing.assert_frame_equal(model.history, history)
    pd.testing.assert_series_equal(model.history_dates, dates)
    np.testing.assert_array_equal(model.changepoints_t, changepoints)


def _files(bundle):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.is_file()
    }


@pytest.mark.parametrize("origin", ("train_end", "validation_end", "test_end"))
@pytest.mark.parametrize("backing", BACKINGS)
def test_saved_generation_uses_the_selected_clock_origin(fitted, origin, backing, tmp_path):
    family, frame, n_train, bundle = fitted
    session = _session(frame, n_train, tmp_path, backing)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before, files = _snapshot(plan), _files(bundle)
    data, split = session.dataset.to_pandas().copy(deep=True), joblib.hash(session.split_plan)
    offset = n_train + {"train_end": 0, "validation_end": 12, "test_end": 24}[origin]
    expected = _reference(plan, offset, 4)
    result = session.forecast.generate(origin=origin, horizon=np.int64(4))
    assert result.origin == origin and result.horizon == len(result.predictions) == 4
    np.testing.assert_allclose(result.predictions, expected, atol=1e-10)
    # A separate numerical sanity check on known trend/seasonal processes.
    np.testing.assert_allclose(
        result.predictions, _signal(family, np.arange(offset, offset + 4)), atol=0.3, rtol=0
    )
    assert any("synthetic daily" in note for note in plan.disclosures)
    _unchanged(plan, before)
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), data)
    assert joblib.hash(session.split_plan) == split and _files(bundle) == files


@pytest.mark.parametrize("partition", ("validation", "test"))
@pytest.mark.parametrize("strategy", ("origin", "rolling_origin", "rolling_one_step"))
@pytest.mark.parametrize("backing", BACKINGS)
def test_saved_evaluation_scores_each_holdout_clock_once(
    fitted, partition, strategy, backing, tmp_path
):
    _, frame, n_train, bundle = fitted
    session = _session(frame, n_train, tmp_path, backing)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before, files = _snapshot(plan), _files(bundle)
    data = session.dataset.to_pandas().copy(deep=True)
    start = n_train + (12 if partition == "test" else 0)
    expected = _reference(plan, start, 12)
    actuals = frame.target.iloc[start : start + 12].to_numpy()
    report = session.forecast.evaluate(partition=partition, strategy=strategy)
    np.testing.assert_allclose(report.predictions, expected, atol=1e-10)
    np.testing.assert_array_equal(report.actuals, actuals)
    assert report.n_points == 12
    errors = expected - actuals
    assert report.metrics["mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert report.metrics["rmse"] == pytest.approx(np.sqrt(np.mean(errors * errors)))
    assert report.metrics["mape"] == pytest.approx(np.mean(np.abs(errors / actuals)) * 100)
    _unchanged(plan, before)
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), data)
    assert _files(bundle) == files


@pytest.mark.parametrize("history_kind", ("default", "train", "extended", "poisoned", "prefix"))
def test_direct_clock_alignment_does_not_refit_from_history_values(fitted, history_kind, tmp_path):
    _, frame, n_train, bundle = fitted
    session = _session(frame, n_train, tmp_path)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before = _snapshot(plan)
    start = (
        n_train
        if history_kind in {"default", "train"}
        else 8
        if history_kind == "prefix"
        else n_train + 3
    )
    history = None if history_kind == "default" else frame.target.iloc[:start].tolist()
    if history_kind == "poisoned":
        history = [1e12] * start
    expected = _reference(plan, start, 4)
    actual = industry_predict(plan.industry_estimator_, steps=4, history=history)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(
        generate_forecast(plan, horizon=4, history=history).predictions, expected, atol=1e-10
    )
    if history is not None:
        assert industry_one_step(plan.industry_estimator_, history) == pytest.approx(
            expected[0], abs=1e-10
        )
    _unchanged(plan, before)


@pytest.mark.parametrize("history", ([np.nan], [np.inf], [], ["bad"]))
def test_invalid_explicit_history_preserves_fitted_parameters(fitted, history, tmp_path):
    _, frame, n_train, bundle = fitted
    session = _session(frame, n_train, tmp_path)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before = _snapshot(plan)
    with pytest.raises(ValidationError):
        generate_forecast(plan, horizon=4, history=history)
    _unchanged(plan, before)
