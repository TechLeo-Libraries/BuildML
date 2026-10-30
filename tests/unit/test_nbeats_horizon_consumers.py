"""Real N-BEATS weights determine each requested forecast position and backtest."""

import copy
import hashlib
import os
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.extras import _subprocess_import_ok
from buildml.forecasting.backends import industry_one_step, industry_predict
from buildml.forecasting.predict import generate_forecast

pytest.importorskip("neuralforecast", reason="real NeuralForecast backend not installed")
import torch  # noqa: E402

BACKINGS = (
    "pandas",
    "pandas-parquet",
    "polars-memory",
    "polars-lazy",
    "duckdb-memory",
    "duckdb-lazy",
)


@contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _frame(family):
    t = np.arange(34)
    y = 3 + 0.4 * t + np.sin(t) if family == "trend" else 7 + 2 * np.cos(t / 2)
    return pd.DataFrame(
        {
            "clock": pd.date_range("2024-01-01", periods=len(t)),
            "target": y,
            "row_id": [f"record-{i}" for i in t],
        }
    )


def _session(frame, directory, backing="pandas"):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith(("parquet", "lazy")):
        path = directory / "rows.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    else:
        session = Session.ingest(frame, engine=engine)
    session.set_roles({"clock": "time", "target": "target", "row_id": "id"})
    return session.inject_split(
        train_indices=list(range(24)),
        validation_indices=list(range(24, 29)),
        test_indices=list(range(29, 34)),
    )


@pytest.fixture(scope="module", autouse=True)
def actual_runtime():
    # A genuine bounded import: cold Torch/Lightning imports exceed 12s on Windows.
    # This also isolates horizon regressions on archived sources with the old timeout.
    assert _subprocess_import_ok("neuralforecast", timeout=90)


@pytest.fixture(scope="module", params=("trend", "oscillation"))
def fitted(request, tmp_path_factory):
    frame = _frame(request.param)
    directory = tmp_path_factory.mktemp("nbeats_" + request.param)
    session = _session(frame, directory)
    with _working_directory(directory):
        session.forecast.fit(
            method="nbeats",
            horizon=3,
            nbeats_input_size=6,
            nbeats_horizon=4,
            max_iter=1,
            random_state=7,
        )
    plan = session.forecast_plan
    assert plan.backend == "neuralforecast" and plan.n_train_rows == 24
    model = plan.industry_estimator_["result"].models[0]
    assert model.h == 4 and model.input_size == 6 and model.hparams["scaler_type"] == "identity"
    bundle = session.forecast.save_bundle(directory / "bundle")
    return frame, bundle


def _reference(plan, history, count):
    # Independent tensor inference, with no BuildML or NeuralForecast predict helper.
    # Only the most recent input window enters the fitted identity-scaled network.
    model = plan.industry_estimator_["result"].models[0]
    model.eval()
    context, predictions = list(history), []
    while len(predictions) < count:
        window = torch.tensor(
            context[-model.input_size :], dtype=torch.float32, device=model.device
        ).reshape(1, -1, 1)
        with torch.inference_mode():
            block = model({"insample_y": window, "insample_mask": torch.ones_like(window)})
        values = block.cpu().reshape(-1).tolist()
        predictions.extend(values)
        context.extend(values)
    values = np.asarray(predictions[:count])
    assert np.isfinite(values).all()
    return values


def _snapshot(plan):
    nf = plan.industry_estimator_["result"]
    return (
        copy.deepcopy(plan.to_dict()),
        {k: v.detach().cpu().clone() for k, v in nf.models[0].state_dict().items()},
        copy.deepcopy(nf.ds),
        copy.deepcopy(nf.last_dates),
        nf.dataset.temporal.detach().cpu().clone(),
    )


def _unchanged(plan, before):
    metadata, weights, dates, last_dates, training = before
    nf = plan.industry_estimator_["result"]
    assert plan.to_dict() == metadata
    for name, value in weights.items():
        torch.testing.assert_close(
            nf.models[0].state_dict()[name].detach().cpu(), value, rtol=0, atol=0
        )
    np.testing.assert_array_equal(nf.ds, dates)
    pd.testing.assert_index_equal(nf.last_dates, last_dates)
    torch.testing.assert_close(nf.dataset.temporal.detach().cpu(), training, rtol=0, atol=0)


def _files(bundle):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.is_file()
    }


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("origin", ("train_end", "validation_end", "test_end"))
@pytest.mark.parametrize("horizon", (1, 2, 4, 7, 9))
def test_saved_horizons_start_at_the_next_position(fitted, backing, origin, horizon, tmp_path):
    frame, bundle = fitted
    session = _session(frame, tmp_path, backing)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before, files = _snapshot(plan), _files(bundle)
    data, split = session.dataset.to_pandas().copy(deep=True), joblib.hash(session.split_plan)
    end = {"train_end": 24, "validation_end": 29, "test_end": 34}[origin]
    expected = _reference(plan, frame.target.iloc[:end].tolist(), horizon)
    result = session.forecast.generate(origin=origin, horizon=np.int64(horizon))
    assert result.horizon == len(result.predictions) == horizon and result.origin == origin
    np.testing.assert_allclose(result.predictions, expected, atol=3e-5, rtol=3e-6)
    _unchanged(plan, before)
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), data)
    assert joblib.hash(session.split_plan) == split and _files(bundle) == files


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("partition", ("validation", "test"))
@pytest.mark.parametrize("strategy", ("origin", "rolling_origin", "rolling_one_step"))
def test_backtests_use_only_preceding_observations(fitted, backing, partition, strategy, tmp_path):
    frame, bundle = fitted
    session = _session(frame, tmp_path, backing)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before, files = _snapshot(plan), _files(bundle)
    data = session.dataset.to_pandas().copy(deep=True)
    start = 24 if partition == "validation" else 29
    width = {"origin": 5, "rolling_origin": 3, "rolling_one_step": 1}[strategy]
    expected = []
    for i in range(0, 5, width):
        expected.extend(
            _reference(plan, frame.target.iloc[: start + i].tolist(), min(width, 5 - i))
        )
    actuals = frame.target.iloc[start : start + 5].to_numpy()
    report = session.forecast.evaluate(partition=partition, strategy=strategy)
    np.testing.assert_allclose(report.predictions, expected, atol=3e-5, rtol=3e-6)
    np.testing.assert_array_equal(report.actuals, actuals)
    assert report.n_points == 5
    errors = np.asarray(expected) - actuals
    assert report.metrics["mae"] == pytest.approx(np.mean(np.abs(errors)), rel=1e-5, abs=3e-5)
    assert report.metrics["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)), rel=1e-5, abs=3e-5)
    assert report.metrics["mape"] == pytest.approx(
        np.mean(np.abs(errors / actuals)) * 100, rel=1e-5
    )
    _unchanged(plan, before)
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), data)
    assert _files(bundle) == files


@pytest.mark.parametrize("history_kind", ("train", "extended", "older_values", "recent_values"))
def test_context_window_and_one_step_use_frozen_weights(fitted, history_kind, tmp_path):
    frame, bundle = fitted
    session = _session(frame, tmp_path)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before = _snapshot(plan)
    history = frame.target.iloc[: 24 if history_kind == "train" else 29].tolist()
    original = _reference(plan, history, 7)
    if history_kind == "older_values":
        history[:-6] = [1e6] * (len(history) - 6)
    if history_kind == "recent_values":
        history[-6:] = [v + 5 for v in history[-6:]]
    expected = _reference(plan, history, 7)
    if history_kind == "older_values":
        np.testing.assert_array_equal(expected, original)
    if history_kind == "recent_values":
        assert not np.allclose(expected, original)
    np.testing.assert_allclose(
        generate_forecast(plan, horizon=7, history=history).predictions,
        expected,
        atol=3e-5,
        rtol=3e-6,
    )
    assert industry_one_step(plan.industry_estimator_, history) == pytest.approx(
        expected[0], rel=3e-6
    )
    _unchanged(plan, before)


def test_real_native_long_horizon_agrees_with_tensor_reference(fitted, tmp_path):
    frame, bundle = fitted
    session = _session(frame, tmp_path)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    nf = plan.industry_estimator_["result"]
    history = frame.target.iloc[:29].tolist()
    df = pd.DataFrame(
        {"unique_id": ["series"] * 29, "ds": pd.date_range("2000-01-01", periods=29), "y": history}
    )
    # NF 3.1.2 misaligns a partial final block in predict(h=9). Ask for complete
    # native blocks before taking the prefix; the bare tensor oracle stays fixed.
    native = nf.predict(df=df, h=12)["NBEATS"].to_numpy()[:9]
    np.testing.assert_allclose(native, _reference(plan, history, 9), atol=3e-5, rtol=3e-6)


def test_default_one_step_fit_replays_a_real_bundle(tmp_path):
    frame = _frame("trend")
    session = _session(frame, tmp_path)
    with _working_directory(tmp_path):
        session.forecast.fit(method="nbeats", nbeats_input_size=6, max_iter=1, random_state=7)
    plan = session.forecast_plan
    assert plan.horizon == plan.industry_estimator_["result"].models[0].h == 1
    expected = _reference(plan, frame.target.iloc[:24].tolist(), 3)
    bundle = session.forecast.save_bundle(tmp_path / "one-step")
    receiver = _session(frame, tmp_path)
    receiver.forecast.load_bundle(bundle, trusted=True)
    before = _snapshot(receiver.forecast_plan)
    np.testing.assert_allclose(
        receiver.forecast.generate(horizon=3).predictions, expected, atol=3e-5, rtol=3e-6
    )
    _unchanged(receiver.forecast_plan, before)


@pytest.mark.parametrize("history", (None, [], [1, 2], [1, 2, 3, 4, 5]))
def test_missing_or_insufficient_context_is_actionable(fitted, history, tmp_path):
    frame, bundle = fitted
    session = _session(frame, tmp_path)
    session.forecast.load_bundle(bundle, trusted=True)
    plan = session.forecast_plan
    before = _snapshot(plan)
    with pytest.raises(ValidationError, match="history|input_size"):
        industry_predict(plan.industry_estimator_, steps=2, history=history)
    _unchanged(plan, before)


@pytest.mark.parametrize("name", ("nbeats_input_size", "nbeats_horizon", "max_iter"))
@pytest.mark.parametrize("value", (0, True, 2.5, "4"))
def test_invalid_native_fit_sizes_are_refused_before_training(name, value, tmp_path):
    session = _session(_frame("trend"), tmp_path)
    kwargs = {"horizon": 4, "nbeats_input_size": 6, "nbeats_horizon": 4, "max_iter": 1, name: value}
    with _working_directory(tmp_path), pytest.raises(ValidationError, match=name):
        session.forecast.fit(method="nbeats", **kwargs)
    assert session.forecast_plan is None
