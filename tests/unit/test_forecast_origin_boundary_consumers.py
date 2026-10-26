"""Forecast origins, refused inputs, and bundle replay have observable contracts."""

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.forecasting.predict import generate_forecast


def _session(frame=None):
    if frame is None:
        t = np.arange(40.)
        frame = pd.DataFrame({"clock": pd.date_range("2024-01-01", periods=40),
                              "y": 4 + .5*t + np.cos(t), "x": np.sin(t)})
    return (Session.ingest(frame).set_roles({"clock": "time", "y": "target", "x": "feature"})
            .time_split(test_size=.2, validation_size=.2))


def _baseline(method, history, training, horizon):
    if method == "mean":
        return np.repeat(np.mean(training), horizon)
    if method == "naive":
        return np.repeat(history[-1], horizon)
    if method == "drift":
        slope = (training[-1] - training[0]) / (len(training)-1)
        return history[-1] + slope*np.arange(1, horizon+1)
    return np.resize(history[-3:], horizon)


@pytest.mark.parametrize("method", ["naive", "mean", "drift", "seasonal_naive"])
@pytest.mark.parametrize("origin", ["train_end", "validation_end", "test_end"])
@pytest.mark.parametrize("replay", [False, True])
def test_named_origins_match_baseline_formulas_and_trusted_replay(method, origin, replay, tmp_path):
    session = _session()
    session.forecast.fit(method=method, horizon=5, seasonal_period=3)
    if replay:
        session.forecast.save_bundle(tmp_path / "forecast")
        restored = _session()
        restored.forecast.load_bundle(tmp_path / "forecast", trusted=True)
        session = restored
    training = session.partition("train").y.to_numpy()
    parts = ["train"]
    if origin != "train_end":
        parts.append("validation")
    if origin == "test_end":
        parts.append("test")
    history = np.concatenate([session.partition(p).y.to_numpy() for p in parts])
    fingerprint = joblib.hash(session.forecast_plan)
    result = session.forecast.generate(horizon=5, origin=origin)
    np.testing.assert_allclose(result.predictions, _baseline(method, history, training, 5), atol=1e-12)
    assert result.origin == origin and result.horizon == len(result.predictions) == 5
    assert joblib.hash(session.forecast_plan) == fingerprint
    direct = generate_forecast(session.forecast_plan, horizon=5, history=history, origin=origin)
    np.testing.assert_array_equal(direct.predictions, result.predictions)


@pytest.mark.parametrize("method", ["naive", "mean", "drift", "seasonal_naive"])
@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("strategy", ["origin", "rolling_origin", "rolling_one_step"])
def test_evaluation_origins_use_only_available_actuals(method, partition, strategy):
    session = _session()
    session.forecast.fit(method=method, horizon=3, seasonal_period=3)
    training = session.partition("train").y.to_numpy()
    history = list(training)
    if partition == "test":
        history.extend(session.partition("validation").y)
    actuals = session.partition(partition).y.to_numpy()
    step = 1 if strategy == "rolling_one_step" else 3 if strategy == "rolling_origin" else len(actuals)
    expected = []
    for offset in range(0, len(actuals), step):
        length = min(step, len(actuals)-offset)
        expected.extend(_baseline(method, np.asarray(history), training, length))
        history.extend(actuals[offset:offset+length])
    report = session.forecast.evaluate(partition=partition, strategy=strategy)
    np.testing.assert_allclose(report.predictions, expected, atol=1e-12)
    assert report.metrics["mae"] == pytest.approx(np.abs(expected-actuals).mean())
    assert report.metrics["rmse"] == pytest.approx(np.sqrt(np.mean((expected-actuals)**2)))


@pytest.mark.parametrize("method", ["arima", "sarimax"])
@pytest.mark.parametrize("origin", ["train_end", "validation_end", "test_end"])
@pytest.mark.parametrize("replay", [False, True])
def test_exogenous_random_walk_generation_matches_origin_formula(method, origin, replay, tmp_path):
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(947)
    x = rng.normal(size=40)
    y = 2*x + np.cumsum(rng.normal(size=40))
    frame = pd.DataFrame({"clock": pd.date_range("2024-01-01", periods=40), "y": y, "x": x})
    session = _session(frame)
    options = {"seasonal_order": (0, 0, 0, 0)} if method == "sarimax" else {}
    session.forecast.fit(method=method, order=(0, 1, 0), exog_columns=["x"], horizon=4, **options)
    if replay:
        session.forecast.save_bundle(tmp_path / "exog")
        session = _session(frame).forecast.load_bundle(tmp_path / "exog", trusted=True)
    fitted = session.forecast_plan.industry_estimator_["result"]
    params = fitted.params.copy()
    name = origin.removesuffix("_end")
    previous = int((session.partition(name).clock.max()-frame.clock.iloc[0]).days)
    future = pd.DataFrame({"irrelevant": np.arange(4), "x": [.3, -.7, 1.2, 0.]})
    metadata = session.forecast_plan.to_dict()
    endog = fitted.model.endog.copy()
    result = session.forecast.generate(horizon=4, origin=origin, future_exog=future)
    expected = params[0]*future.x.to_numpy() + y[previous]-params[0]*x[previous]
    np.testing.assert_allclose(result.predictions, expected, atol=1e-5)
    np.testing.assert_array_equal(fitted.params, params)
    np.testing.assert_array_equal(fitted.model.endog, endog)
    assert session.forecast_plan.industry_estimator_["result"] is fitted
    assert session.forecast_plan.to_dict() == metadata


INVALID_COUNTS = [True, np.bool_(True), 1.5, "3", 0, -1, float("nan"), float("inf")]


@pytest.mark.parametrize("value", INVALID_COUNTS)
@pytest.mark.parametrize("argument", ["horizon", "lags", "seasonal_period"])
def test_fit_refuses_lossy_counts_without_replacing_the_active_plan(value, argument):
    session = _session()
    session.forecast.fit(method="naive", horizon=3)
    before = list(session._history)
    fingerprint = joblib.hash(session.forecast_plan)
    options = {argument: [1, value] if argument == "lags" else value}
    with pytest.raises(ValidationError, match="integer"):
        session.forecast.fit(method="seasonal_naive", **options)
    assert session._history == before
    assert joblib.hash(session.forecast_plan) == fingerprint


@pytest.mark.parametrize("value", INVALID_COUNTS)
def test_generate_refuses_lossy_horizons_without_replacing_prior_result(value):
    session = _session()
    session.forecast.fit(method="naive", horizon=3)
    prior = session.forecast.generate()
    before = list(session._history)
    with pytest.raises(ValidationError, match="integer"):
        session.forecast.generate(horizon=value)
    assert session._history == before and session.forecast_generate_result is prior


@pytest.mark.parametrize("method", ["naive", "drift", "seasonal_naive", "lag_ridge"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_generation_refuses_nonfinite_origin_history(method, invalid):
    session = _session()
    session.forecast.fit(method=method, lags=[1, 3], seasonal_period=3)
    history = list(session.forecast_plan.last_train_values_)
    history[-1] = invalid
    fingerprint = joblib.hash(session.forecast_plan)
    with pytest.raises(ValidationError, match="finite"):
        generate_forecast(session.forecast_plan, history=history)
    assert joblib.hash(session.forecast_plan) == fingerprint


@pytest.mark.parametrize("method", ["naive", "mean", "drift", "seasonal_naive"])
@pytest.mark.parametrize("partition", ["train", "test"])
def test_nonfinite_targets_refuse_before_fit_or_metrics(method, partition):
    session = _session()
    if partition == "train":
        session.dataset.frame.loc[0, "y"] = float("inf")
        with pytest.raises(ValidationError, match="finite"):
            session.forecast.fit(method=method, seasonal_period=3)
        assert session.forecast_plan is None
    else:
        session.forecast.fit(method=method, seasonal_period=3)
        session.dataset.frame.loc[39, "y"] = float("inf")
        with pytest.raises(ValidationError, match="finite"):
            session.forecast.evaluate(partition="test")
        assert session.forecast_eval_result is None


@pytest.mark.parametrize("method", ["naive", "lag_ridge", "arima"])
def test_injected_validation_cannot_extend_beyond_test_origin(method):
    if method == "arima":
        pytest.importorskip("statsmodels")
    session = _session()
    session.inject_split(train_indices=list(range(20)), validation_indices=list(range(30, 40)),
                         test_indices=list(range(20, 30)))
    with pytest.raises(LeakageError, match="Temporal leakage"):
        session.forecast.fit(method=method, lags=[1])
    assert session.forecast_plan is None


@pytest.mark.parametrize("method", ["naive", "lag_ridge", "arima"])
def test_changed_partition_order_is_rechecked_before_scoring(method):
    if method == "arima":
        pytest.importorskip("statsmodels")
    session = _session()
    session.forecast.fit(method=method, lags=[1])
    session.inject_split(train_indices=list(range(20)), validation_indices=list(range(30, 40)),
                         test_indices=list(range(20, 30)))
    with pytest.raises(LeakageError, match="Temporal leakage"):
        session.forecast.evaluate(partition="test")
    assert session.forecast_eval_result is None


@pytest.mark.parametrize("method", ["lag_ridge", "arima", "sarimax"])
@pytest.mark.parametrize("future", [None, [], [[1.]], [[1., 2.]]*3, [[1.], [2., 3.], [4.]],
                                   [["bad"]]*3, [[float("nan")]]*3, [[float("inf")]]*3,
                                   pd.DataFrame({"wrong": [1., 2., 3.]})])
def test_exogenous_generation_refuses_malformed_inputs_consistently(method, future):
    if method != "lag_ridge":
        pytest.importorskip("statsmodels")
    session = _session()
    options = {"seasonal_order": (0, 0, 0, 0)} if method == "sarimax" else {}
    session.forecast.fit(method=method, lags=[1], order=(0, 1, 0),
                         exog_columns=["x"], horizon=3, **options)
    before = list(session._history)
    metadata = session.forecast_plan.to_dict()
    with pytest.raises(ValidationError, match="future_exog"):
        session.forecast.generate(horizon=3, future_exog=future)
    assert session._history == before and session.forecast_generate_result is None
    assert session.forecast_plan.to_dict() == metadata


@pytest.mark.parametrize("method", ["lag_ridge", "arima", "sarimax"])
@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_nonfinite_exogenous_rows_refuse_fit_or_evaluation(method, partition):
    if method != "lag_ridge":
        pytest.importorskip("statsmodels")
    session = _session()
    options = {"seasonal_order": (0, 0, 0, 0)} if method == "sarimax" else {}
    if partition == "train":
        session.dataset.frame.loc[0, "x"] = float("inf")
        with pytest.raises(ValidationError, match="finite"):
            session.forecast.fit(method=method, exog_columns=["x"], order=(0, 1, 0), **options)
        assert session.forecast_plan is None
    else:
        session.forecast.fit(method=method, exog_columns=["x"], order=(0, 1, 0), **options)
        index = session.split_plan.indices_for(partition)[-1]
        session.dataset.frame.loc[index, "x"] = float("inf")
        with pytest.raises(ValidationError, match="finite"):
            session.forecast.evaluate(partition="test")
        assert session.forecast_eval_result is None


@pytest.mark.parametrize("history", [[], [[1., 2.]], ["bad"]])
def test_invalid_custom_history_has_a_validation_error(history):
    session = _session()
    session.forecast.fit(method="naive")
    with pytest.raises(ValidationError, match="history"):
        generate_forecast(session.forecast_plan, history=history)


def test_numpy_integral_counts_and_normalized_lags_keep_the_requested_forecast():
    session = _session()
    result = session.forecast.fit(method="seasonal_naive", horizon=np.int64(4),
                                  lags=[np.int64(3), np.int64(1), np.int64(3)],
                                  seasonal_period=np.int64(3))
    assert result.horizon == 4 and result.lags == (1, 3)
    training = session.partition("train").y.to_numpy()
    generated = session.forecast.generate(horizon=np.int64(5))
    np.testing.assert_array_equal(generated.predictions, np.resize(training[-3:], 5))
