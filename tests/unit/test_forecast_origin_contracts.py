"""Forecast origins must advance with observed history without learning future targets."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.forecasting.backends import industry_one_step
from buildml.forecasting.predict import rolling_one_step_predictions


@pytest.mark.parametrize("strategy", ["rolling_one_step", "rolling_origin", "origin"])
def test_random_walk_arima_matches_independent_origin_formula(strategy):
    pytest.importorskip("statsmodels")
    y = np.r_[np.arange(32, dtype=float), [100., 90., 120., 110., 150., 140., 170., 160.]]
    session = (Session.ingest(pd.DataFrame({"time": pd.date_range("2024-01-01", periods=40), "y": y}))
               .set_roles({"time": "time", "y": "target"}).time_split(test_size=.2))
    session.forecast.fit(method="arima", order=(0, 1, 0), horizon=2)
    plan = session.forecast_plan
    original_result = plan.industry_estimator_["result"]
    params = original_result.params.copy()
    report = session.forecast.evaluate(partition="test", strategy=strategy)
    if strategy == "rolling_one_step":
        expected = np.r_[31., y[32:-1]]
    elif strategy == "rolling_origin":
        expected = np.repeat(np.r_[31., y[33:39:2]], 2)
    else:
        expected = np.repeat(31., 8)
    np.testing.assert_allclose(report.predictions, expected, atol=1e-6)
    assert plan.industry_estimator_["result"] is original_result
    np.testing.assert_array_equal(original_result.params, params)
    assert original_result.nobs == 32
    if strategy == "rolling_one_step":
        changed = y[32:].copy()
        changed[4:] += 10000
        predictions, _ = rolling_one_step_predictions(plan, list(y[:32]), changed)
        np.testing.assert_allclose(predictions[:5], report.predictions[:5], atol=1e-6)


def test_ets_uses_prior_actuals_with_frozen_smoothing_coefficients():
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    train = np.arange(1., 21.)
    fitted = ExponentialSmoothing(train, trend="add", initialization_method="known",
                                 initial_level=0., initial_trend=1.).fit(
                                     smoothing_level=.5, smoothing_trend=.25, optimized=False)
    payload = {"kind": "ets", "result": fitted}
    history = [*train, 100., 40.]
    # Holt's additive recurrences with the original, fixed coefficients.
    level, trend = 0., 1.
    for actual in history:
        next_level = .5 * actual + .5 * (level + trend)
        trend = .25 * (next_level - level) + .75 * trend
        level = next_level
    prediction = industry_one_step(payload, history)
    assert prediction == pytest.approx(level + trend)
    assert payload["result"] is fitted
    assert fitted.params["smoothing_level"] == .5
    assert fitted.model.nobs == 20


@pytest.mark.parametrize("method", ["arima", "sarimax"])
@pytest.mark.parametrize("strategy", ["rolling_one_step", "rolling_origin", "origin"])
def test_exogenous_random_walk_conditions_on_validation_and_prior_holdout(method, strategy):
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(371)
    x = rng.normal(size=60)
    y = 2 * x + np.cumsum(rng.normal(size=60))
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=60), "x": x, "y": y})
    session = (Session.ingest(frame).set_roles({"time": "time", "x": "feature", "y": "target"})
               .time_split(test_size=.2, validation_size=.2))
    options = {"seasonal_order": (0, 0, 0, 0)} if method == "sarimax" else {}
    session.forecast.fit(method=method, order=(0, 1, 0), exog_columns=["x"], horizon=3, **options)
    fitted = session.forecast_plan.industry_estimator_["result"]
    n_train = len(session.partition("train"))
    beta = fitted.params[0]
    report = session.forecast.evaluate(partition="test", strategy=strategy)
    expected = []
    for step in range(12):
        offset = step if strategy == "rolling_one_step" else (step // 3 * 3 if strategy == "rolling_origin" else 0)
        previous = 47 + offset
        expected.append(beta * x[48 + step] + y[previous] - beta * x[previous])
    np.testing.assert_allclose(report.predictions, expected, atol=1e-5)
    assert fitted.nobs == n_train


@pytest.mark.parametrize("damped", [False, True])
def test_seasonal_ets_advances_fixed_states_without_estimating_new_coefficients(damped):
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    train = np.arange(1., 25.) + np.tile([1., -.5, -.5], 8)
    phi = .8 if damped else 1.
    options = {"damping_trend": phi} if damped else {}
    fitted = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=3,
                                 damped_trend=damped, initialization_method="known",
                                 initial_level=0., initial_trend=1., initial_seasonal=[1., -.5, -.5]).fit(
                                     smoothing_level=.5, smoothing_trend=.25,
                                     smoothing_seasonal=.2, optimized=False, **options)
    history = [*train, 80., 50.]
    level, trend, seasons = 0., 1., [1., -.5, -.5]
    for index, actual in enumerate(history):
        slot = index % 3
        projected = level + phi * trend
        next_level = .5 * (actual - seasons[slot]) + .5 * projected
        trend = .25 * (next_level - level) + .75 * phi * trend
        seasons[slot] = .2 * (actual - projected) + .8 * seasons[slot]
        level = next_level
    expected = level + phi * trend + seasons[len(history) % 3]
    assert industry_one_step({"kind": "ets", "result": fitted}, history) == pytest.approx(expected)
    assert fitted.params["smoothing_seasonal"] == .2
