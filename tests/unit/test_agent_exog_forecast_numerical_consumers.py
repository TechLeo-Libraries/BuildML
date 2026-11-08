"""Agent exogenous forecasts match normal equations and independent backtest histories."""

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, arguments):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(name, arguments, registry),
                          confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


def _predict(history, exog, lags, coefficients):
    return float(np.array([1.] + [history[-lag] for lag in lags] + list(exog)) @ coefficients)


@pytest.mark.parametrize('lags', [[1], [1, 3]])
@pytest.mark.parametrize('columns', [['a'], ['b', 'a']])
@pytest.mark.parametrize('partition', ['validation', 'test'])
@pytest.mark.parametrize('strategy', ['origin', 'rolling_one_step', 'rolling_origin'])
def test_agent_exog_generation_and_evaluation_match_ridge_oracles(lags, columns, partition, strategy, tmp_path):
    rng = np.random.default_rng(412)
    n = 80
    a, b = rng.normal(size=(2, n))
    y = np.zeros(n)
    y[:3] = [10, 12, 11]
    for i in range(3, n):
        y[i] = 5 + .4*y[i-1] - .1*y[i-3] + 1.2*a[i] - .7*b[i]
    frame = pd.DataFrame({'clock': pd.date_range('2024-01-01', periods=n, tz='UTC'),
                          'y': y, 'a': a, 'b': b})
    shuffled = frame.sample(frac=1, random_state=6).reset_index(drop=True)
    s = Session.ingest(shuffled).set_roles({'clock': 'time', 'y': 'target', 'a': 'feature', 'b': 'feature'})
    _run(s, 'time_split', {'time_column': 'clock', 'test_size': .2, 'validation_size': .2})
    before = s.to_pandas().copy(deep=True)
    training = s.partition('train').sort_values('clock')
    assert training.clock.tolist() == frame.clock.iloc[:51].tolist()
    x = np.array([[1.] + [y[i-lag] for lag in lags] + frame.loc[i, columns].tolist()
                  for i in range(max(lags), 51)])
    target = y[max(lags):51]
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ target)
    _run(s, 'fit_forecast', {'method': 'lag_ridge', 'horizon': 5, 'lags': lags, 'exog_columns': columns})
    plan = s.forecast_plan
    np.testing.assert_allclose(plan.estimator_.coef_, coefficients[1:], atol=1e-10)
    assert plan.estimator_.intercept_ == pytest.approx(coefficients[0])
    fingerprint = joblib.hash(plan)
    future = frame.loc[51:55, columns].to_numpy()
    generated = _run(s, 'generate_forecast', {'horizon': 5, 'future_exog': future.tolist()})
    history = y[:51].tolist()
    expected = []
    for row in future:
        predicted = _predict(history, row, lags, coefficients)
        expected.append(predicted)
        history.append(predicted)
    np.testing.assert_allclose(generated.predictions, expected, atol=1e-10)
    start = 51 if partition == 'validation' else 64
    size = 13 if partition == 'validation' else 16
    history = y[:start].tolist()
    expected_eval = []
    for offset in range(0, size, 5 if strategy == 'rolling_origin' else size):
        length = min(5 if strategy == 'rolling_origin' else size, size-offset)
        window = history.copy()
        for j in range(length):
            i = start + offset + j
            predicted = _predict(window, frame.loc[i, columns].to_numpy(), lags, coefficients)
            expected_eval.append(predicted)
            window.append(y[i] if strategy == 'rolling_one_step' else predicted)
        history.extend(y[start+offset:start+offset+length])
    evaluated = _run(s, 'evaluate_forecast', {'partition': partition, 'strategy': strategy})
    np.testing.assert_allclose(evaluated.predictions, expected_eval, atol=1e-10)
    truth = y[start:start+size]
    np.testing.assert_allclose(evaluated.actuals, truth)
    assert evaluated.metrics['mae'] == pytest.approx(np.abs(np.array(expected_eval)-truth).mean())
    assert evaluated.metrics['rmse'] == pytest.approx(np.sqrt(np.mean((np.array(expected_eval)-truth)**2)))
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(s.to_pandas(), before)
    path = str(tmp_path / 'forecast')
    _run(s, 'save_forecast_bundle', {'path': path})
    restored = Session.ingest(shuffled).set_roles({'clock': 'time', 'y': 'target', 'a': 'feature', 'b': 'feature'})
    _run(restored, 'load_forecast_bundle', {'path': path, 'trusted': True})
    replay = _run(restored, 'generate_forecast', {'horizon': 5, 'future_exog': future.tolist()})
    np.testing.assert_allclose(replay.predictions, expected, atol=1e-10)


@pytest.mark.parametrize('future', [[], [[1.]], [[1., 2.]]*3, [[1.], [2., 3.], [4.]], [['bad']]*3, [True]*3, [[True]]*3, [[float('nan')]]*3, [[float('inf')]]*3])
def test_agent_invalid_future_exog_refuses_without_state_mutation(future):
    frame = pd.DataFrame({'clock': pd.date_range('2024-01-01', periods=40),
                          'y': np.arange(40.) + 10, 'a': np.cos(np.arange(40.))})
    s = Session.ingest(frame).set_roles({'clock': 'time', 'y': 'target', 'a': 'feature'})
    _run(s, 'time_split', {'time_column': 'clock', 'test_size': .2})
    _run(s, 'fit_forecast', {'method': 'lag_ridge', 'lags': [1], 'exog_columns': ['a'], 'horizon': 3})
    before, history = s.to_pandas().copy(deep=True), list(s._history)
    fingerprint = joblib.hash(s.forecast_plan)
    registry = build_default_registry()
    result = execute_tool(s, propose_tool_execution('generate_forecast',
                          {'horizon': 3, 'future_exog': future}, registry), confirmed=True, registry=registry)
    assert not result.executed and result.error
    assert result.state_changes == () and s._forecast_generate_result is None
    assert s._history == history and joblib.hash(s.forecast_plan) == fingerprint
    pd.testing.assert_frame_equal(s.to_pandas(), before)
