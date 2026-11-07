"""Agent temporal splits and forecasts respect independently derived prefixes."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def run(session, name, **arguments):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, arguments, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


@pytest.mark.parametrize('method', ['naive', 'mean', 'drift', 'seasonal_naive'])
@pytest.mark.parametrize('validation', [None, .2])
def test_agent_split_unsorted_dates_and_forecast_from_training_prefix(method, validation):
    t = np.arange(80)
    chronological = pd.DataFrame({'clock': pd.date_range('2024-01-01', periods=80, tz='UTC'),
                                   'y': 17+.5*t+np.sin(t/3)})
    frame = chronological.sample(frac=1, random_state=13).reset_index(drop=True)
    s = Session.ingest(frame).set_roles({'clock': 'time', 'y': 'target'})
    snapshot = s.to_pandas().copy(deep=True)
    run(s, 'time_split', time_column='clock', test_size=.2, validation_size=validation)
    plan = s.split_plan
    assert set(plan.train_indices).isdisjoint(plan.test_indices)
    assert set(plan.train_indices).isdisjoint(plan.validation_indices)
    assert set(plan.validation_indices).isdisjoint(plan.test_indices)
    training = s.partition('train').sort_values('clock')
    test = s.partition('test').sort_values('clock')
    assert training.clock.max() < test.clock.min()
    assert test.clock.tolist() == chronological.clock.iloc[-16:].tolist()
    pd.testing.assert_series_equal(training.y.reset_index(drop=True),
                                   chronological.y.iloc[:len(training)].reset_index(drop=True))
    if validation is not None:
        middle = s.partition('validation').sort_values('clock')
        assert training.clock.max() < middle.clock.min() < test.clock.min()
    history = training.y.to_numpy()
    run(s, 'fit_forecast', method=method, horizon=5, seasonal_period=7)
    result = run(s, 'generate_forecast', horizon=5, origin='train_end')
    if method == 'naive':
        expected = np.repeat(history[-1], 5)
    elif method == 'mean':
        expected = np.repeat(np.mean(history), 5)
    elif method == 'drift':
        expected = history[-1]+np.arange(1, 6)*(history[-1]-history[0])/(len(history)-1)
    else:
        expected = history[-7:][:5]
    np.testing.assert_allclose(result.predictions, expected, atol=1e-10)
    pd.testing.assert_frame_equal(s.to_pandas(), snapshot)
