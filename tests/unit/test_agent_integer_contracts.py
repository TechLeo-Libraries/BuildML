"""Malformed agent budgets cannot be silently truncated into executable calls."""
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _session():
    return (Session.ingest(pd.DataFrame({'x': np.arange(60), 'y': [0, 1] * 30}))
            .set_roles({'y': 'target'}).split(test_size=.2, random_state=2))


@pytest.mark.parametrize('tool,key,args', [
    ('evolutionary_search', 'population_size', {'estimator': 'DecisionTreeClassifier', 'param_space': {'max_depth': [1, 2]}}),
    ('run_automl', 'n_trials', {}),
    ('fit_torch', 'epochs', {}),
    ('fit_clusters', 'n_clusters', {}),
    ('fit_forecast', 'horizon', {}),
])
@pytest.mark.parametrize('value', [True, 2.5, float('nan'), float('inf'), '3', None])
def test_non_integer_arguments_refused_without_session_mutation(tool, key, args, value):
    session = _session()
    before = session.to_pandas().copy(deep=True)
    history = tuple(session.history)
    split = session.split_plan
    registry = build_default_registry()
    proposal = propose_tool_execution(tool, {**args, key: value}, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert not result.executed
    assert f'{key} must be an integer' in result.error
    assert not result.state_changes
    assert session.split_plan is split
    assert tuple(session.history) == history
    pd.testing.assert_frame_equal(session.to_pandas(), before)


def test_real_agent_evolutionary_search_matches_direct_budget_and_ranking():
    agent, direct = _session(), _session()
    registry = build_default_registry()
    args = dict(estimator='DecisionTreeClassifier', param_space={'max_depth': [1, 2, 3]},
                population_size=4, n_generations=3, max_evaluations=5,
                cv=2, refit=False, ranking_metric='accuracy')
    proposal = propose_tool_execution('evolutionary_search', args, registry)
    result = execute_tool(agent, proposal, confirmed=True, registry=registry)
    assert result.executed, result.error
    expected = direct.evolutionary_search(DecisionTreeClassifier(), **{
        key: value for key, value in args.items() if key != 'estimator'
    }, random_state=0)
    assert result.result.best_score == pytest.approx(expected.best_score)
    assert result.result.best_params == expected.best_params
    assert 1 <= len(result.result.trials) <= 5
    assert result.result.refit_result is None
