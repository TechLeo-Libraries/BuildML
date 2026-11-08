"""Agent evolutionary recipe search matches fold-local numerical Ridge oracles."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, arguments):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution('evolutionary_search', arguments, registry),
                          confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


def _session(poison):
    rng = np.random.default_rng(922)
    x = rng.normal(size=(96, 3))*[1, 20, .1]
    y = 3*x[:, 0] - .08*x[:, 1] + 5*x[:, 2] + rng.normal(0, .3, 96)
    x[np.arange(0, 64, 11), 1] = np.nan
    if poison:
        x[64:] = np.nan
        y[64:] = 1e9
    frame = pd.DataFrame(x, columns=['x', 'z', 'q'])
    frame['y'] = y
    s = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'q': 'feature', 'y': 'target'})
    s.inject_split(train_indices=list(range(64)), validation_indices=list(range(64, 80)),
                   test_indices=list(range(80, 96)))
    return s, x, y


def _prediction(train, target, probe, scale, k, alpha):
    medians = np.nanmedian(train, axis=0)
    train = np.where(np.isnan(train), medians, train)
    probe = np.where(np.isnan(probe), medians, probe)
    if scale == 'standard':
        center, divisor = train.mean(0), train.std(0)
    elif scale == 'minmax':
        center, divisor = train.min(0), np.ptp(train, axis=0)
    else:
        center, divisor = np.zeros(train.shape[1]), np.ones(train.shape[1])
    train, probe = (train-center)/divisor, (probe-center)/divisor
    centered = train-train.mean(0)
    centered_target = target-target.mean()
    correlations = (centered.T @ centered_target)**2 / ((centered**2).sum(0)*(centered_target**2).sum())
    selected = sorted(np.argsort(correlations)[-k:])
    design = np.column_stack([np.ones(len(train)), train[:, selected]])
    penalty = np.eye(k+1)*alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return np.column_stack([np.ones(len(probe)), probe[:, selected]]) @ coefficients


@pytest.mark.parametrize('scale', [None, 'standard', 'minmax'])
@pytest.mark.parametrize('refit', [False, True])
@pytest.mark.parametrize('poison', [False, True])
def test_agent_evolutionary_recipe_scores_winner_and_refit_match_oracles(scale, refit, poison):
    s, x, y = _session(poison)
    before = s.to_pandas().copy(deep=True)
    result = _run(s, {'estimator': 'Ridge', 'task': 'regression', 'ranking_metric': 'mse',
                      'param_space': {'alpha': [.1, 3.]}, 'recipe_space': {'select_k': [1, 2]},
                      'preprocess': {'impute': 'median', 'scale': scale, 'select': 'univariate'},
                      'population_size': 4, 'n_generations': 2, 'max_evaluations': 4,
                      'cv': 3, 'refit': refit})
    configurations = set()
    for trial in result.trials:
        alpha, k = trial.params['alpha'], trial.recipe_knobs['select_k']
        configurations.add((alpha, k))
        scores = []
        for training, evaluation in KFold(3, shuffle=True, random_state=42).split(x[:64]):
            predicted = _prediction(x[training], y[training], x[evaluation], scale, k, alpha)
            scores.append(np.mean((predicted-y[evaluation])**2))
        assert trial.mean_score == pytest.approx(np.mean(scores), rel=1e-8)
        assert trial.std_score == pytest.approx(np.std(scores, ddof=1), rel=1e-8)
    assert len(configurations) == len(result.trials) <= 4
    assert result.study['n_evaluations'] == len(configurations)
    assert result.best_score == min(trial.mean_score for trial in result.trials)
    assert result.best_params == result.trials[0].params
    assert result.best_recipe_knobs == result.trials[0].recipe_knobs
    if refit:
        expected = _prediction(x[:64], y[:64], x[80:], scale,
                               result.best_recipe_knobs['select_k'], result.best_params['alpha'])
        np.testing.assert_allclose(s.predict(partition='test'), expected, rtol=1e-8, atol=1e-8)
    else:
        assert s.fit_result is None
    pd.testing.assert_frame_equal(s.to_pandas(), before)
    assert tuple(s.split_plan.train_indices) == tuple(range(64))


@pytest.mark.parametrize('recipe', [[], 'standard', {'scale': 'bad'}, {'impute': 'bad'},
                                   {'select': 'bad'}, {'select_k': 0}, {'select_k': True},
                                   {'select_k': 1.5}, {'select_threshold': -1},
                                   {'select_threshold': float('nan')},
                                   {'select_threshold': True}, {'select_estimator': 'Ridge'}])
def test_invalid_agent_base_recipes_refuse_without_fitting_or_state_changes(recipe):
    s, _, _ = _session(False)
    before, history = s.to_pandas().copy(deep=True), list(s._history)
    registry = build_default_registry()
    arguments = {'estimator': 'Ridge', 'param_space': {'alpha': [.1, 3.]},
                 'recipe_space': {'select_k': [1, 2]}, 'preprocess': recipe,
                 'population_size': 4, 'n_generations': 2, 'max_evaluations': 4}
    result = execute_tool(s, propose_tool_execution('evolutionary_search', arguments, registry),
                          confirmed=True, registry=registry)
    assert not result.executed and 'preprocess' in result.error
    assert result.state_changes == ()
    assert s.fit_result is None and s.last_search is None
    assert s._history == history
    pd.testing.assert_frame_equal(s.to_pandas(), before)
