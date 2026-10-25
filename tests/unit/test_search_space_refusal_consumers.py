"""Malformed public search domains cannot replace a working fitted model."""
import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.model.selection import optuna_search

ENGINES = [
    'evolutionary_search',
    pytest.param('optuna_search', marks=pytest.mark.skipif(
        importlib.util.find_spec('optuna') is None, reason='optional Optuna backend absent')),
]
BAD_SPECS = [
    [], (), 5, None, {}, {'type': 'unknown'},
    {'type': 'categorical', 'choices': []},
    {'type': 'float', 'high': 2},
    {'type': 'float', 'low': 1},
    {'type': 'float', 'low': None, 'high': 2},
    {'type': 'float', 'low': True, 'high': 2},
    {'type': 'float', 'low': 1, 'high': False},
    {'type': 'float', 'low': float('nan'), 'high': 2},
    {'type': 'float', 'low': 1, 'high': float('inf')},
    {'type': 'float', 'low': 2, 'high': 1},
    {'type': 'float', 'low': 1, 'high': 2, 'log': 'false'},
    {'type': 'float', 'low': 0, 'high': 2, 'log': True},
    {'type': 'int', 'low': 1.5, 'high': 3},
    {'type': 'int', 'low': 1, 'high': 2.5},
]


def fitted_session():
    x = np.linspace(-2, 3, 36)
    frame = pd.DataFrame({'a': x, 'b': np.cos(x), 'y': 1.5*x - .7*np.cos(x) + .1*np.sin(x)})
    session = (Session.ingest(frame).set_roles({'a': 'feature', 'b': 'feature', 'y': 'target'})
               .split(test_size=.2, validation_size=.2, random_state=11).fit(Ridge(alpha=.4), task='regression'))
    train = session.partition('train')
    design = train[['a', 'b']].to_numpy()
    target = train.y.to_numpy()
    center = design.mean(axis=0)
    centered = design - center
    coef = np.linalg.solve(centered.T @ centered + .4*np.eye(2), centered.T @ (target-target.mean()))
    holdout = session.partition('test')[['a', 'b']].to_numpy()
    oracle = (holdout-center) @ coef + target.mean()
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)
    return session, oracle


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('spec', BAD_SPECS)
def test_invalid_search_domains_preserve_fitted_predictions(engine, spec):
    session, oracle = fitted_session()
    fitted = session.fit_result
    previous_search = session.last_search
    history = copy.deepcopy(session.history)
    options = dict(param_space={'alpha': spec}, task='regression', cv=2, refit=True, random_state=7)
    if engine == 'evolutionary_search':
        options.update(population_size=2, n_generations=1, elite_size=1, max_evaluations=2)
    else:
        options.update(n_trials=1)
    with pytest.raises(ValidationError):
        getattr(session, engine)(Ridge(), **options)
    assert session.fit_result is fitted
    assert session.last_search is previous_search
    assert session.history == history
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)


@pytest.mark.skipif(importlib.util.find_spec('optuna') is None, reason='optional Optuna backend absent')
@pytest.mark.parametrize('spec', BAD_SPECS)
def test_invalid_declared_domain_cannot_consume_existing_study(spec):
    import optuna

    session, oracle = fitted_session()
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: float(trial.suggest_int('previous', 1, 1)), n_trials=1)
    previous = [(trial.state, trial.params, trial.value) for trial in study.trials]
    with pytest.raises(ValidationError):
        optuna_search(session.dataset, session.split_plan, Ridge(), param_space={'alpha': spec}, study=study,
                              n_trials=1, task='regression', ranking_metric='rmse', cv=2)
    assert [(trial.state, trial.params, trial.value) for trial in study.trials] == previous
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('spec', [
    {'type': 'float', 'low': .1, 'high': 2.},
    {'type': 'float', 'low': .1, 'high': 2., 'log': True},
    {'type': 'int', 'low': 1, 'high': 3},
    [.1, .7, 2.],
    {'type': 'categorical', 'choices': [.1, .7, 2.]},
])
def test_valid_candidate_domains_match_independent_ridge_cv(engine, spec):
    session, oracle = fitted_session()
    original_fit = session.fit_result
    options = dict(param_space={'alpha': spec}, task='regression', cv=2,
                   ranking_metric='rmse', refit=False, random_state=7)
    if engine == 'evolutionary_search':
        options.update(population_size=4, n_generations=1, elite_size=1, max_evaluations=4)
    else:
        options.update(n_trials=4)
    result = getattr(session, engine)(Ridge(), **options)
    train = session.partition('train')
    x, y = train[['a', 'b']].to_numpy(), train.y.to_numpy()
    independent_scores = []
    for candidate in result.trials:
        alpha = candidate.params['alpha']
        if isinstance(spec, list) or spec.get('type') == 'categorical':
            assert alpha in (spec if isinstance(spec, list) else spec['choices'])
        else:
            assert spec['low'] <= alpha <= spec['high']
            if spec['type'] == 'int':
                assert isinstance(alpha, int)
        errors = []
        for fit, score in KFold(n_splits=2, shuffle=True, random_state=42).split(x):
            center = x[fit].mean(axis=0)
            centered = x[fit]-center
            coef = np.linalg.solve(centered.T @ centered + alpha*np.eye(2),
                                   centered.T @ (y[fit]-y[fit].mean()))
            prediction = (x[score]-center) @ coef + y[fit].mean()
            errors.append(np.sqrt(np.mean((prediction-y[score])**2)))
        assert candidate.mean_score == pytest.approx(np.mean(errors))
        assert candidate.std_score == pytest.approx(np.std(errors, ddof=1))
        independent_scores.append(np.mean(errors))
    assert result.best_score == pytest.approx(min(independent_scores))
    assert session.fit_result is original_fit
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)
