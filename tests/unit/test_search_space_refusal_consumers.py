"""Malformed public search domains cannot replace a working fitted model."""
import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

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


@pytest.mark.parametrize('method', ['cv_score', 'grid_search', 'randomized_search',
                                     'evolutionary_search', 'nested_cv_score'])
@pytest.mark.parametrize('problem', ['one_fold', 'unknown_strategy', 'regression_stratified',
                                      'missing_groups', 'too_few_groups', 'short_groups',
                                      'long_groups', 'matrix_groups', 'missing_time', 'wrong_metric'])
def test_cv_configuration_refusals_preserve_working_model(method, problem):
    session, oracle = fitted_session()
    fitted, previous_search = session.fit_result, session.last_search
    holdout = session.partition('test').copy(deep=True)
    options = dict(task='regression', cv=2)
    if method == 'grid_search':
        options.update(param_grid={'alpha': [.4]}, refit=True)
    elif method == 'randomized_search':
        options.update(param_distributions={'alpha': [.4]}, n_iter=1, refit=True)
    elif method == 'evolutionary_search':
        options.update(param_space={'alpha': [.4]}, population_size=2,
                       n_generations=1, elite_size=1, max_evaluations=2, refit=True)
    elif method == 'nested_cv_score':
        options = dict(task='regression', outer_cv=2, inner_cv=2, param_grid={'alpha': [.4]})
    if problem == 'one_fold':
        options['outer_cv' if method == 'nested_cv_score' else 'cv'] = 1
    elif problem == 'unknown_strategy':
        options['cv_strategy'] = 'typo'
    elif problem == 'regression_stratified':
        options['cv_strategy'] = 'stratified'
    elif problem in {'missing_groups', 'too_few_groups', 'short_groups', 'long_groups', 'matrix_groups'}:
        options['cv_strategy'] = 'group'
        if problem == 'too_few_groups':
            options['groups'] = pd.Series(np.zeros(len(session.partition('train'))))
        elif problem == 'short_groups':
            options['groups'] = pd.Series([0, 1])
        elif problem == 'long_groups':
            options['groups'] = pd.Series(np.arange(len(session.partition('train')) + 1))
        elif problem == 'matrix_groups':
            options['groups'] = pd.DataFrame({'g': np.arange(len(session.partition('train')))})
    elif problem == 'missing_time':
        options['cv_strategy'] = 'time'
    else:
        options['scoring_metric' if method in {'cv_score', 'nested_cv_score'} else 'ranking_metric'] = 'accuracy'
    history = copy.deepcopy(session.history)
    with pytest.raises(ValidationError):
        getattr(session, method)(Ridge(), **options)
    assert session.fit_result is fitted
    assert session.last_search is previous_search
    assert session.history == history
    pd.testing.assert_frame_equal(session.partition('test'), holdout)
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)


@pytest.mark.parametrize('overrides', [
    {'param_grid': None},
    {'param_distributions': {'alpha': [.4]}},
    {'recipe_distributions': {'scale': ['standard']}},
    {'inner_search': 'evolutionary'},
    {'inner_search': 'optuna'},
    {'param_space': {'alpha': [.4]}},
    {'inner_search': 'grid', 'param_grid': None, 'param_distributions': {'alpha': [.4]}},
    {'inner_search': 'grid', 'param_space': {'alpha': [.4]}},
    {'inner_search': 'randomized', 'param_grid': None, 'param_space': {'alpha': [.4]}},
    {'inner_search': 'optuna', 'param_space': {'alpha': [.4]}},
    {'inner_search': 'evolutionary', 'param_space': {'alpha': [.4]}},
    {'warm_start_studies': True},
    {'param_grid': {}},
    {'param_grid': None, 'param_distributions': {}},
    {'param_grid': None, 'recipe_grid': {}},
    {'param_grid': None, 'recipe_distributions': {}},
    {'param_grid': None, 'param_space': {'alpha': [.4]}, 'inner_search': 'evolutionary', 'n_trials': 0},
    {'inner_search': 'optuna', 'param_grid': None, 'param_space': {'alpha': [.4]}, 'recipe_grid': {'scale': ['standard']}},
])
def test_nested_search_routing_conflicts_leave_existing_fit_usable(overrides):
    session, oracle = fitted_session()
    fitted, previous_search = session.fit_result, session.last_search
    history = copy.deepcopy(session.history)
    options = dict(param_grid={'alpha': [.4]}, outer_cv=2, inner_cv=2, task='regression')
    options.update(overrides)
    with pytest.raises(ValidationError):
        session.nested_cv_score(Ridge(), **options)
    assert session.fit_result is fitted
    assert session.last_search is previous_search
    assert session.history == history
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)


@pytest.mark.parametrize('method', ['cv_score', 'nested_cv_score'])
@pytest.mark.parametrize('index_kind', ['default', 'duplicate', 'reversed'])
def test_valid_group_labels_are_positional_and_match_independent_scores(method, index_kind):
    session, oracle = fitted_session()
    fitted = session.fit_result
    train = session.partition('train')
    x, y = train[['a', 'b']].to_numpy(), train.y.to_numpy()
    groups = pd.Series(np.arange(len(train)) % 4)
    if index_kind == 'duplicate':
        groups.index = [1000] * len(train)
    elif index_kind == 'reversed':
        groups.index = list(reversed(range(1000, 1000 + len(train))))
    expected = []
    for fit, score in GroupKFold(n_splits=2).split(x, y, groups.to_numpy()):
        center = x[fit].mean(axis=0)
        centered = x[fit] - center
        coef = np.linalg.solve(centered.T @ centered + .4*np.eye(2),
                               centered.T @ (y[fit] - y[fit].mean()))
        prediction = (x[score] - center) @ coef + y[fit].mean()
        expected.append(np.sqrt(np.mean((prediction - y[score])**2)))
    options = dict(task='regression', cv_strategy='group', groups=groups, scoring_metric='rmse')
    if method == 'nested_cv_score':
        options.update(outer_cv=2, inner_cv=2, param_grid={'alpha': [.4]})
    else:
        options.update(cv=2)
    result = getattr(session, method)(Ridge(alpha=.4), **options)
    assert result.mean_metrics['rmse'] == pytest.approx(np.mean(expected))
    assert result.std_metrics['rmse'] == pytest.approx(np.std(expected, ddof=1))
    assert session.fit_result is fitted
    np.testing.assert_allclose(session.predict(), oracle, atol=1e-12)
