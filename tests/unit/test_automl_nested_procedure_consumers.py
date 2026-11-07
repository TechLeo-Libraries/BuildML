"""Nested AutoML must measure its configured search, budgets and ensemble choices."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from buildml import Session
from buildml.automl import search
from buildml.core.errors import ValidationError
from buildml.model.supervised import fit_kwargs_for_sample_weight
from buildml.preprocess import PreprocessRecipe


def session():
    rng = np.random.default_rng(43)
    x, z = rng.normal(size=(2, 72))
    frame = pd.DataFrame({'x': x, 'z': z, 'y': 2*x+.3*z+rng.normal(size=72),
                          'weight': 1+np.arange(72)%6}, index=[f'row-{i}' for i in range(72)])
    return (Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature',
                                            'y': 'target', 'weight': 'weight'})
            .split(test_size=.2, validation_size=.1, random_state=3))


def ridge_prediction(fit, score, alpha):
    x = np.column_stack([np.ones(len(fit)), fit[['x', 'z']]])
    target = np.column_stack([np.ones(len(score)), score[['x', 'z']]])
    weighted = x.T*fit.weight.to_numpy()
    coefficients = np.linalg.solve(weighted@x+np.diag([0., alpha, alpha]), weighted@fit.y)
    return target@coefficients


def metric_value(frame, prediction, metric):
    mse = np.average((frame.y-prediction)**2, weights=frame.weight)
    if metric == 'mse':
        return mse
    variance = np.average((frame.y-np.average(frame.y, weights=frame.weight))**2,
                           weights=frame.weight)
    return 1-mse/variance


def trace_real_search(monkeypatch):
    original = search.run_automl
    calls = []
    def traced(dataset, split_plan, **options):
        result = original(dataset, split_plan, **options)
        calls.append((dataset, split_plan, options, result))
        return result
    monkeypatch.setattr(search, 'run_automl', traced)
    return calls


@pytest.mark.parametrize('method', ['grid', 'randomized', 'evolutionary', 'optuna'])
@pytest.mark.parametrize('metric', ['mse', 'r2'])
def test_every_outer_fold_runs_actual_requested_search_and_budget(monkeypatch, method, metric):
    if method == 'optuna':
        pytest.importorskip('optuna')
    s = session()
    before = s.to_pandas().copy(deep=True)
    calls = trace_real_search(monkeypatch)
    result = s.automl.run(method=method, selection='nested', task='regression',
        families=('ridge',), ranking_metric=metric, n_trials=3, cv=2, outer_cv=3,
        include_recipe_search=False, preprocess=PreprocessRecipe(impute=None), random_state=7)
    inner_calls = [item for item in calls if item[2]['selection'] == 'cv']
    assert len(inner_calls) == 3, 'Each outer fold must repeat the actual configured search'
    expected = []
    for dataset, split, options, (_, selected, fitted) in inner_calls:
        assert options['method'] == method and options['n_trials'] == 3
        assert options['budget'].study_storage is None
        assert options['include_ensembles'] is False
        assert set(split.train_indices).isdisjoint(split.test_indices)
        assert set(split.train_indices+split.test_indices) == set(s.split_plan.train_indices)
        data = dataset._ensure_pandas()
        fit, score = data.iloc[list(split.train_indices)], data.iloc[list(split.test_indices)]
        for trial in selected.trials:
            scores = []
            for a, b in KFold(2, shuffle=True, random_state=42).split(fit):
                training, evaluation = fit.iloc[a], fit.iloc[b]
                scores.append(metric_value(evaluation,
                    ridge_prediction(training, evaluation, trial.params['alpha']), metric))
            assert trial.mean_score == pytest.approx(np.mean(scores), rel=1e-9, abs=1e-10)
        predicted = ridge_prediction(fit, score, selected.best_params['alpha'])
        np.testing.assert_allclose(fitted.estimator.predict(score[['x', 'z']]), predicted, atol=1e-9)
        expected.append(metric_value(score, predicted, metric))
    assert result.outer_score_mean == pytest.approx(np.mean(expected))
    assert result.outer_score_std == pytest.approx(np.std(expected, ddof=1))
    pd.testing.assert_frame_equal(s.to_pandas(), before)


@pytest.mark.parametrize('metric', ['mse', 'r2'])
def test_outer_search_includes_requested_real_voting_candidates(monkeypatch, metric):
    s = session()
    calls = trace_real_search(monkeypatch)
    result = s.automl.run(method='grid', selection='nested', task='regression',
        families=('ridge', 'lasso'), ranking_metric=metric, n_trials=6, cv=2, outer_cv=3,
        include_recipe_search=False, preprocess=PreprocessRecipe(impute=None),
        include_ensembles=True, ensemble_mode='voting', max_ensemble_bases=2, random_state=7)
    inner_calls = [item for item in calls if item[2]['selection'] == 'cv']
    assert len(inner_calls) == 3
    expected = []
    for dataset, split, options, (_, selected, fitted) in inner_calls:
        assert options['include_ensembles'] is True and options['ensemble_mode'] == 'voting'
        assert options['max_ensemble_bases'] == 2 and options['n_trials'] == 6
        assert any(trial.kind == 'voting' for trial in selected.trials)
        score = dataset._ensure_pandas().iloc[list(split.test_indices)]
        expected.append(metric_value(score, fitted.estimator.predict(score[['x', 'z']]), metric))
    assert result.outer_score_mean == pytest.approx(np.mean(expected))
    assert result.outer_score_std == pytest.approx(np.std(expected, ddof=1))


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_real_weighted_voting_routes_weights_to_members(task):
    x = np.zeros((4, 1))
    weights = pd.Series([9., 1., 1., 1.])
    if task == 'regression':
        y = np.array([1., 2., 20., 30.])
        model = VotingRegressor([('a', DummyRegressor()), ('b', DummyRegressor()), ('unused', 'drop')])
    else:
        y = np.array([0, 1, 1, 0])
        model = VotingClassifier([('a', DummyClassifier(strategy='prior')),
                                  ('b', DummyClassifier(strategy='prior'))], voting='soft')
    model.fit(x, y, **fit_kwargs_for_sample_weight(model, weights))
    if task == 'regression':
        np.testing.assert_allclose(model.predict(x), np.average(y, weights=weights))
    else:
        probabilities = np.array([np.sum(weights[y == value])/weights.sum() for value in [0, 1]])
        np.testing.assert_allclose(model.predict_proba(x), np.tile(probabilities, (4, 1)))


def test_voting_member_without_weight_support_is_refused_before_fit():
    model = VotingRegressor([('weighted', DummyRegressor()), ('unweighted', KNeighborsRegressor())])
    with pytest.raises(ValidationError, match='KNeighborsRegressor.*sample_weight'):
        fit_kwargs_for_sample_weight(model, pd.Series([1., 2., 3.]))
    assert not hasattr(model, 'estimators_')


@pytest.mark.parametrize('metric', ['mse', 'r2'])
def test_voting_uses_best_diverse_families_in_metric_direction(metric):
    s = session().set_roles({'weight': 'ignore'})
    result = s.automl.run(method='grid', selection='cv', task='regression',
        families=('ridge', 'lasso', 'decision_tree'), ranking_metric=metric, n_trials=30,
        cv=2, include_recipe_search=False, preprocess=PreprocessRecipe(impute=None),
        include_ensembles=True, ensemble_mode='voting', max_ensemble_bases=2, random_state=7)
    scores = {}
    for trial in result.trials:
        if trial.kind == 'single':
            scores.setdefault(trial.family, []).append(trial.mean_score)
    assert len(scores) == 3
    best = {family: (max if metric == 'r2' else min)(values) for family, values in scores.items()}
    expected = tuple(sorted(best, key=best.get, reverse=metric == 'r2')[:2])
    voting = [trial for trial in result.trials if trial.kind == 'voting']
    assert len(voting) == 1 and voting[0].ensemble_bases == expected
