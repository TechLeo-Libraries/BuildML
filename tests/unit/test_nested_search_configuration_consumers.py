"""Nested search must honor its entire configuration and score independent outer fits."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe


def make_session():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(90, 2))
    frame = pd.DataFrame({'a': x[:, 0], 'b': x[:, 1],
                          'y': 1.2 + x @ [2., -.8] + rng.normal(size=90),
                          'weight': 1. + np.arange(90) % 4})
    return (Session.ingest(frame).set_roles({'a': 'feature', 'b': 'feature',
                                             'y': 'target', 'weight': 'weight'})
            .split(test_size=.2, validation_size=.1, random_state=6))


@pytest.mark.parametrize('method', ['typo', '', None])
@pytest.mark.parametrize('space', [{'param_grid': {'alpha': [1., 2.]}},
                                  {'param_distributions': {'alpha': [1., 2.]}}])
def test_unknown_inner_method_is_refused_before_work(method, space, monkeypatch):
    s = make_session()
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Invalid search configuration reached model fitting')
    monkeypatch.setattr(Ridge, 'fit', forbidden_fit)
    with pytest.raises(ValidationError, match='inner_search'):
        s.nested_cv_score(Ridge(), inner_search=method, outer_cv=2, inner_cv=2, **space)
    assert s.last_nested_cv is None and s.fit_result is None


@pytest.mark.parametrize('method', ['auto', 'grid', 'randomized'])
@pytest.mark.parametrize('spaces', [
    {'param_grid': {'alpha': [1., 2.]}, 'recipe_distributions': {'select_k': [1, 2]}},
    {'param_distributions': {'alpha': [1., 2.]}, 'recipe_grid': {'select_k': [1, 2]}},
])
def test_mixed_search_families_cannot_silently_drop_estimator_or_recipe_space(method, spaces, monkeypatch):
    s = make_session()
    before = s.to_pandas().copy(deep=True)
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Mixed search families reached fitting after silently dropping one space')
    monkeypatch.setattr(Ridge, 'fit', forbidden_fit)
    with pytest.raises(ValidationError, match='grid|randomized'):
        s.nested_cv_score(Ridge(), inner_search=method, outer_cv=2, inner_cv=2,
                          preprocess=PreprocessRecipe(scale='standard'), **spaces)
    assert s.last_nested_cv is None and s.fit_result is None
    pd.testing.assert_frame_equal(s.to_pandas(), before)


@pytest.mark.parametrize('options,match', [
    ({'inner_search': 'grid', 'param_space': {'alpha': [1.]}}, 'grid'),
    ({'inner_search': 'randomized', 'param_space': {'alpha': [1.]}}, 'randomized'),
    ({'param_grid': {'alpha': [1.]}, 'param_space': {'alpha': [1.]}}, 'either'),
    ({'inner_search': 'optuna', 'param_grid': {'alpha': [1.]}}, 'requires param_space'),
    ({'inner_search': 'evolutionary', 'param_grid': {'alpha': [1.]}}, 'requires param_space'),
    ({'inner_search': 'optuna', 'param_space': {'alpha': [1.]},
      'recipe_grid': {'select_k': [1]}}, 'omit recipe_grid'),
    ({'param_grid': {}, 'inner_search': 'grid'}, 'param_grid must not be empty'),
    ({'param_distributions': {}, 'inner_search': 'randomized'}, 'param_distributions must not be empty'),
    ({'param_grid': {'alpha': [1.]}, 'recipe_grid': {}}, 'recipe_grid must not be empty'),
    ({'param_distributions': {'alpha': [1.]}, 'recipe_distributions': {}}, 'recipe_distributions must not be empty'),
    ({'param_grid': {'alpha': [1.]}, 'warm_start_studies': True}, 'requires Optuna'),
])
def test_incompatible_nested_configuration_leaves_session_unfitted(options, match, monkeypatch):
    s = make_session()
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Invalid search request reached fitting')
    monkeypatch.setattr(Ridge, 'fit', forbidden_fit)
    with pytest.raises(ValidationError, match=match):
        s.nested_cv_score(Ridge(), outer_cv=2, inner_cv=2, **options)
    assert s.last_nested_cv is None and s.fit_result is None


def independent_fit_predict(fit, score, alpha):
    x = np.column_stack([np.ones(len(fit)), fit[['a', 'b']].to_numpy()])
    design = np.column_stack([np.ones(len(score)), score[['a', 'b']].to_numpy()])
    weighted = x.T * fit.weight.to_numpy()
    coefficients = np.linalg.solve(weighted @ x + np.diag([0., alpha, alpha]),
                                   weighted @ fit.y.to_numpy())
    return design @ coefficients


def independent_metrics(frame, predictions):
    error = frame.y.to_numpy() - predictions
    mse = np.average(error ** 2, weights=frame.weight)
    variance = np.average((frame.y - np.average(frame.y, weights=frame.weight)) ** 2,
                           weights=frame.weight)
    return {'mae': np.average(abs(error), weights=frame.weight), 'mse': mse,
            'rmse': np.sqrt(mse), 'r2': 1 - mse / variance}


@pytest.mark.parametrize('method', ['grid', 'randomized'])
@pytest.mark.parametrize('metric', ['mae', 'mse', 'r2'])
def test_nested_weighted_ridge_selection_matches_independent_normal_equations(method, metric):
    s = make_session()
    before = s.to_pandas().copy(deep=True)
    train = s.partition('train').reset_index(drop=True)
    expected = []
    for outer_fit, outer_eval in KFold(3, shuffle=True, random_state=42).split(train):
        outer_train = train.iloc[outer_fit]
        candidates = []
        for alpha in [.2, 2., 20.]:
            inner_scores = []
            for fit_pos, eval_pos in KFold(2, shuffle=True, random_state=42).split(outer_train):
                fit, score = outer_train.iloc[fit_pos], outer_train.iloc[eval_pos]
                inner_scores.append(independent_metrics(score, independent_fit_predict(fit, score, alpha))[metric])
            candidates.append((np.mean(inner_scores), np.std(inner_scores, ddof=1), alpha))
        winner = (max if metric == 'r2' else min)(candidates, key=lambda value: value[0])
        expected.append((winner, independent_metrics(train.iloc[outer_eval],
                         independent_fit_predict(outer_train, train.iloc[outer_eval], winner[2]))))
    options = {'param_grid': {'alpha': [.2, 2., 20.]}} if method == 'grid' else {
        'param_distributions': {'alpha': [.2, 2., 20.]}, 'n_iter': 3}
    result = s.nested_cv_score(Ridge(), task='regression', inner_search=method,
                               outer_cv=3, inner_cv=2, cv_strategy='kfold',
                               scoring_metric=metric, **options)
    for fold, (winner, metrics) in zip(result.outer_folds, expected, strict=True):
        assert fold.best_params == {'alpha': winner[2]}
        assert fold.inner_best_score == pytest.approx(winner[0])
        assert fold.inner_best_std == pytest.approx(winner[1])
        assert fold.inner_n_trials == 3
        assert fold.metrics == pytest.approx(metrics)
    for key in expected[0][1]:
        values = [item[1][key] for item in expected]
        assert result.mean_metrics[key] == pytest.approx(np.mean(values))
        assert result.std_metrics[key] == pytest.approx(np.std(values, ddof=1))
    assert result.search_method == method and result.population == 'train'
    assert s.fit_result is None
    pd.testing.assert_frame_equal(s.to_pandas(), before)
