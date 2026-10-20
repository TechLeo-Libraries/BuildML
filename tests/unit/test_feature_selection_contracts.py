"""Feature selection numerical populations and caller-estimator ownership."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan
from buildml.preprocess.select import fit_feature_selector, transform_feature_selector


def population(task='regression'):
    rng = np.random.default_rng(82)
    x = rng.normal(size=100)
    z = rng.normal(size=100)
    y = 4*x+.1*z if task == 'regression' else np.digitize(x, [-.5, .5])
    frame = pd.DataFrame({'x': x, 'z': z, 'constant': np.ones(100), 'y': y,
                          'record': [f'row{i}' for i in range(100)]}, index=[9]*100)
    frame.iloc[80:, :3] = np.nan  # Holdout input must not enter fitting.
    session = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature',
        'constant': 'feature', 'y': 'target', 'record': 'id'})
    split = SplitPlan('injected', None, None, 3, None, tuple(range(80)), tuple(range(80, 90)), tuple(range(90, 100)))
    return session.dataset, split, frame


@pytest.mark.parametrize('strategy,options', [
    ('variance', {'threshold': float('nan')}), ('variance', {'threshold': float('inf')}),
    ('variance', {'threshold': True}), ('variance', {'threshold': -1}),
    ('univariate', {'k': True}), ('univariate', {'k': 1.5}), ('univariate', {'k': 0}),
    ('univariate', {'score_func': 'typo'}),
])
def test_invalid_selection_controls_raise_validation_error(strategy, options):
    dataset, split, _ = population()
    with pytest.raises(ValidationError):
        fit_feature_selector(dataset, split, strategy=strategy, **options)


@pytest.mark.parametrize('prefitted', [False, True])
@pytest.mark.parametrize('model', [Ridge(alpha=2), LogisticRegression(max_iter=200)])
def test_selection_never_fits_or_overwrites_the_callers_estimator(prefitted, model):
    from sklearn.base import clone

    task = 'classification' if isinstance(model, LogisticRegression) else 'regression'
    dataset, split, frame = population(task)
    caller = clone(model)
    if prefitted:
        caller.fit(frame.iloc[:40][['x', 'z', 'constant']], frame.iloc[:40].y)
    state = deepcopy(caller.__dict__)
    fit_feature_selector(dataset, split, strategy='model', estimator=caller)
    assert caller.__dict__.keys() == state.keys()
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(caller.__dict__[key], value)
        else:
            assert caller.__dict__[key] == value


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_default_model_selection_uses_train_only_and_reports_actual_scores(task):
    dataset, split, frame = population(task)
    plan = fit_feature_selector(dataset, split, strategy='model')
    assert plan.estimator_name == ('RandomForestClassifier' if task == 'classification' else 'RandomForestRegressor')
    assert set(plan.scores_) == {'x', 'z', 'constant'}
    assert sum(plan.scores_.values()) == pytest.approx(1)
    assert plan.scores_['constant'] == 0
    assert 'x' in plan.selected_features_
    narrowed, _ = transform_feature_selector(dataset, plan)
    expected = frame[list(plan.selected_features_) + ['y', 'record']]
    pd.testing.assert_frame_equal(narrowed._ensure_pandas(), expected)


def test_variance_scores_are_population_variances_from_train_positions():
    dataset, split, frame = population()
    plan = fit_feature_selector(dataset, split, strategy='variance')
    for column in ['x', 'z', 'constant']:
        assert plan.scores_[column] == pytest.approx(np.var(frame[column].iloc[:80], ddof=0))
    assert plan.selected_features_ == ('x', 'z')


def test_univariate_regression_scores_match_the_correlation_f_statistic():
    dataset, split, frame = population()
    plan = fit_feature_selector(dataset, split, strategy='univariate', k=1)
    for column in ['x', 'z']:
        r = np.corrcoef(frame[column].iloc[:80], frame.y.iloc[:80])[0, 1]
        assert plan.scores_[column] == pytest.approx(r*r/(1-r*r)*78, rel=1e-8)
    assert plan.selected_features_ == ('x',)
    assert plan.score_func == 'f_regression'


def test_zero_coefficient_model_retains_a_single_declared_feature():
    dataset, split, _ = population()
    plan = fit_feature_selector(dataset, split, strategy='model', estimator=Lasso(alpha=1e6))
    assert len(plan.selected_features_) == 1
    assert set(plan.scores_.values()) == {0}


@pytest.mark.parametrize('strategy', ['variance', 'univariate', 'model'])
def test_protected_only_candidates_are_refused_before_sklearn(strategy):
    dataset, split, _ = population()
    with pytest.raises(ValidationError, match='feature|candidate'):
        fit_feature_selector(dataset, split, strategy=strategy, columns=['record', 'y'])


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_mutual_information_scores_use_the_correct_task_and_train_population(task):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    dataset, split, frame = population(task)
    rng_state = np.random.get_state()
    try:
        np.random.seed(5)
        plan = fit_feature_selector(dataset, split, strategy='univariate', k=1,
                                    score_func='mutual_info')
        function = mutual_info_classif if task == 'classification' else mutual_info_regression
        reference = function(frame.iloc[:80][['x', 'z', 'constant']], frame.iloc[:80].y,
                             random_state=5)
    finally:
        np.random.set_state(rng_state)
    np.testing.assert_allclose(list(plan.scores_.values()), reference, rtol=1e-12, atol=1e-12)
    assert plan.selected_features_ == ('x',)
    assert plan.score_func == ('mutual_info_classif' if task == 'classification' else 'mutual_info_regression')
