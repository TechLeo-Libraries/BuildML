"""Selector task routing and persistence follow target meaning, not cardinality."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import PreprocessRecipe, build_fold_preprocessor


def _frame():
    rng = np.random.default_rng(84)
    return pd.DataFrame(rng.normal(size=(120, 3)), columns=['signal', 'noise', 'other'])


@pytest.mark.parametrize('target,expected', [
    (pd.Series([f'class_{i % 30}' for i in range(120)]), RandomForestClassifier),
    (pd.Series(np.tile([.1, .25, .5], 40)), RandomForestRegressor),
])
def test_model_selector_handles_many_string_classes_and_few_continuous_values(target, expected):
    frame = _frame()
    recipe = PreprocessRecipe(impute=None, select='model')
    processor = build_fold_preprocessor(frame, recipe, target)
    assert isinstance(processor._selector.estimator, expected)
    assert processor.transform(frame).shape[1] >= 1


@pytest.mark.parametrize('score', ['f_classif', 'f_regression'])
def test_explicit_univariate_scorer_is_honored_above_twenty_levels(score):
    frame = _frame()
    target = pd.Series(np.arange(120) % 30)
    processor = build_fold_preprocessor(frame, PreprocessRecipe(
        impute=None, select='univariate', select_k=2, select_score_func=score), target)
    expected, _ = (f_classif if score == 'f_classif' else f_regression)(frame, target)
    np.testing.assert_allclose(processor._selector.scores_, expected)


def test_default_univariate_routes_continuous_targets_to_regression():
    frame = _frame()
    target = pd.Series(np.tile([.1, .25, .5], 40))
    processor = build_fold_preprocessor(frame, PreprocessRecipe(
        impute=None, select='univariate', select_k=2), target)
    expected, _ = f_regression(frame, target)
    np.testing.assert_allclose(processor._selector.scores_, expected)


@pytest.mark.parametrize('score', ['unknown', '', None, [], {}])
def test_invalid_scorer_is_refused_before_fitting(score, monkeypatch):
    from sklearn.feature_selection import SelectKBest

    def unexpected_fit(*args, **kwargs):
        pytest.fail('Invalid scorer started feature fitting')

    monkeypatch.setattr(SelectKBest, 'fit', unexpected_fit)
    with pytest.raises(ValidationError, match='select_score_func'):
        build_fold_preprocessor(_frame(), PreprocessRecipe(
            impute=None, select='univariate', select_score_func=score), pd.Series([0, 1] * 60))


def test_zero_importance_fallback_remains_serializable_and_preserves_input(tmp_path):
    frame = _frame()
    target = pd.Series([0, 1] * 60)
    estimator = LogisticRegression(solver='liblinear', penalty='l1', C=1e-6)
    processor = build_fold_preprocessor(frame, PreprocessRecipe(
        impute=None, select='model', select_estimator=estimator), target)
    transformed = processor.transform(frame)
    assert transformed.shape == (120, 1)
    assert not hasattr(estimator, 'coef_')
    path = tmp_path / 'selector.joblib'
    joblib.dump(processor, path)
    restored = joblib.load(path)
    pd.testing.assert_frame_equal(restored.transform(frame), transformed)


@pytest.mark.parametrize('value', [np.inf, -np.inf])
@pytest.mark.parametrize('selection', ['univariate', 'model'])
def test_nonfinite_target_is_refused_before_supervised_selection(value, selection):
    target = pd.Series([0., 1.] * 60)
    target.iloc[3] = value
    with pytest.raises(ValidationError, match='finite target'):
        build_fold_preprocessor(_frame(), PreprocessRecipe(impute=None, select=selection), target)


def test_regression_cv_refits_the_fixed_selector_inside_each_fold():
    from sklearn.linear_model import Ridge

    frame = _frame()
    frame['target'] = .5 * frame.signal + .2 * frame.other
    session = Session.ingest(frame).set_roles({'target': 'target'}).split(test_size=.2, random_state=4)
    held_out = session.partition('test').copy(deep=True)
    result = session.cv_score(Ridge(), task='regression', cv=3,
                              preprocess=PreprocessRecipe(impute=None, select='model'))
    assert np.isfinite(result.mean_metrics['r2'])
    pd.testing.assert_frame_equal(session.partition('test'), held_out)


@pytest.mark.parametrize('method', ['cv_score', 'grid_search', 'nested_cv_score'])
@pytest.mark.parametrize('selection', ['univariate', 'model'])
def test_declared_regression_task_controls_integer_target_selection(method, selection, monkeypatch):
    from sklearn.linear_model import Ridge

    from buildml.model import selection as api

    frame = _frame()
    frame['target'] = np.round(15 * frame.signal).astype(int) + 100
    session = Session.ingest(frame).set_roles({'target': 'target'}).split(test_size=.2, random_state=4)
    recipe = PreprocessRecipe(impute=None, select=selection, select_k=1)
    observed = []
    real = api.build_fold_preprocessor

    def capture(*args, **kwargs):
        processor = real(*args, **kwargs)
        observed.append(processor._selector)
        return processor

    monkeypatch.setattr(api, 'build_fold_preprocessor', capture)
    options = dict(task='regression', preprocess=recipe)
    if method == 'nested_cv_score':
        options.update(outer_cv=3, inner_cv=2, param_grid={'alpha': [1.]})
    elif method == 'grid_search':
        options.update(cv=3, param_grid={'alpha': [1.]})
    else:
        options.update(cv=3)
    result = getattr(session, method)(Ridge(), **options)
    assert observed
    for selector in observed:
        if selection == 'model':
            assert isinstance(selector.estimator, RandomForestRegressor)
        else:
            assert selector.score_func is f_regression
    assert recipe.select_estimator is None
    assert recipe.select_score_func == 'auto'
    if method == 'cv_score':
        assert result.mean_metrics['r2'] > .9
