"""Real search candidates must share reusable CV evidence and truthful metadata."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from buildml import Session
from buildml.preprocess import PreprocessRecipe


def population():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(64, 2))
    return (Session.ingest(pd.DataFrame({'x': x[:, 0], 'z': x[:, 1],
                                        'y': x @ [2., -.7] + rng.normal(size=64)}))
            .set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'})
            .split(test_size=.25, random_state=19))


def independent_score(frame, splits, alpha):
    scores = []
    for fit, score in splits:
        a = np.column_stack([np.ones(len(fit)), frame.iloc[fit][['x', 'z']]])
        b = np.column_stack([np.ones(len(score)), frame.iloc[score][['x', 'z']]])
        coefficients = np.linalg.solve(a.T @ a + np.diag([0., alpha, alpha]),
                                       a.T @ frame.iloc[fit].y)
        scores.append(np.mean((frame.iloc[score].y - b @ coefficients)**2))
    return np.mean(scores), np.std(scores, ddof=1)


@pytest.mark.parametrize('method', ['grid', 'randomized', 'evolutionary', 'optuna'])
@pytest.mark.parametrize('form', ['splitter', 'list', 'tuple', 'generator'])
def test_all_candidates_replay_custom_cv_and_report_its_actual_kind(method, form):
    if method == 'optuna':
        pytest.importorskip('optuna')
    s = population()
    frame = s.to_pandas().iloc[list(s.split_plan.train_indices)]
    splits = list(KFold(4, shuffle=True, random_state=31).split(frame))
    cv = {'splitter': KFold(4, shuffle=True, random_state=31),
          'list': splits, 'tuple': tuple(splits),
          'generator': (pair for pair in splits)}[form]
    result = s.automl.run(method=method, task='regression', selection='cv',
        families=('ridge',), ranking_metric='mse', n_trials=3, cv=cv,
        include_recipe_search=False, preprocess=PreprocessRecipe(impute=None), random_state=17)
    assert len(result.trials) == 3, 'A one-shot CV iterator must not silently discard candidates'
    for trial in result.trials:
        mean, std = independent_score(frame, splits, trial.params['alpha'])
        assert trial.mean_score == pytest.approx(mean, rel=1e-9, abs=1e-10)
        assert trial.std_score == pytest.approx(std, rel=1e-9, abs=1e-10)
    assert result.config['cv'] is None, 'A custom splitter is not a request for three folds'
    assert result.config['extras']['cv_splitter'] == f'{type(cv).__module__}.{type(cv).__qualname__}'
    assert json.loads(json.dumps(result.to_dict()))['config']['cv'] is None


def test_nested_custom_splitters_are_not_serialized_as_three_folds():
    s = population()
    result = s.automl.run(method='grid', selection='nested', task='regression',
        families=('ridge',), ranking_metric='mse', n_trials=2,
        cv=KFold(2), outer_cv=KFold(4), include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None))
    assert result.config['cv'] is None
    assert result.config['outer_cv'] is None
    assert result.config['extras']['outer_cv_splitter'].endswith('.KFold')
    assert np.isfinite(result.outer_score_mean)


def test_integer_requests_remain_serializable_and_are_not_custom_splitters():
    result = population().automl.run(method='grid', task='regression',
        families=('ridge',), n_trials=2, cv=4, outer_cv=2,
        include_recipe_search=False, preprocess=PreprocessRecipe(impute=None))
    assert result.config['cv'] == 4 and result.config['outer_cv'] == 2
    assert result.config['extras']['cv_splitter'] is None
    assert result.config['extras']['outer_cv_splitter'] is None


@pytest.mark.parametrize('mode', ['voting', 'stacking'])
def test_one_shot_splits_reach_real_ensemble_candidates_as_well_as_single_models(mode):
    first, second = population(), population()
    splits = list(KFold(4).split(first.split_plan.train_indices))
    options = dict(method='grid', task='regression', families=('ridge', 'lasso'),
        ranking_metric='mse', n_trials=6, include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None), include_ensembles=True,
        ensemble_mode=mode, max_ensemble_bases=2, random_state=17)
    reference = first.automl.run(cv=splits, **options)
    replayed = second.automl.run(cv=(pair for pair in splits), **options)
    assert any(trial.kind == mode for trial in replayed.trials)
    assert [(t.family, t.kind, t.params) for t in replayed.trials] == [
        (t.family, t.kind, t.params) for t in reference.trials]
    np.testing.assert_allclose([t.mean_score for t in replayed.trials],
                               [t.mean_score for t in reference.trials], atol=1e-10)
    features = first.to_pandas()[['x', 'z']]
    np.testing.assert_allclose(first.fit_result.estimator.predict(features),
                               second.fit_result.estimator.predict(features), atol=1e-10)


def test_custom_cv_metadata_survives_real_bundle_roundtrip(tmp_path):
    s = population()
    result = s.automl.run(method='grid', task='regression', families=('ridge',),
        n_trials=2, cv=KFold(4), include_recipe_search=False,
        preprocess=PreprocessRecipe(impute=None))
    features = s.to_pandas()[['x', 'z']]
    prediction = s.fit_result.estimator.predict(features)
    path = s.automl.save_bundle(tmp_path / 'custom-cv')
    restored = population()
    restored.automl.load_bundle(path, trusted=True)
    assert restored.automl_plan.config == result.config
    assert restored.automl_plan.config['cv'] is None
    assert restored.automl_plan.config['extras']['cv_splitter'].endswith('.KFold')
    np.testing.assert_allclose(restored.fit_result.estimator.predict(features), prediction)
