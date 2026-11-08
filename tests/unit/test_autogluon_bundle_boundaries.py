"""Bundle filesystem guards; numerical AutoGluon fits are tested separately."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from buildml.automl.adapters.autogluon import _AutoGluonSklearnWrapper
from buildml.automl.checkpoint import load_automl_bundle, save_automl_bundle
from buildml.automl.results import AutoMLPlan, fit_result_from_plan
from buildml.core.errors import ValidationError


@pytest.fixture
def plan():
    x = np.arange(20).reshape(-1,1)
    return AutoMLPlan(task='regression', method='randomized', selection='cv',
        ranking_metric='r2', best_family='ridge', best_recipe_strategy='none',
        best_kind='single', best_params={}, best_recipe={}, best_score=.9, best_std=0.,
        feature_columns=('x',), target_column='y', n_train_rows=20,
        estimator_=Ridge().fit(x, x[:,0]*2))


@pytest.mark.parametrize('value', [42, [], {}, '', '.', 'missing', '../outside', 'ABSOLUTE'])
def test_invalid_asset_location_is_refused_before_backend_loading(tmp_path, plan, value):
    bundle = save_automl_bundle(tmp_path/'bundle', plan)
    path = bundle/'meta.json'
    metadata = json.loads(path.read_text())
    metadata['autogluon_assets'] = str(tmp_path/'outside') if value == 'ABSOLUTE' else value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError, match='assets'):
        load_automl_bundle(bundle, trusted=True)


def test_native_estimator_cannot_be_reinterpreted_as_autogluon(tmp_path, plan):
    bundle = save_automl_bundle(tmp_path/'bundle', plan)
    (bundle/'assets').mkdir()
    path = bundle/'meta.json'
    metadata = json.loads(path.read_text())
    metadata['autogluon_assets'] = 'assets'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError, match='requires an AutoGluon estimator'):
        load_automl_bundle(bundle, trusted=True)


def _backend_plan(plan, path, clone):
    predictor = SimpleNamespace(path=str(path), problem_type='regression', clone=clone)
    wrapper = _AutoGluonSklearnWrapper(predictor, target='y', feature_cols=['x'])
    return replace(plan, estimator_=wrapper)


def test_recursive_destination_does_not_start_backend_clone(tmp_path, plan):
    def unexpected(path):
        pytest.fail('Recursive destination reached backend clone')
    source = tmp_path/'predictor'
    source.mkdir()
    wrapped = _backend_plan(plan, source, unexpected)
    destination = source/'nested'
    with pytest.raises(ValidationError, match='outside'):
        save_automl_bundle(destination, wrapped)
    assert not destination.exists()


def test_mismatched_fit_result_is_refused_before_writing(tmp_path, plan):
    def unexpected(path):
        pytest.fail('Mismatched fit result reached backend clone')
    wrapped = _backend_plan(plan, tmp_path/'predictor', unexpected)
    destination = tmp_path/'bundle'
    with pytest.raises(ValidationError, match='same estimator'):
        save_automl_bundle(destination, wrapped, fit_result=fit_result_from_plan(plan))
    assert not destination.exists()


def test_backend_clone_failure_preserves_existing_bundle(tmp_path, plan):
    destination = save_automl_bundle(tmp_path/'bundle', plan)
    previous = {name:(destination/name).read_bytes() for name in ('meta.json','automl_plan.joblib')}
    def failed_clone(path):
        raise OSError('backend storage unavailable')
    wrapped = _backend_plan(plan, tmp_path/'predictor', failed_clone)
    with pytest.raises(OSError, match='storage unavailable'):
        save_automl_bundle(destination, wrapped)
    for name, data in previous.items():
        assert (destination/name).read_bytes() == data
    restored, _ = load_automl_bundle(destination, trusted=True)
    np.testing.assert_allclose(restored.estimator_.predict([[2.]]),plan.estimator_.predict([[2.]]))
