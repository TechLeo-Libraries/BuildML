"""AutoML bundle model and classical FitResult contracts must remain consistent."""

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.automl.checkpoint import load_automl_bundle, save_automl_bundle
from buildml.core.errors import ValidationError


@pytest.fixture(scope='module', params=['regression', 'classification'])
def fitted(request):
    task = request.param
    rng = np.random.default_rng(881)
    x = rng.normal(size=(90, 2))
    signal = x @ [2., -.4] + rng.normal(0, .2, 90)
    y = (signal > 0).astype(int) if task == 'classification' else signal
    def fit(target):
        frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'y': target})
        s = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'})
        s.split(test_size=.25, validation_size=.2, random_state=21, stratify=task == 'classification')
        s.automl.run(method='grid', task=task, families=('logistic' if task == 'classification' else 'ridge',),
                     include_recipe_search=False, include_ensembles=False, n_trials=2, cv=3)
        return s
    return fit(y), fit(1-y if task == 'classification' else y+8)


def _expected(plan, frame):
    model = plan.estimator_
    if hasattr(model, 'named_steps'):
        model = list(model.named_steps.values())[-1]
    x = frame[list(plan.feature_columns)].to_numpy()
    if plan.task == 'regression':
        return x @ model.coef_ + model.intercept_
    logits = (x @ model.coef_.T + model.intercept_)[:, 0]
    return np.asarray(model.classes_)[(logits > 0).astype(int)]


def _destination(source, root):
    destination = Session.ingest(source.to_pandas()).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target'})
    destination.inject_split(train_indices=list(source.split_plan.train_indices),
                              validation_indices=list(source.split_plan.validation_indices),
                              test_indices=list(source.split_plan.test_indices))
    destination.automl.load_bundle(root, trusted=True)
    return destination


@pytest.mark.parametrize('layout', ['full', 'direct', 'missing_fit', 'legacy_fit'])
def test_automl_legacy_and_current_layouts_preserve_both_prediction_paths(fitted, layout, tmp_path):
    source, _ = fitted
    root = tmp_path / 'automl'
    source.automl.save_bundle(root)
    payload = joblib.load(root / 'automl_plan.joblib')
    if layout == 'direct':
        payload = payload['plan']
    elif layout == 'missing_fit':
        payload.pop('fit_result')
    elif layout == 'legacy_fit':
        payload['fit_result'] = {'historical': 'metadata'}
    joblib.dump(payload, root / 'automl_plan.joblib')
    plan, fit = load_automl_bundle(root, trusted=True)
    assert fit.estimator is plan.estimator_
    probe = source.partition('test')
    expected = _expected(source.automl_plan, probe)
    np.testing.assert_allclose(plan.estimator_.predict(probe[list(plan.feature_columns)]), expected)
    destination = _destination(source, root)
    np.testing.assert_allclose(destination.predict(partition='test'), expected)
    pd.testing.assert_frame_equal(destination.to_pandas(), source.to_pandas())


FIT_FAULTS = ['estimator', 'feature_order', 'target', 'task', 'row_count']
PLAN_FAULTS = ['missing_estimator', 'empty_features', 'target_feature', 'invalid_task', 'invalid_rows']


def _corrupt(payload, fault, alternative):
    plan, fit = payload['plan'], payload['fit_result']
    if fault == 'estimator':
        payload['fit_result'] = replace(fit, estimator=alternative.fit_result.estimator)
    elif fault == 'feature_order':
        payload['fit_result'] = replace(fit, feature_columns=tuple(reversed(fit.feature_columns)))
    elif fault == 'target':
        payload['fit_result'] = replace(fit, target_column='other_target')
    elif fault == 'task':
        payload['fit_result'] = replace(fit, task='classification' if plan.task == 'regression' else 'regression')
    elif fault == 'row_count':
        payload['fit_result'] = replace(fit, n_train_rows=fit.n_train_rows+1)
    elif fault == 'missing_estimator':
        payload['plan'] = replace(plan, estimator_=None)
    elif fault == 'empty_features':
        payload['plan'] = replace(plan, feature_columns=())
    elif fault == 'target_feature':
        payload['plan'] = replace(plan, feature_columns=(*plan.feature_columns, plan.target_column))
    elif fault == 'invalid_task':
        payload['plan'] = replace(plan, task='not-a-task')
    else:
        payload['plan'] = replace(plan, n_train_rows=0)
    return payload


@pytest.mark.parametrize('fault', FIT_FAULTS+PLAN_FAULTS)
def test_inconsistent_automl_payload_refuses_before_session_replacement(fitted, fault, tmp_path):
    source, alternative = fitted
    root = tmp_path / 'automl'
    source.automl.save_bundle(root)
    destination = _destination(source, root)
    payload = _corrupt(joblib.load(root / 'automl_plan.joblib'), fault, alternative)
    joblib.dump(payload, root / 'automl_plan.joblib')
    before = dict(vars(destination))
    fingerprint = joblib.hash(destination.automl_plan)
    with pytest.raises(ValidationError):
        destination.automl.load_bundle(root, trusted=True)
    assert vars(destination).keys() == before.keys()
    assert all(vars(destination)[name] is value for name, value in before.items())
    assert joblib.hash(destination.automl_plan) == fingerprint


@pytest.mark.parametrize('fault', FIT_FAULTS+PLAN_FAULTS)
def test_inconsistent_automl_save_refuses_before_creating_destination(fitted, fault, tmp_path):
    source, alternative = fitted
    payload = _corrupt({'plan': source.automl_plan, 'fit_result': source.fit_result}, fault, alternative)
    destination = tmp_path / 'invalid'
    with pytest.raises(ValidationError):
        save_automl_bundle(destination, payload['plan'], fit_result=payload['fit_result'])
    assert not destination.exists()
