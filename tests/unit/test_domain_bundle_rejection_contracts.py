"""Malformed domain bundles fail before changing an existing Session."""
import importlib
import json

import joblib
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

BUNDLES = [
    ('activelearning', 'activelearning', 'active_learning'),
    ('anomaly', 'anomaly', 'anomaly'),
    ('automl', 'automl', 'automl'),
    ('causal', 'causal', 'causal'),
    ('federated', 'federated', 'federated'),
    ('probabilistic', 'probabilistic', 'probabilistic'),
    ('metalearning', 'metalearning', 'metalearning'),
    ('forecasting', 'forecast', 'forecast'),
    ('graph', 'graph', 'graph'),
    ('recommenders', 'recommender', 'recommender'),
    ('ranking', 'ranker', 'ranker'),
    ('multitask', 'multitask', 'multitask'),
    ('online', 'online', 'online'),
    ('optimize', 'decision', 'decision'),
    ('semisupervised', 'semisupervised', 'semisupervised'),
    ('symbolic', 'symbolic', 'symbolic'),
    ('kg', 'kg', 'kg'),
    ('tda', 'tda', 'tda'),
    ('synthetic', 'synthetic', 'synthetic'),
    ('unsupervised', 'cluster', 'unsupervised'),
]


@pytest.mark.parametrize('package,payload_name,session_name', BUNDLES)
@pytest.mark.parametrize('fault', ['missing_payload', 'foreign_format', 'missing_plan', 'wrong_plan_type'])
def test_invalid_bundle_does_not_mutate_session(package, payload_name, session_name, fault, tmp_path):
    checkpoint = importlib.import_module(f'buildml.{package}.checkpoint')
    root = tmp_path / package
    root.mkdir()
    metadata = {'format': checkpoint.BUNDLE_FORMAT}
    if fault == 'foreign_format':
        metadata['format'] = 'buildml.foreign_bundle.v99'
    (root / 'meta.json').write_text(json.dumps(metadata), encoding='utf-8')
    if fault != 'missing_payload':
        joblib.dump({} if fault == 'missing_plan' else {'plan': 42}, root / f'{payload_name}_plan.joblib')
    session = Session.ingest(pd.DataFrame({'x': [1, 2], 'y': [0, 1]}))
    before = dict(vars(session))
    method = getattr(session, f'load_{session_name}_bundle')
    with pytest.raises(ValidationError):
        method(root, trusted=True)
    assert vars(session).keys() == before.keys()
    for name, value in before.items():
        assert vars(session)[name] is value, f'Invalid {package} bundle changed {name}'
