"""TDA class identities survive public prediction, scoring and bundle replay."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan


def population(labels):
    rng = np.random.default_rng(812)
    x = rng.normal(size=(32, 3))
    x[20:] += [7, -4, 2]
    frame = pd.DataFrame(x, columns=['a', 'b', 'c'], index=[f'row-{i // 2}' for i in range(32)])
    frame['target'] = pd.Series([labels[i % len(labels)] for i in range(32)], dtype=object).to_numpy()
    session = Session.ingest(frame).set_roles({'target': 'target'})
    session._split_plan = SplitPlan('injected', .1875, .1875, None, None,
                                    tuple(range(20)), tuple(range(20, 26)), tuple(range(26, 32)))
    return session


def require_backend(backend):
    pytest.importorskip('ripser')
    pytest.importorskip('persim')
    if backend == 'giotto':
        pytest.importorskip('gtda')


def fit_options(backend):
    return dict(backend=backend, vectorization='landscape', knn=5,
                n_bins=5, n_layers=2, random_state=7)


LABELS = [('001', '010'), ('1.00', '2.50'), (False, True), (-3, 7)]


@pytest.fixture(scope='module', params=[(b, labels) for b in ['native', 'giotto'] for labels in LABELS])
def fitted(request):
    backend, labels = request.param
    require_backend(backend)
    session = population(labels)
    session.tda.fit(**fit_options(backend))
    return session, labels


def macro_f1(truth, predictions):
    scores = []
    for label in set(truth) | set(predictions):
        tp = sum(t == label and p == label for t, p in zip(truth, predictions, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predictions, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predictions, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return float(np.mean(scores))


def learned_state(plan):
    # sklearn trees update query counters; these do not change the fitted index.
    return joblib.hash((plan.train_x_, plan.nn_._fit_X, plan.mean_, plan.scale_,
                        plan.vectorizer_state_, plan.head_estimator_, plan.label_encoder_))


@pytest.mark.parametrize('replay', [False, True])
def test_scalar_predictions_and_metrics_use_original_train_vocabulary(fitted, replay, tmp_path):
    source, labels = fitted
    session = Session(dataset=source.dataset, split_plan=source.split_plan)
    if replay:
        session.tda.load_bundle(source.tda.save_bundle(tmp_path / 'tda'), trusted=True)
    else:
        session._tda_plan = source.tda_plan
    plan = session.tda_plan
    vocabulary = sorted(labels, key=str)
    train_codes = np.array([vocabulary.index(labels[i % len(labels)]) for i in range(20)])
    reference = clone(plan.head_estimator_).fit(plan.train_tda_features_, train_codes)
    fingerprint = learned_state(plan)
    for partition in ['validation', 'test']:
        features = session.tda.transform(partition=partition).features
        expected = [vocabulary[int(code)] for code in reference.predict(features)]
        predictions = session.tda.predict(partition=partition).predictions
        assert list(predictions) == expected
        assert [type(value) for value in predictions] == [type(value) for value in expected]
        positions = getattr(session.split_plan, partition + '_indices')
        truth = [labels[i % len(labels)] for i in positions]
        metrics = session.tda.evaluate(partition=partition).metrics
        assert metrics['accuracy'] == pytest.approx(np.mean(np.array(truth, dtype=object) == expected))
        assert metrics['macro_f1'] == pytest.approx(macro_f1(truth, expected))
    assert learned_state(plan) == fingerprint


@pytest.mark.parametrize('backend', ['native', 'giotto'])
@pytest.mark.parametrize('labels', [(1, '1'), (True, 'True'), (1.5, '1.5')])
def test_ambiguous_scalar_labels_refuse_before_replacing_fitted_state(backend, labels):
    require_backend(backend)
    source = population(('left', 'right'))
    source.tda.fit(**fit_options(backend))
    before = source.tda_plan
    history = len(source.history)
    source.dataset._ensure_pandas()['target'] = pd.Series(
        [labels[i % 2] for i in range(32)], dtype=object).to_numpy()
    with pytest.raises(ValidationError, match='Ambiguous'):
        source.tda.fit(**fit_options(backend))
    assert source.tda_plan is before
    assert len(source.history) == history


@pytest.mark.parametrize('backend', ['native', 'giotto'])
@pytest.mark.parametrize('fault', [dict(knn=1), dict(maxdim=-1), dict(homology_dims=()),
                                  dict(homology_dims=(2,), maxdim=1), dict(homology_dims=(-1,)),
                                  dict(n_bins=1), dict(n_layers=0)])
def test_invalid_topology_configuration_refuses_atomically(backend, fault):
    require_backend(backend)
    source = population(('left', 'right'))
    source.tda.fit(**fit_options(backend))
    before = source.tda_plan
    fingerprint = joblib.hash(before)
    history = len(source.history)
    with pytest.raises(ValidationError):
        source.tda.fit(**(fit_options(backend) | fault))
    assert source.tda_plan is before
    assert joblib.hash(before) == fingerprint
    assert len(source.history) == history
