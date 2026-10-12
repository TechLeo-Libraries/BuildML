"""Federated learning must remain invariant to equivalent class-label names."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

from buildml import Session
from buildml.federated.adapters.flower import _BuildMLSklearnClient, _ClientPartition


def _frame(labels):
    rng = np.random.default_rng(431)
    codes = np.tile([0, 1], 120)
    return pd.DataFrame({'x': (2 * codes - 1) * 3 + rng.normal(0, .1, len(codes)),
                         'z': rng.normal(0, .1, len(codes)),
                         'y': np.asarray(labels)[codes],
                         'client': np.arange(len(codes)) % 3})


@pytest.mark.parametrize('labels', [(3, 7), ('negative', 'positive')])
@pytest.mark.parametrize('method', ['fedavg', 'fedprox'])
def test_native_federated_classification_decodes_noncanonical_labels(labels, method):
    frame = _frame(labels)
    session = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'y': 'target', 'client': 'group'}).split(test_size=.2, validation_size=.2, random_state=5)
    before = session.partition('test').copy()
    session.federated.fit(backend='native', method=method, mu=.05 if method == 'fedprox' else 0, n_rounds=2, local_epochs=1, client_column='client', prefer_reduce_components=False)
    result = session.federated.evaluate(partition='test')
    assert result.metrics['accuracy'] >= .9
    prediction = session.federated.predict(partition='test')
    assert set(prediction.predictions).issubset(set(labels))
    assert len(prediction.predictions) == len(before)
    pd.testing.assert_frame_equal(session.partition('test'), before)


def test_local_flower_client_evaluation_matches_independent_classifier_error():
    frame = _frame(('negative', 'positive'))
    encoder = LabelEncoder().fit(frame.y)
    x = frame[['x', 'z']].to_numpy()
    y = encoder.transform(frame.y)
    model = SGDClassifier(loss='log_loss', random_state=5).fit(x, y)
    client = _BuildMLSklearnClient(_ClientPartition('client-a', frame, len(frame)), global_template=model,
        columns=['x', 'z'], target_column='y', task='classification', estimator_key='sgd_classifier',
        local_epochs=1, mu=0, label_encoder=encoder, classes=tuple(encoder.classes_), random_state=5)
    parameters = client.get_parameters({})
    loss, count, _ = client.evaluate(parameters, {})
    assert count == len(frame)
    assert loss == pytest.approx(float(np.mean(model.predict(x) != y)))
    saved = model.coef_.copy()
    parameters[0][:] = 999
    np.testing.assert_array_equal(model.coef_, saved)


@pytest.mark.parametrize("fitted", [False, True])
def test_linear_clone_uses_explicit_classes_and_independent_parameter_storage(fitted):
    from buildml.federated.features import clone_estimator_with_params

    template = SGDClassifier(loss="log_loss", random_state=7)
    if fitted:
        template.fit(np.array([[1., 0.], [0., 1.]]), np.array([3, 7]))
    parameters = {"coef_": np.array([[1., -1.]]), "intercept_": np.array([0.])}
    clone = clone_estimator_with_params(template, parameters, classes=np.array([-2, 8]))
    np.testing.assert_array_equal(clone.predict(np.array([[2., 1.], [1., 2.]])), [8, -2])
    clone.coef_[:] = 99
    np.testing.assert_array_equal(parameters["coef_"], [[1., -1.]])
