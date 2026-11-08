"""Multi-task target identity and mixed-head numeric boundary consumers."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.multitask.features import decode_multitask_predictions, encode_multitask_y


@pytest.mark.parametrize('task', ['classification', 'mixed'])
@pytest.mark.parametrize('labels', [[1, 2, 1, 2], [.5, 1.5, .5, 1.5],
                                   ['001', '010', '001', '010'], ['.50', '1.00', '.50', '1.00']])
def test_encoding_roundtrip_preserves_original_class_values(task, labels):
    frame = pd.DataFrame({'label': labels, 'other': [2., 3., 4., 5.]})
    kinds = {'label': 'classification', 'other': 'regression'}
    columns = ['label', 'other'] if task == 'mixed' else ['label']
    encoded, encoders, classes = encode_multitask_y(frame, columns, task=task, task_kinds=kinds)
    import json
    json.dumps(classes)
    repeated, _, _ = encode_multitask_y(frame, columns, task=task, task_kinds=kinds, label_encoders=encoders)
    np.testing.assert_array_equal(repeated, encoded)
    decoded = decode_multitask_predictions(encoded, columns, task=task,
                                           label_encoders=encoders, task_kinds=kinds)
    assert decoded['label'] == tuple(labels)
    assert all(type(a) is type(b) for a, b in zip(decoded['label'], labels, strict=True))
    if task == 'mixed':
        np.testing.assert_array_equal(decoded['other'], frame.other)


@pytest.mark.parametrize('column', ['label', 'other'])
@pytest.mark.parametrize('value', [None, np.nan, pd.NA])
def test_mixed_target_nulls_are_refused_before_encoding(column, value):
    frame = pd.DataFrame({'label': ['a', 'b', 'a'], 'other': [1., 2., 3.]})
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValidationError, match='null'):
        encode_multitask_y(frame, ['label', 'other'], task='mixed',
            task_kinds={'label': 'classification', 'other': 'regression'})


@pytest.mark.parametrize('task', ['mixed', 'regression'])
@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_nonfinite_regression_targets_are_refused(task, value):
    frame = pd.DataFrame({'a': [1., value, 3.], 'b': [3., 2., 1.]})
    with pytest.raises(ValidationError, match='finite'):
        encode_multitask_y(frame, ['a', 'b'], task=task,
                           task_kinds={'a': 'regression', 'b': 'regression'})


@pytest.mark.parametrize('kinds', [{'a': 'regression'}, {'a': 'regression', 'b': 'typo'}])
def test_mixed_metadata_must_describe_every_requested_head(kinds):
    frame = pd.DataFrame({'a': [1., 2., 3.], 'b': ['x', 'y', 'x']})
    with pytest.raises(ValidationError):
        encode_multitask_y(frame, ['a', 'b'], task='mixed', task_kinds=kinds)


@pytest.mark.parametrize('partition', ['train', 'validation', 'test', 'all'])
def test_real_multitask_predictions_and_accuracy_preserve_string_class_identity(partition):
    rng = np.random.default_rng(32)
    x = rng.normal(size=100)
    frame = pd.DataFrame({'x': x, 'z': rng.normal(size=100),
        'a': np.where(x > 0, '001', '010'), 'b': np.where(x > 0, '.50', '1.00')}, index=[8]*100)
    session = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature',
        'a': 'target', 'b': 'target'}).split(test_size=.2, validation_size=.2, random_state=4)
    session.multitask.fit(task='classification', base_estimator='logistic_regression',
                          prefer_reduce_components=False)
    part = session.to_pandas() if partition == 'all' else session.partition(partition)
    prediction = session.multitask.predict(partition=partition)
    evaluation = session.multitask.evaluate(partition=partition)
    notes = ' '.join(evaluation.disclosures)
    assert ('in-sample' in notes) == (partition in {'train', 'all'})
    raw = session.multitask_plan.estimator_.predict(part[['x', 'z']].to_numpy())
    for index, target in enumerate(['a', 'b']):
        labels = np.array(sorted(frame[target].unique()))
        expected = labels[np.asarray(raw[:, index], dtype=int)]
        assert prediction.predictions[target] == tuple(expected)
        assert evaluation.per_task_metrics[target]['accuracy'] == pytest.approx(
            np.mean(part[target].to_numpy() == expected))


def test_real_torch_mixed_heads_report_independent_heldout_metrics():
    pytest.importorskip('torch')
    rng = np.random.default_rng(62)
    x = rng.normal(size=80)
    frame = pd.DataFrame({'x': x, 'z': rng.normal(size=80),
        'label': np.where(x > 0, '001', '010'), 'amount': 3*x + rng.normal(size=80)}, index=[4]*80)
    session = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature',
        'label': 'target', 'amount': 'target'}).split(test_size=.2, random_state=3)
    session.multitask.fit(backend='torch', method='shared_trunk_multihead', task='auto',
        epochs=3, batch_size=16, prefer_reduce_components=False)
    assert session.multitask_plan.task == 'mixed'
    part = session.partition('test')
    predicted = session.multitask.predict(partition='test').predictions
    evaluation = session.multitask.evaluate(partition='test')
    assert set(predicted['label']) <= {'001', '010'}
    assert evaluation.per_task_metrics['label']['accuracy'] == pytest.approx(
        np.mean(part.label.to_numpy() == np.asarray(predicted['label'])))
    residual = part.amount.to_numpy() - np.asarray(predicted['amount'])
    assert evaluation.per_task_metrics['amount']['mae'] == pytest.approx(np.abs(residual).mean())
    assert evaluation.per_task_metrics['amount']['rmse'] == pytest.approx(np.sqrt(np.square(residual).mean()))


@pytest.mark.parametrize('task', ['classification', 'mixed'])
def test_incompatible_class_label_types_are_refused_without_string_collision(task):
    frame = pd.DataFrame({'label': ['1', 1, 'a'], 'amount': [1., 2., 3.]})
    with pytest.raises(ValidationError, match='class-label types'):
        encode_multitask_y(frame, ['label'], task=task, task_kinds={'label': 'classification'})
