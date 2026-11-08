"""Meta-learning preserves raw class identities through adaptation and persistence."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError
from buildml.metalearning.features import decode_labels, encode_labels

LABELS = [('01', '1'), ('1.0', '1.00'), (-3, 7), (1.25, 2.5), (False, True)]


def _run(session, name, arguments):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(name, arguments, registry),
                          confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


def _session(labels):
    index = np.arange(192)
    y = np.asarray(labels)[index % 2]
    frame = pd.DataFrame({'x': (index % 2) * 5. + index / 1000,
                          'z': index / 100., 'label': y, 'task': index // 48})
    s = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'label': 'target', 'task': 'group'})
    s.inject_split(train_indices=list(range(96)), validation_indices=list(range(96, 144)),
                   test_indices=list(range(144, 192)))
    return s


@pytest.mark.parametrize('labels', LABELS)
@pytest.mark.parametrize('method', ['prototypical', 'warm_start'])
def test_agent_adaptation_labels_and_bundle_roundtrip_are_lossless(labels, method, tmp_path):
    session = _session(labels)
    session.metalearning.fit(backend='sklearn', method=method, k_shot=2,
                             n_query=4, n_way=2, n_episodes=2, random_state=9)
    plan = session.metalearning_plan
    expected_classes = tuple(sorted(labels))
    assert plan.classes_ == expected_classes
    assert tuple(type(v) for v in plan.classes_) == tuple(type(v) for v in expected_classes)
    fingerprint = joblib.hash(plan)
    before = session.to_pandas().copy(deep=True)
    _run(session, 'adapt_to_task', {'task_id': 2, 'partition': 'validation', 'max_support_per_class': 3})
    result = session._metalearning_adapt_result
    assert result.classes_ == expected_classes
    assert tuple(type(v) for v in result.classes_) == tuple(type(v) for v in expected_classes)
    support = session.partition('validation')
    rng = np.random.default_rng(0)
    selected = pd.concat([group.iloc[rng.choice(len(group), 3, replace=False)]
                          for _, group in support.groupby('label', sort=False)])
    if method == 'prototypical':
        assert set(result.prototypes_) == set(labels)
        assert {type(v) for v in result.prototypes_} == {type(v) for v in labels}
        for label in labels:
            expected = selected.loc[selected.label == label, ['x', 'z']].mean().to_numpy()
            np.testing.assert_allclose(result.prototypes_[label], expected)
    else:
        probe = support[['x', 'z']].to_numpy()
        predictions = result.adapted_estimator_.predict(probe)
        decoded = decode_labels(predictions, plan.label_encoder_)
        assert set(decoded) <= set(labels)
        assert {type(v) for v in decoded} <= {type(v) for v in labels}
    assert joblib.hash(plan) == fingerprint
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    path = str(tmp_path / 'meta')
    _run(session, 'save_metalearning_bundle', {'path': path})
    restored = _session(labels)
    _run(restored, 'load_metalearning_bundle', {'path': path, 'trusted': True})
    assert restored.metalearning_plan.classes_ == expected_classes
    _run(restored, 'adapt_to_task', {'task_id': 2, 'partition': 'validation', 'max_support_per_class': 3})
    again = restored._metalearning_adapt_result
    assert again.classes_ == result.classes_
    assert again.prototypes_ == result.prototypes_
    assert tuple(type(v) for v in again.classes_) == tuple(type(v) for v in expected_classes)


@pytest.mark.parametrize('labels', LABELS)
def test_encoder_roundtrip_and_unknown_labels_refuse_without_coercion(labels):
    values = pd.Series(list(labels) * 3)
    codes, encoder, classes = encode_labels(values)
    assert tuple(decode_labels(codes, encoder)) == tuple(values)
    assert tuple(type(v) for v in decode_labels(codes, encoder)) == tuple(type(v) for v in values)
    assert classes == tuple(sorted(labels))
    with pytest.raises(ValidationError, match='unseen|compatible|consistent'):
        encode_labels(pd.Series(['new-class']), label_encoder=encoder)


def test_incompatible_mixed_labels_refuse_instead_of_collapsing():
    with pytest.raises(ValidationError, match='consistent|compatible'):
        encode_labels(pd.Series([1, '1', 2, '2'], dtype=object))


def test_legacy_string_encoder_retains_historical_decode_and_transform():
    encoder = LabelEncoder().fit(['01', '2.5'])
    assert decode_labels(np.array([0, 1]), encoder) == (1, 2.5)
    codes, reused, classes = encode_labels(pd.Series([1, 2.5]), label_encoder=LabelEncoder().fit(['1.0', '2.5']))
    assert reused is not None and classes == ('1.0', '2.5')
    np.testing.assert_array_equal(codes, [0, 1])
