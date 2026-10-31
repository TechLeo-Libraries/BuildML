"""Real agent anomaly scores match support-vector algebra and row-aligned flags."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import OneClassSVM

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, arguments, success=True):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(name, arguments, registry),
                          confirmed=True, registry=registry)
    assert result.executed is success, result.error
    return result.result if success else result.error


def _scores(estimator, rows):
    distances = ((rows[:, None, :] - estimator.support_vectors_[None, :, :]) ** 2).sum(2)
    return -(np.exp(-estimator._gamma * distances) @ estimator.dual_coef_[0]
             + estimator.intercept_[0])


@pytest.mark.parametrize('mode', ['unsupervised', 'novelty'])
@pytest.mark.parametrize('policy', ['quantile', 'decision_zero'])
@pytest.mark.parametrize('partition', ['validation', 'test'])
def test_agent_svm_scores_thresholds_metrics_and_attachments(mode, policy, partition):
    rng = np.random.default_rng(337)
    x = rng.normal(size=(120, 2))
    x[::5] += 4
    labels = (np.arange(120) % 5 == 0).astype(int)
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 'label': labels})
    session = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 'label': 'target'})
    session.inject_split(train_indices=list(range(72)), validation_indices=list(range(72, 96)),
                         test_indices=list(range(96, 120)))
    before = session.to_pandas().copy(deep=True)
    fit = _run(session, 'fit_anomaly', {
        'method': 'one_class_svm', 'backend': 'sklearn', 'mode': mode,
        'contamination': .2, 'threshold_policy': policy, 'normal_label_column': 'label',
        'prefer_reduce_components': False,
    })
    training = x[:72] if mode == 'unsupervised' else x[:72][labels[:72] == 0]
    oracle = OneClassSVM(nu=.05, kernel='rbf', gamma='scale').fit(training)
    plan = session.anomaly_plan
    assert fit.n_fit_rows == len(training)
    np.testing.assert_allclose(plan.estimator_.support_vectors_, oracle.support_vectors_)
    expected_threshold = 0 if policy == 'decision_zero' else np.quantile(_scores(oracle, x[:72]), .8)
    assert fit.threshold == pytest.approx(expected_threshold)
    fingerprint = joblib.hash(plan)
    start = 72 if partition == 'validation' else 96
    rows, truth = x[start:start + 24], labels[start:start + 24]
    scores = _scores(oracle, rows)
    override = float(np.median(scores))
    scored = _run(session, 'score_anomalies', {'partition': partition, 'override_threshold': override})
    np.testing.assert_allclose(scored.scores, scores, atol=1e-12)
    np.testing.assert_array_equal(scored.flags, scores >= override)
    assert scored.threshold_policy == 'override'
    assert joblib.hash(plan) == fingerprint
    evaluation = _run(session, 'evaluate_anomaly', {'partition': partition, 'label_column': 'label', 'k': 7})
    flags = scores >= expected_threshold
    tp = int(np.sum(flags & (truth == 1)))
    fp = int(np.sum(flags & (truth == 0)))
    fn = int(np.sum(~flags & (truth == 1)))
    assert evaluation.labeled_metrics['precision'] == pytest.approx(tp / max(tp + fp, 1))
    assert evaluation.labeled_metrics['recall'] == pytest.approx(tp / max(tp + fn, 1))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:7]
    assert evaluation.labeled_metrics['precision_at_k'] == pytest.approx(truth[top].sum() / 7)
    assert evaluation.alert_rate == pytest.approx(flags.mean())
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    history = list(session._history)
    error = _run(session, 'score_anomalies', {'partition': partition, 'attach': True}, success=False)
    assert "partition='all'" in error
    assert session._history == history
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    attached = _run(session, 'score_anomalies', {'partition': 'all', 'attach': True, 'override_threshold': override})
    all_scores = _scores(oracle, x)
    assert attached.attached
    np.testing.assert_allclose(session.to_pandas()['anomaly_score'], all_scores, atol=1e-12)
    np.testing.assert_array_equal(session.to_pandas()['is_anomaly'], all_scores >= override)
    pd.testing.assert_frame_equal(session.to_pandas()[before.columns], before)
    assert joblib.hash(plan) == fingerprint
    assert tuple(session.split_plan.train_indices) == tuple(range(72))
    history = list(session._history)
    error = _run(session, 'score_anomalies', {'partition': 'all', 'attach': True}, success=False)
    assert 'already exists' in error
    assert session._history == history
