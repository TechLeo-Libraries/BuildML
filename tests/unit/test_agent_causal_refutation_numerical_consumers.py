"""Agent native causal sensitivity results agree with independent nuisance algebra."""

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, arguments, success=True):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(name, arguments, registry),
                          confirmed=True, registry=registry)
    assert result.executed is success, result.error
    return result.result if success else result.error


def _session(labels, poison):
    rng = np.random.default_rng(823)
    x = rng.normal(size=(192, 2))
    treatment = (rng.random(192) < 1/(1+np.exp(-.6*x[:, 0]))).astype(int)
    y = 2.3*treatment + .5*x[:, 0] - x[:, 1] + rng.normal(0, .2, 192)
    if poison:
        x[128:] = np.nan
        y[128:] = 1e12
    frame = pd.DataFrame({'x': x[:, 0], 'z': x[:, 1], 't': np.asarray(labels)[treatment], 'y': y})
    s = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature', 't': 'ignore', 'y': 'target'})
    s.inject_split(train_indices=list(range(128)), validation_indices=list(range(128, 160)),
                   test_indices=list(range(160, 192)))
    return s, x[:128], treatment[:128], y[:128]


def _effect(x, t, y, model):
    means = []
    for arm in (0, 1):
        training, target = x[t == arm], y[t == arm]
        center, divisor = training.mean(0), training.std(0)
        scaled = (training-center)/divisor
        design = np.column_stack([np.ones(len(training)), scaled])
        penalty = np.eye(design.shape[1])*(1 if model == 'ridge' else 0)
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        means.append(np.column_stack([np.ones(len(x)), (x-center)/divisor]) @ coefficients)
    return float(np.mean(means[1]-means[0]))


@pytest.mark.parametrize('labels', [('off', 'on'), (10, 20), (False, True)])
@pytest.mark.parametrize('model', ['ridge', 'linear_regression'])
@pytest.mark.parametrize('kind', ['placebo_treatment', 'random_confounder'])
@pytest.mark.parametrize('poison', [False, True])
def test_agent_refutations_match_train_only_algebra_and_bundle_replay(labels, model, kind, poison, tmp_path):
    s, x, t, y = _session(labels, poison)
    before = s.to_pandas().copy(deep=True)
    declared = _run(s, 'declare_causal_assumptions', {'treatment': 't', 'outcome': 'y',
                     'confounders': ['x', 'z'], 'acknowledge_unconfoundedness': True,
                     'acknowledge_positivity': True})
    assert declared['treatment'] == 't' and declared['outcome'] == 'y'
    _run(s, 'fit_causal', {'backend': 'native', 'method': 't_learner',
                          'outcome_model': model, 'bootstrap_samples': 0, 'random_state': 13})
    original = _effect(x, t, y, model)
    assert s.causal.plan.ate == pytest.approx(original, abs=1e-10)
    fingerprint = joblib.hash(s.causal.plan)
    rng = np.random.default_rng(13)
    if kind == 'placebo_treatment':
        expected = _effect(x, rng.permutation(t), y, model)
    else:
        augmented = np.column_stack([x, rng.normal(size=len(x))])
        expected = _effect(augmented, t, y, model)
    result = _run(s, 'refute_causal', {'kind': kind, 'random_state': 13})
    assert result.original_ate == pytest.approx(original, abs=1e-10)
    assert result.refute_ate == pytest.approx(expected, abs=1e-10)
    assert result.ate_shift == pytest.approx(expected-original, abs=1e-10)
    assert result.n_rows == 128 and result.refute_p_value is None
    assert any('does not prove identification' in note for note in result.disclosures)
    assert joblib.hash(s.causal.plan) == fingerprint
    pd.testing.assert_frame_equal(s.to_pandas(), before)
    path = str(tmp_path / 'causal')
    _run(s, 'save_causal_bundle', {'path': path})
    restored, _, _, _ = _session(labels, poison)
    error = _run(restored, 'load_causal_bundle', {'path': path}, success=False)
    assert 'trusted=True' in error and restored.causal.plan is None
    _run(restored, 'load_causal_bundle', {'path': path, 'trusted': True})
    replay = _run(restored, 'refute_causal', {'kind': kind, 'random_state': 13})
    assert replay.refute_ate == pytest.approx(expected, abs=1e-10)
    assert tuple(s.split_plan.train_indices) == tuple(range(128))
