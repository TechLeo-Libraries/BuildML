"""Agent SSL budgets and training boundaries are checked with real encoders."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _session():
    rng = np.random.default_rng(75)
    frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=['a', 'b', 'c'])
    frame['y'] = (frame.a > 0).astype(int)
    return (Session.ingest(frame).set_roles({'y': 'target'})
            .split(test_size=.25, validation_size=.25, random_state=3))


def _run(session, tool, **args):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(tool, args, registry),
                          confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


@pytest.mark.parametrize('method', [
    'simclr_tabular', 'byol_tabular', 'vicreg_tabular', 'mae_tabular', 'vae_tabular',
])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_encoder_honors_agent_budget_and_does_not_learn_from_held_out_rows(method, device, tmp_path):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required for actual GPU encoder verification')
    baseline, changed = _session(), _session()
    train = list(changed.split_plan.train_indices)
    held_out = list(changed.split_plan.test_indices) + list(changed.split_plan.validation_indices)
    changed.dataset._ensure_pandas().loc[held_out, ['a', 'b', 'c']] = 1e6
    untouched = changed.to_pandas().copy(deep=True)
    options = dict(method=method, columns=['a', 'b', 'c'], latent_dim=3,
                   epochs=2, batch_size=8, learning_rate=.004, device=device, random_state=17)
    _run(changed, 'fit_ssl_pretext', **options)
    baseline.ssl.fit_pretext(**options)
    plan = changed.ssl_plan
    for name in ('method', 'epochs', 'batch_size', 'learning_rate', 'device', 'random_state'):
        assert plan.config[name] == options[name]
    assert plan.n_train_rows == len(train)
    numeric_train = untouched.loc[train, ['a', 'b', 'c']].to_numpy(dtype=float)
    np.testing.assert_allclose(plan.encoder_.transform(numeric_train),
                               baseline.ssl_plan.encoder_.transform(numeric_train), rtol=1e-6, atol=1e-6)
    _run(changed, 'finetune_ssl_head', estimator='logistic_regression')
    score = _run(changed, 'evaluate_ssl', partition='test')
    assert score.n_rows == len(changed.split_plan.test_indices)
    assert np.isfinite(list(score.metrics.values())).all()
    _run(changed, 'save_ssl_bundle', path=str(tmp_path / method))
    assert changed.ssl.evaluate(partition='test').metrics == score.metrics
    restored = _session()
    registry = build_default_registry()
    proposal = propose_tool_execution('load_ssl_bundle', {'path': str(tmp_path / method)}, registry)
    denied = execute_tool(restored, proposal, confirmed=True, registry=registry)
    assert not denied.executed
    assert 'trust' in denied.error.lower()
    pd.testing.assert_frame_equal(changed.to_pandas(), untouched)
