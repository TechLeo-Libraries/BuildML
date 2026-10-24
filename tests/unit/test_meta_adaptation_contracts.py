"""Few-shot adaptation must score held-out rows without changing meta weights."""
import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session


def _tasks():
    rng = np.random.default_rng(391)
    labels = np.tile(np.arange(2), 240)
    frame = pd.DataFrame({
        'x': (2 * labels - 1) * 3 + rng.normal(0, .1, len(labels)),
        'z': rng.normal(0, .1, len(labels)),
        'label': labels,
        'task_id': np.repeat(np.arange(8), 60),
    })
    return Session.ingest(frame).set_roles({
        'x': 'feature', 'z': 'feature', 'label': 'target', 'task_id': 'group',
    }).split(test_size=.3, validation_size=.2, random_state=18)


@pytest.mark.parametrize('method', ['reptile', 'maml', 'prototypical_torch'])
def test_meta_adaptation_preserves_trained_weights_and_holdout(method, tmp_path):
    torch = pytest.importorskip('torch')
    session = _tasks()
    test_before = session.partition('test').copy()
    session.metalearning.fit(
        backend='torch' if method == 'prototypical_torch' else 'industry',
        method=method, k_shot=2, n_query=3, n_episodes=3,
        meta_epochs=2, inner_steps=2, hidden_dim=8, embed_dim=4,
        task_holdout_fraction=.25, prefer_reduce_components=False,
        random_state=7,
    )
    plan = session.metalearning_plan
    learner = plan.meta_learner_
    module = learner.module_ if hasattr(learner, 'module_') else learner.encoder_
    weights = {name: value.detach().clone() for name, value in module.state_dict().items()}
    adaptation = session.metalearning.adapt(task_id=plan.train_task_ids[0], partition='train', max_support_per_class=2)
    assert adaptation.n_support == 4
    result = session.metalearning.evaluate(partition='test', k_shot=2, n_query=3)
    assert result.n_tasks_evaluated > 0
    assert result.n_query_rows > 0
    assert 0 <= result.metrics['mean_accuracy'] <= 1
    json.dumps(result.to_dict(), allow_nan=False)
    for name, weight in module.state_dict().items():
        assert torch.equal(weight, weights[name]), f'Adaptation mutated meta weight {name}'
    pd.testing.assert_frame_equal(session.partition('test'), test_before)
    destination = tmp_path / method
    session.metalearning.save_bundle(destination)
    restored = _tasks()
    restored.metalearning.load_bundle(destination, trusted=True)
    repeated = restored.metalearning.evaluate(partition='test', k_shot=2, n_query=3)
    assert repeated.metrics == result.metrics
