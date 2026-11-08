"""Native first-order MAML updates agree with an independent one-step oracle."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from buildml.metalearning.adapters.industry_maml import meta_train_maml
from buildml.metalearning.extras import learn2learn_available
from buildml.metalearning.features import (
    encode_labels,
    frame_for_task,
    matrix_from_frame,
    sample_support_query,
)


def train(frame, sampler, inner_steps):
    return meta_train_maml(frame, task_column="task", target_column="y", columns=["x", "z"],
        label_encoder=LabelEncoder().fit(["0", "1"]), meta_train_ids=["one"],
        n_way=2, k_shot=3, n_query=3, n_episodes=1, rng=np.random.default_rng(7),
        n_classes=2, meta_epochs=1, inner_lr=0.07, inner_steps=inner_steps,
        meta_lr=0.003, hidden_dim=8, random_state=7, frame_for_task=frame_for_task,
        sample_support_query=sampler, matrix_from_frame=matrix_from_frame, encode_labels=encode_labels)


@pytest.mark.parametrize("seed", [2, 19])
@pytest.mark.parametrize("inner_steps", [1, 3])
def test_native_outer_update_matches_first_order_adam(seed, inner_steps):
    torch = pytest.importorskip("torch")
    if learn2learn_available():
        pytest.skip("Native fallback requires learn2learn to be absent")
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"x": rng.normal(size=32), "z": rng.normal(size=32),
                          "y": [0, 1] * 16, "task": "one"})
    episodes = []

    def record_episode(*args, **kwargs):
        episode = sample_support_query(*args, **kwargs)
        episodes.append(episode)
        return episode

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
        torch.manual_seed(seed)
        learner, score, notes, warnings = train(frame, record_episode, inner_steps)
    assert len(episodes) == 1 and episodes[0] is not None
    support, query, _ = episodes[0]
    xs = torch.tensor(support[["x", "z"]].to_numpy(), dtype=torch.float32)
    ys = torch.tensor(support["y"].to_numpy(), dtype=torch.long)
    xq = torch.tensor(query[["x", "z"]].to_numpy(), dtype=torch.float32)
    yq = torch.tensor(query["y"].to_numpy(), dtype=torch.long)
    adapted = copy.deepcopy(initial)
    # Explicit gradient descent, independent of the adapter's SGD optimizer.
    for _ in range(inner_steps):
        gradients = torch.autograd.grad(torch.nn.functional.cross_entropy(adapted(xs), ys), tuple(adapted.parameters()))
        with torch.no_grad():
            for parameter, gradient in zip(adapted.parameters(), gradients):
                parameter.sub_(0.07 * gradient)
    query_gradients = torch.autograd.grad(torch.nn.functional.cross_entropy(adapted(xq), yq), tuple(adapted.parameters()))
    changes = []
    for before, actual, gradient in zip(initial.parameters(), learner.module_.parameters(), query_gradients):
        # Adam's bias-corrected first update: m_hat=g, v_hat=g^2.
        expected = before.detach() - 0.003 * gradient / (gradient.abs() + 1e-8)
        np.testing.assert_allclose(actual.detach().numpy(), expected.numpy(), atol=2e-7, rtol=1e-6)
        changes.append(not torch.equal(before, actual))
    assert any(changes), "Meta-training must update the initialization"
    with torch.no_grad():
        expected_score = float((adapted(xq).argmax(1) == yq).float().mean())
    assert score == pytest.approx(expected_score, abs=1e-7)
    assert learner.maml_wrapper_ is None
    assert any("native" in note.lower() for note in notes)
    assert not warnings


def test_no_valid_episode_reports_no_training_score():
    pytest.importorskip("torch")
    if learn2learn_available():
        pytest.skip("Native fallback requires learn2learn to be absent")
    frame = pd.DataFrame({"x": [0., 1.], "z": [1., 0.], "y": [0, 1], "task": "one"})
    _, score, _, warnings = train(frame, sample_support_query, 1)
    assert score is None
    assert any("No successful MAML episodes" in warning for warning in warnings)
