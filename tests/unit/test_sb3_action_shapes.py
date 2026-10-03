"""Single-environment SB3 actions may be scalars or one-element arrays."""

from types import SimpleNamespace

import numpy as np
import pytest

from buildml.rl.adapters.stable_baselines3 import SB3PolicyWrapper, evaluate_sb3_policy


@pytest.mark.parametrize("action", [np.int64(1), np.array([1]), np.array([[1]])])
def test_single_action_prediction_and_rollout(action, monkeypatch):
    model = SimpleNamespace(predict=lambda observation, deterministic: (action, None))
    policy = SB3PolicyWrapper(model, "example", "ppo", obs_dim=4, n_actions=2)
    assert policy.predict(np.zeros(4)) == (1, None)
    received = []
    closed = []

    def step(value):
        assert type(value) is int
        received.append(value)
        return np.zeros(4), 2.0, True, False, {}

    env = SimpleNamespace(
        reset=lambda seed: (np.zeros(4), {}), step=step,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        "buildml.rl.adapters.stable_baselines3.require_gymnasium",
        lambda **kwargs: SimpleNamespace(make=lambda name: env),
    )
    result = evaluate_sb3_policy(policy, n_episodes=2)
    assert received == [1, 1]
    assert result["mean_return"] == 2.0
    assert closed == [True]
