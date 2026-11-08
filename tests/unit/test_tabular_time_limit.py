"""Time limits stop rollouts while preserving the TD bootstrap value."""

import numpy as np
import pytest

from buildml.rl.tabular import evaluate_tabular_policy, train_tabular_control


@pytest.fixture
def make_env_id():
    gym = pytest.importorskip("gymnasium")
    registered = []

    def register(legacy, time_limit, box):
        class OneState(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.Discrete(1)
                if box:
                    self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
                else:
                    self.observation_space = gym.spaces.Discrete(1)

            def _observation(self):
                return np.zeros(1, dtype=np.float32) if box else 0

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                observation = self._observation()
                return observation if legacy else (observation, {})

            def step(self, action):
                assert action == 0
                observation = self._observation()
                if legacy:
                    return observation, 1.0, True, {"TimeLimit.truncated": time_limit}
                return observation, 1.0, not time_limit, time_limit, {}

        env_id = f"BuildMLTabularTimeLimit{len(registered)}-v0"
        gym.register(env_id, entry_point=OneState, disable_env_checker=True, order_enforce=False)
        registered.append(env_id)
        return env_id

    yield register
    for env_id in registered:
        gym.registry.pop(env_id)


@pytest.mark.parametrize("box", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("time_limit", [False, True])
@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa", "expected_sarsa"])
def test_tabular_time_limit_bootstrap_and_rollout_length(
    make_env_id, box, legacy, time_limit, algorithm
):
    env_id = make_env_id(legacy, time_limit, box)
    policy, train, _, _ = train_tabular_control(
        env_id=env_id,
        algorithm=algorithm,
        n_episodes=3,
        max_steps=5,
        learning_rate=1.0,
        gamma=0.5,
        epsilon=0,
        epsilon_min=0,
    )
    state = policy.state_index([0.0] if box else 0)
    # For truncation, the one-state Bellman updates are 1, 1.5, then 1.75.
    assert policy.q_table[state, 0] == pytest.approx(1.75 if time_limit else 1.0)
    assert train["total_steps"] == 3
    assert train["mean_return"] == 1.0
    evaluation = evaluate_tabular_policy(policy, n_episodes=2, max_steps=5)
    assert evaluation["mean_episode_steps"] == 1.0
    assert evaluation["mean_return"] == 1.0
