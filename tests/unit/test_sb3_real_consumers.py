"""Real SB3 interaction accounting, saved actions and episode-return consumers."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('stable_baselines3')
import gymnasium as gym

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.rl.adapters.stable_baselines3 import (
    act_sb3_observation,
    evaluate_sb3_policy,
    train_sb3_policy,
)


@pytest.fixture(scope='module', params=['ppo', 'dqn', 'a2c'])
def trained(request):
    return train_sb3_policy(algorithm=request.param, total_timesteps=129,
                            max_steps=12, random_state=4)


def test_training_reports_real_executed_interactions(trained):
    policy, metrics, disclosures, warnings = trained
    actual = policy.model.num_timesteps
    assert actual > 129  # These algorithms complete rollout blocks.
    assert metrics['total_timesteps'] == actual
    assert metrics['requested_timesteps'] == 129
    assert policy.total_timesteps == actual
    assert any('requested' in text and 'executed' in text for text in disclosures)
    assert any('rollout' in text for text in warnings)


def test_evaluation_returns_match_independent_episode_replay(trained):
    policy, _, _, _ = trained
    report = evaluate_sb3_policy(policy, n_episodes=3, max_steps=12, random_state=9)
    returns = []
    env = gym.make('CartPole-v1')
    try:
        for episode in range(3):
            observation, _ = env.reset(seed=10009+episode)
            rewards = []
            for _ in range(12):
                action, _ = policy.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(action))
                rewards.append(reward)
                if terminated or truncated:
                    break
            returns.append(sum(rewards))
    finally:
        env.close()
    for key, expected in [('mean_return', np.mean(returns)), ('std_return', np.std(returns)),
                          ('min_return', min(returns)), ('max_return', max(returns))]:
        assert report[key] == pytest.approx(expected)
    assert report['n_eval_episodes'] == 3


def test_saved_model_actions_and_onehot_indicators_match_real_model(trained, tmp_path):
    policy, _, _, _ = trained
    path = tmp_path / 'policy.zip'
    policy.model.save(path)
    restored = replace(policy, model=type(policy.model).load(path))
    for observation in [[0., 0., 0., 0.], [.1, .3, -.1, -.2]]:
        expected, _ = policy.model.predict(np.asarray(observation), deterministic=True)
        action, indicators = act_sb3_observation(restored, observation)
        assert action == int(expected)
        np.testing.assert_array_equal(indicators, np.eye(2)[int(expected)])


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_invalid_training_interaction_count_is_refused(value):
    with pytest.raises(ValidationError):
        train_sb3_policy(algorithm='dqn', total_timesteps=value, max_steps=2)


@pytest.mark.parametrize('field', ['n_episodes', 'max_steps'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_evaluation_counts_are_refused(trained, field, value):
    with pytest.raises(ValidationError):
        evaluate_sb3_policy(trained[0], **{field: value})


def test_explicit_sb3_session_does_not_require_the_separate_imitation_package():
    import pandas as pd

    session = Session.ingest(pd.DataFrame({'x': [0., 1.], 'y': [0, 1]})).set_roles({'y': 'target'})
    from buildml.rl.catalog import rl_capability_matrix
    from buildml.rl.extras import imitation_spec_present

    matrix = rl_capability_matrix()
    assert matrix['rl_backends']['industry']['available']
    if not imitation_spec_present():
        assert not matrix['imitation_backends']['industry']['available']
    result = session.rl.fit(backend='industry', mode='gym_sb3', algorithm='dqn',
                            total_timesteps=129, max_steps=3, random_state=5)
    assert result.n_train_rows == session.rl_plan.policy_.model.num_timesteps
