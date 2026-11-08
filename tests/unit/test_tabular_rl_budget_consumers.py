"""Tabular control budgets and finite action probabilities with real Gym rollouts."""

import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.rl import tabular


def policy():
    return tabular.TabularValuePolicy(n_actions=1, n_states=3, algorithm='q_learning',
        discretizer=tabular.ObservationDiscretizer(kind='discrete', n_states=3),
        env_id='unused-invalid-budget')


@pytest.mark.parametrize('operation', ['train', 'evaluate'])
@pytest.mark.parametrize('argument', ['n_episodes', 'max_steps'])
@pytest.mark.parametrize('value', [True, np.bool_(False), 1.5, '2', 0, -1, np.nan, np.inf, None])
def test_invalid_rollout_budgets_are_refused_before_backend_or_environment(operation, argument, value, monkeypatch):
    def forbidden_backend(**kwargs):
        pytest.fail('Invalid rollout budget reached backend/environment allocation')
    monkeypatch.setattr(tabular, 'require_gymnasium', forbidden_backend)
    candidate = policy()
    before = candidate.q_table.copy()
    with pytest.raises(ValidationError, match=argument):
        if operation == 'train':
            tabular.train_tabular_control(**{argument: value})
        else:
            tabular.evaluate_tabular_policy(candidate, **{argument: value})
    np.testing.assert_array_equal(candidate.q_table, before)


@pytest.mark.parametrize('values', [[], [0., np.nan], [0., np.inf], [0., -np.inf], [1j, 2.]])
def test_invalid_action_values_cannot_produce_a_probability_distribution(values):
    with pytest.raises(ValidationError):
        tabular.epsilon_greedy_probabilities(np.array(values), epsilon=.2)


@pytest.mark.parametrize('epsilon', [np.nan, np.inf, -np.inf, 'bad', None])
def test_nonfinite_or_invalid_exploration_rates_are_refused(epsilon):
    with pytest.raises(ValidationError, match='epsilon'):
        tabular.epsilon_greedy_probabilities(np.array([2., 0., 2.]), epsilon=epsilon)


@pytest.mark.parametrize('epsilon', [-.3, 0., .25, 1., 3.])
def test_finite_exploration_clipping_and_ties_match_independent_masses(epsilon):
    bounded = min(1., max(0., epsilon))
    expected = [bounded/3 + (1-bounded)/2, bounded/3, bounded/3 + (1-bounded)/2]
    actual = tabular.epsilon_greedy_probabilities(np.array([2., 0., 2.]), epsilon=epsilon)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert actual.sum() == pytest.approx(1.)


@pytest.fixture
def chain_environment():
    gym = pytest.importorskip('gymnasium')
    instances = []

    class Chain(gym.Env):
        observation_space = gym.spaces.Discrete(3)
        action_space = gym.spaces.Discrete(1)

        def __init__(self, truncated=False):
            super().__init__()
            self.truncated = truncated
            self.closed = False
            self.interactions = 0
            self.position = 0
            instances.append(self)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.position = 0
            return 0, {}

        def step(self, action):
            assert action == 0
            self.interactions += 1
            self.position += 1
            if self.position == 1:
                return 1, 1., False, False, {}
            return (0 if self.truncated else 2), 3., not self.truncated, self.truncated, {}

        def close(self):
            self.closed = True

    name = 'BuildMLBudgetChain-v0'
    gym.register(name, entry_point=Chain)
    try:
        yield name, instances
    finally:
        gym.registry.pop(name, None)


@pytest.mark.parametrize('algorithm', ['q_learning', 'sarsa', 'expected_sarsa'])
@pytest.mark.parametrize('truncated', [False, True])
def test_real_rollout_td_updates_and_evaluation_match_independent_recurrence(chain_environment, algorithm, truncated, monkeypatch):
    name, instances = chain_environment
    gym = pytest.importorskip('gymnasium')
    original_make = gym.make
    # Select the registered environment's documented horizon variant.
    monkeypatch.setattr(gym, 'make', lambda env_id: original_make(env_id, truncated=truncated))
    fitted, metrics, disclosures, warnings = tabular.train_tabular_control(
        env_id=name, algorithm=algorithm, n_episodes=np.int64(40), max_steps=np.int64(5),
        learning_rate=.3, gamma=.4, epsilon=0., epsilon_min=0., random_state=9)
    expected = np.zeros(3)
    errors = []
    for _ in range(40):
        error = 1. + .4 * expected[1] - expected[0]
        expected[0] += .3 * error
        errors.append(abs(error))
        error = 3. + (.4 * expected[0] if truncated else 0.) - expected[1]
        expected[1] += .3 * error
        errors.append(abs(error))
    np.testing.assert_allclose(fitted.q_table[:, 0], expected, atol=1e-12)
    np.testing.assert_array_equal(fitted.state_visits, [40, 40, 0])
    assert metrics['n_episodes'] == 40 and metrics['total_steps'] == 80
    assert metrics['mean_episode_steps'] == 2 and metrics['mean_return'] == 4
    assert metrics['mean_abs_td_error'] == pytest.approx(np.mean(errors))
    assert metrics['state_coverage'] == pytest.approx(2/3)
    assert instances[-1].closed and instances[-1].interactions == 80
    assert any('did not improve' in warning for warning in warnings)
    before = fitted.q_table.copy()
    before_visits = fitted.state_visits.copy()
    evaluated = tabular.evaluate_tabular_policy(fitted, n_episodes=np.int64(3), max_steps=np.int64(5))
    assert evaluated == pytest.approx({'n_eval_episodes': 3, 'mean_return': 4,
        'std_return': 0, 'min_return': 4, 'max_return': 4, 'mean_episode_steps': 2,
        'unseen_state_rate': 0})
    assert instances[-1].closed and instances[-1].interactions == 6
    np.testing.assert_array_equal(fitted.q_table, before)
    np.testing.assert_array_equal(fitted.state_visits, before_visits)
