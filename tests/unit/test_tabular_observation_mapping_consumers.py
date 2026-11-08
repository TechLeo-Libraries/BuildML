"""Physical discrete observations through real RL training and saved consumers."""

import json
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.rl import tabular

ALGORITHMS = ("q_learning", "sarsa", "expected_sarsa", "double_q_learning")
STARTS = (0, np.int64(5), -4, 2**53 + 5)


def _session():
    return (
        Session.ingest(pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]}))
        .set_roles({"x": "feature", "y": "target"})
        .split(test_size=0.5, random_state=0)
    )


@pytest.fixture(scope="module")
def registered_environments():
    gym = pytest.importorskip("gymnasium")
    registrations = []

    def register(kind, start):
        instances = []

        class KnownProcess(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Discrete(
                    3 if kind == "chain" else 2, start=int(start)
                )
                self.action_space = gym.spaces.Discrete(1 if kind == "chain" else 2)
                self.position = 0
                self.closed = False
                self.rewards = []
                self.actions = []
                instances.append(self)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.position = 0
                return int(start), {}

            def step(self, action):
                assert self.action_space.contains(action)
                self.actions.append(int(action))
                self.position += 1
                reward = (
                    (1.0 if self.position == 1 else 3.0) if kind == "chain" else (1.0, 3.0)[action]
                )
                self.rewards.append(reward)
                terminated = self.position == (2 if kind == "chain" else 1)
                return int(start) + self.position, reward, terminated, False, {}

            def close(self):
                self.closed = True

        name = f"BuildMLObservation{uuid4().hex}-v0"
        gym.register(name, entry_point=KnownProcess)
        registrations.append(name)
        return name, instances

    yield register
    for name in registrations:
        gym.registry.pop(name, None)


def _train(name, algorithm, kind, session=None):
    options = dict(
        env_id=name,
        algorithm=algorithm,
        n_episodes=600 if kind == "chain" else 1200,
        max_steps=3,
        learning_rate=0.3,
        gamma=0.5,
        epsilon=0.4,
        epsilon_min=0.4,
        epsilon_decay=1.0,
        random_state=17,
    )
    if session is None:
        policy, metrics, _, _ = tabular.train_tabular_control(**options)
        return policy, metrics
    report = session.rl.fit(mode="tabular_q", **options)
    return session.rl.plan.policy_, report.train_metrics


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize("kind", ("chain", "bandit"))
@pytest.mark.parametrize("route", ("direct", "session_bundle"))
def test_physical_states_have_known_bellman_values_and_survive_bundle(
    registered_environments, algorithm, start, kind, route, tmp_path
):
    name, instances = registered_environments(kind, start)
    session = _session() if route == "session_bundle" else None
    original_data = None if session is None else session.dataset.to_pandas().copy(deep=True)
    policy, metrics = _train(name, algorithm, kind, session)
    trained = instances[-1]
    assert trained.closed
    # These are the analytic fixed points of two deterministic processes:
    # chain V(1)=3, V(0)=1+.5*3; bandit Q(0,a) is its immediate reward.
    expected = (
        np.array([[2.5], [3.0], [0.0]]) if kind == "chain" else np.array([[1.0, 3.0], [0.0, 0.0]])
    )
    np.testing.assert_allclose(policy.q_table, expected, atol=1e-7, rtol=0)
    if algorithm == "double_q_learning":
        np.testing.assert_allclose(policy.q_table_b, expected, atol=1e-7, rtol=0)
    else:
        assert policy.q_table_b is None
    n_episodes, n_steps = (600, 2) if kind == "chain" else (1200, 1)
    assert metrics["total_steps"] == len(trained.rewards) == n_episodes * n_steps
    assert metrics["mean_return"] == pytest.approx(sum(trained.rewards) / n_episodes)
    assert metrics["mean_episode_steps"] == n_steps
    assert metrics["final_epsilon"] == 0.4
    assert metrics["n_states"] == len(expected)
    assert policy.discretizer.to_dict()["start"] == int(start)
    visits = [n_episodes, n_episodes, 0] if kind == "chain" else [n_episodes, 0]
    np.testing.assert_array_equal(policy.state_visits, visits)
    assert policy.visited_state_fraction() == pytest.approx(n_steps / len(expected))
    np.testing.assert_array_equal(policy.greedy_policy_table(), np.argmax(expected, axis=1))
    np.testing.assert_allclose(policy.state_value_table(), expected.max(axis=1), atol=1e-7, rtol=0)
    before = joblib.hash(policy)
    for state, values in enumerate(expected):
        observation = int(start) + state
        assert policy.state_index(observation) == state
        np.testing.assert_allclose(policy.q_values(observation), values, atol=1e-7, rtol=0)
        masses = [1.0] if kind == "chain" else ([0.1, 0.9] if state == 0 else [0.5, 0.5])
        np.testing.assert_allclose(
            policy.action_probabilities_for_state(state, epsilon=0.2), masses
        )
        action = policy.act(observation, rng=np.random.default_rng(7), deterministic=True)
        assert values[action] == max(values)
        sampled = policy.act(
            observation, rng=np.random.default_rng(7), deterministic=False, epsilon=0.2
        )
        assert 0 <= sampled < len(values)
        returned, scores = tabular.act_tabular_observation(
            policy, observation, deterministic=False, epsilon=0.2, random_state=7
        )
        assert returned == sampled
        np.testing.assert_allclose(scores, values, atol=1e-7, rtol=0)
    if kind == "bandit":
        # A real exploring consumer, checked against probability mass rather
        # than reimplementing the learner's random draws or TD updates.
        rng = np.random.default_rng(123)
        actions = [
            policy.act(int(start), rng=rng, deterministic=False, epsilon=0.2) for _ in range(1000)
        ]
        assert abs(actions.count(0) / len(actions) - 0.1) < 0.04
        assert set(trained.actions) == {0, 1}
    for invalid in (int(start) - 1, int(start) + len(expected)):
        with pytest.raises(ValidationError, match="outside"):
            policy.q_values(invalid)
    evaluation = tabular.evaluate_tabular_policy(policy, n_episodes=3, max_steps=3)
    reward = 4.0 if kind == "chain" else 3.0
    assert evaluation == pytest.approx(
        dict(
            n_eval_episodes=3,
            mean_return=reward,
            std_return=0,
            min_return=reward,
            max_return=reward,
            mean_episode_steps=n_steps,
            unseen_state_rate=0,
        )
    )
    assert instances[-1].closed and len(instances[-1].rewards) == 3 * n_steps
    assert joblib.hash(policy) == before
    if session is not None:
        evaluated = session.rl.evaluate(n_episodes=3, max_steps=3)
        assert not evaluated.offline and evaluated.metrics == pytest.approx(evaluation)
        observations = [int(start) + state for state in range(len(expected))]
        acted = session.rl.act(observations=observations)
        np.testing.assert_allclose(acted.scores, expected, atol=1e-7, rtol=0)
        assert acted.actions[0] == int(np.argmax(expected[0]))
        assert acted.n_rows == len(expected) and acted.partition is None
        destination = session.rl.save_bundle(tmp_path / "policy")
        metadata = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
        assert metadata["format"] == "buildml.rl_bundle.v1"
        assert metadata["plan"]["config"]["discretizer"]["start"] == int(start)
        restored = _session()
        restored.rl.load_bundle(destination, trusted=True)
        assert joblib.hash(restored.rl.plan.policy_) == before
        assert restored.rl.act(observations=observations).scores == acted.scores
        assert restored.rl.evaluate(n_episodes=3, max_steps=3).metrics == pytest.approx(evaluation)
        assert joblib.hash(policy) == before
        pd.testing.assert_frame_equal(session.dataset.to_pandas(), original_data)
    assert all(instance.closed for instance in instances)


@pytest.fixture(scope="module", params=ALGORITHMS)
def trained_session(request, registered_environments):
    name, _ = registered_environments("bandit", 0)
    session = _session()
    _train(name, request.param, "bandit", session)
    return session


BAD_SCALARS = (
    1.75,
    "1",
    b"1",
    np.float32(0.5),
    np.nan,
    np.inf,
    np.complex128(1 + 0j),
    np.complex64(1 + 1j),
    None,
)


@pytest.mark.parametrize("observation", BAD_SCALARS)
@pytest.mark.parametrize(
    "route", ("q_values", "state_index", "policy_act", "act_observation", "session_act")
)
def test_invalid_scalar_cannot_select_another_state_or_mutate_policy(
    trained_session, observation, route
):
    policy = trained_session.rl.plan.policy_
    before = joblib.hash(policy)
    history = joblib.hash(trained_session.history)
    with pytest.raises(ValidationError):
        if route == "policy_act":
            policy.act(observation, rng=np.random.default_rng(1))
        elif route == "act_observation":
            tabular.act_tabular_observation(policy, observation)
        elif route == "session_act":
            trained_session.rl.act(observations=[observation])
        else:
            getattr(policy, route)(observation)
    assert joblib.hash(policy) == before
    assert joblib.hash(trained_session.history) == history


@pytest.mark.parametrize("observation", ([0, 1], []))
@pytest.mark.parametrize("route", ("q_values", "state_index", "policy_act", "act_observation"))
def test_single_observation_shape_refusals_preserve_trained_policy(
    trained_session, observation, route
):
    policy = trained_session.rl.plan.policy_
    before = joblib.hash(policy)
    with pytest.raises(ValidationError, match="scalar"):
        if route == "policy_act":
            policy.act(observation, rng=np.random.default_rng(1))
        elif route == "act_observation":
            tabular.act_tabular_observation(policy, observation)
        else:
            getattr(policy, route)(observation)
    assert joblib.hash(policy) == before


@pytest.mark.parametrize(
    "observation", (0, np.int64(1), 1.0, np.float32(1.0), True, [1], np.array([[1]]))
)
def test_exact_integer_valued_observations_remain_usable(trained_session, observation):
    policy = trained_session.rl.plan.policy_
    state = int(np.asarray(observation).reshape(-1)[0])
    assert policy.state_index(observation) == state
    np.testing.assert_array_equal(policy.q_values(observation), policy.q_values_for_state(state))
