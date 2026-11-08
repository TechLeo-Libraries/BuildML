"""Public RL routes honor the algorithm requested by the caller."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.rl.catalog import list_rl_algorithms
from buildml.rl.fit import fit_rl


def _env_session():
    return Session.ingest(pd.DataFrame({"unused": [0.0, 1.0]}))


@pytest.fixture(scope="module")
def gym():
    return pytest.importorskip("gymnasium")


@pytest.fixture(scope="module")
def sb3(gym):
    return pytest.importorskip("stable_baselines3")


@pytest.mark.parametrize("algorithm_options", ({}, {"algorithm": None}), ids=("omitted", "none"))
@pytest.mark.parametrize("target", ("action", "payoff"))
def test_default_bandit_uses_training_only_linucb(algorithm_options, target):
    i = np.arange(40, dtype=float)
    frame = pd.DataFrame(
        {"signal": np.sin(i), "other": np.cos(i / 2), "action": (i % 2).astype(int)}
    )
    reward_column = "reward" if target == "action" else "payoff"
    frame[reward_column] = 1 + 0.3 * i - 2 * frame["action"]
    session = (
        Session.ingest(frame)
        .set_roles({"signal": "feature", "other": "feature", target: "target"})
        .split(test_size=0.25, random_state=13)
    )
    columns = ["signal", "other"]
    alpha = 0.4
    fit = session.rl.fit(columns=columns, action_column="action", alpha=alpha, **algorithm_options)
    assert (fit.backend, fit.mode, fit.algorithm) == ("sklearn", "contextual_bandit", "linucb")
    assert fit.reward_column == reward_column

    train = frame.iloc[list(session.split_plan.train_indices)]
    holdout = frame.iloc[list(session.split_plan.test_indices)][columns].to_numpy()
    expected = []
    for arm in (0, 1):
        logged = train.loc[train["action"] == arm]
        x = logged[columns].to_numpy()
        precision = np.eye(len(columns)) + x.T @ x
        weights = np.linalg.solve(precision, x.T @ logged[reward_column].to_numpy())
        variance = np.sum(holdout * np.linalg.solve(precision, holdout.T).T, axis=1)
        expected.append(holdout @ weights + alpha * np.sqrt(variance))
    scores = np.column_stack(expected)
    actual = session.rl.act(partition="test", deterministic=True)
    np.testing.assert_allclose(actual.scores, scores, atol=1e-12)
    assert actual.actions == tuple(scores.argmax(axis=1))


@pytest.mark.parametrize("algorithm", ("misspelled_policy", "ppo", "epsilon-greedy", ""))
def test_reinforce_rejects_an_explicit_incompatible_algorithm_without_refitting(gym, algorithm):
    session = _env_session()
    session.rl.fit(
        mode="gym_reinforce", algorithm="reinforce_linear_softmax", n_episodes=2, max_steps=6
    )
    plan = session.rl_plan
    history = list(session.history)
    original_weights = plan.policy_.weights.copy()
    with pytest.raises(ValidationError, match="invalid for gym_reinforce"):
        session.rl.fit(mode="gym_reinforce", algorithm=algorithm, n_episodes=2, max_steps=6)
    assert session.rl_plan is plan and session.history == history
    np.testing.assert_array_equal(plan.policy_.weights, original_weights)


@pytest.mark.parametrize(
    "options", ({}, {"algorithm": None}, {"algorithm": "linucb"}), ids=("omitted", "none", "legacy")
)
def test_native_reinforce_defaults_score_with_the_fitted_softmax(gym, options):
    session = _env_session()
    result = session.rl.fit(
        mode="gym_reinforce", n_episodes=3, max_steps=7, random_state=19, **options
    )
    assert (result.backend, result.mode, result.algorithm) == (
        "native",
        "gym_reinforce",
        "reinforce_linear_softmax",
    )
    observations = np.array([[0.02, -0.05, 0.1, 0.04], [0.13, -0.16, -0.15, 0.01]])
    logits = observations @ session.rl_plan.policy_.weights.T
    numerators = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = numerators / numerators.sum(axis=1, keepdims=True)
    with pytest.raises(ValidationError, match="requires observations"):
        session.rl.act()
    actual = session.rl.act(observations=observations, deterministic=True)
    np.testing.assert_allclose(actual.scores, probabilities, atol=1e-12)
    assert actual.actions == tuple(probabilities.argmax(axis=1))


def test_explicit_none_reaches_tabular_default_through_the_low_level_fit(gym):
    plan, result = fit_rl(
        None,
        None,
        mode="tabular_q",
        algorithm=None,
        env_id="FrozenLake-v1",
        n_episodes=3,
        max_steps=6,
    )
    assert (result.backend, result.mode, result.algorithm) == ("native", "tabular_q", "q_learning")
    assert (plan.algorithm, plan.n_train_rows, plan.n_arms) == ("q_learning", 3, 4)


@pytest.mark.parametrize(
    "options",
    ({"mode": "gym_sb3"}, {"backend": "industry", "algorithm": None}, {"algorithm": "PPO"}),
    ids=("mode-default", "backend-default", "explicit-inference"),
)
def test_sb3_routes_fit_and_score_real_ppo(sb3, options):
    session = _env_session()
    result = session.rl.fit(
        env_id="CartPole-v1", total_timesteps=32, max_steps=8, random_state=5, **options
    )
    assert (result.backend, result.mode, result.algorithm) == ("industry", "gym_sb3", "ppo")
    assert result.train_metrics["requested_timesteps"] == 32
    plan = session.rl_plan
    assert isinstance(plan.policy_.model, sb3.PPO)
    observations = np.array([[0.0, 0.0, 0.0, 0.0], [0.01, -0.03, 0.04, -0.01]], dtype=np.float32)
    predicted, _ = plan.policy_.model.predict(observations, deterministic=True)
    with pytest.raises(ValidationError, match="requires observations"):
        session.rl.act()
    with pytest.raises(ValidationError, match="Observation dim 2"):
        session.rl.act(observations=observations[0, :2])
    actual = session.rl.act(observations=observations, deterministic=True)
    assert actual.actions == tuple(predicted)
    np.testing.assert_array_equal(actual.scores, np.eye(plan.n_arms)[predicted])
    single = session.rl.act(observations=observations[0], deterministic=True)
    assert single.actions == actual.actions[:1]
    assert single.scores == actual.scores[:1]


def test_sb3_rejects_explicit_linucb_and_conflicting_mode(sb3):
    session = _env_session()
    with pytest.raises(ValidationError, match="linucb.*invalid for gym_sb3"):
        session.rl.fit(mode="gym_sb3", algorithm="linucb")
    with pytest.raises(ValidationError, match="industry.*supports mode"):
        session.rl.fit(backend="industry", mode="tabular_q", algorithm="q_learning")


def test_catalog_lists_usable_modes_and_their_distinct_algorithms(sb3):
    assert list_rl_algorithms() == [
        "linucb",
        "epsilon_greedy",
        "softmax",
        "reinforce_linear_softmax",
        "q_learning",
        "sarsa",
        "expected_sarsa",
        "double_q_learning",
        "ppo",
        "dqn",
        "a2c",
    ]
    assert list_rl_algorithms(backend="native", mode="gym_reinforce") == [
        "reinforce_linear_softmax"
    ]
    assert list_rl_algorithms(backend="native", mode="gym_sb3") == []
    assert list_rl_algorithms(backend="industry", mode="gym_sb3") == ["ppo", "dqn", "a2c"]


def test_unknown_algorithm_without_mode_is_rejected_before_inference():
    with pytest.raises(ValidationError, match="Unknown RL algorithm.*misspelled_policy"):
        _env_session().rl.fit(algorithm="misspelled_policy")
