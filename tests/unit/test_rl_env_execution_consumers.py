"""Real environment rollouts honor episode budgets and model observation shapes."""

from inspect import signature

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

MATRIX_ENV = "BuildMLMatrixCartPoleContract-v0"


def _session():
    return Session.ingest(pd.DataFrame({"unused": [0.0, 1.0]}))


def _reshape_cartpole(observation):
    return observation.reshape(2, 2)


def _matrix_cartpole():
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    box = env.observation_space
    space = gym.spaces.Box(box.low.reshape(2, 2), box.high.reshape(2, 2), dtype=box.dtype)
    wrapper = gym.wrappers.TransformObservation
    if "observation_space" in signature(wrapper).parameters:
        return wrapper(env, _reshape_cartpole, observation_space=space)
    wrapped = wrapper(env, _reshape_cartpole)
    wrapped.observation_space = space
    return wrapped


@pytest.fixture(scope="module")
def gym():
    return pytest.importorskip("gymnasium")


@pytest.fixture(scope="module")
def sb3(gym):
    return pytest.importorskip("stable_baselines3")


@pytest.fixture(scope="module")
def matrix_env(gym, sb3):
    gym.register(id=MATRIX_ENV, entry_point=_matrix_cartpole)
    try:
        yield gym
    finally:
        gym.registry.pop(MATRIX_ENV)


@pytest.fixture
def reinforce(gym):
    session = _session()
    session.rl.fit(mode="gym_reinforce", n_episodes=2, max_steps=4, random_state=17)
    return session


@pytest.mark.parametrize("value", (0, -1, True, np.bool_(True), 1.5, "3"))
@pytest.mark.parametrize("parameter", ("n_episodes", "max_steps"))
@pytest.mark.parametrize("operation", ("fit", "evaluate"))
def test_invalid_native_rollout_budget_preserves_fitted_policy(
    reinforce, value, parameter, operation
):
    plan = reinforce.rl_plan
    history = reinforce.history
    weights = plan.policy_.weights.copy()
    options = {"n_episodes": 3, "max_steps": 4, parameter: value}
    if operation == "fit":
        options["mode"] = "gym_reinforce"
    with pytest.raises(ValidationError, match=rf"{parameter} must be a positive integer"):
        getattr(reinforce.rl, operation)(**options)
    assert reinforce.rl_plan is plan and reinforce.history == history
    np.testing.assert_array_equal(plan.policy_.weights, weights)


@pytest.mark.parametrize("count_type", (int, np.int64), ids=("python", "numpy"))
def test_valid_native_rollouts_match_direct_gym_episode_returns(gym, count_type, tmp_path):
    session = _session()
    seed = 7
    episodes, steps = count_type(4), count_type(8)
    result = session.rl.fit(
        mode="gym_reinforce", n_episodes=episodes, max_steps=steps, random_state=seed
    )
    assert result.n_train_rows == episodes
    weights = session.rl_plan.policy_.weights
    returns = []
    env = gym.make("CartPole-v1")
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + 10_000 + ep)
            total = 0.0
            for _ in range(steps):
                action = int(np.argmax(weights @ obs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
    evaluated = session.rl.evaluate(
        n_episodes=episodes, max_steps=steps, random_state=seed, deterministic=True
    )
    assert evaluated.metrics["n_eval_episodes"] == episodes
    assert evaluated.metrics["mean_return"] == pytest.approx(np.mean(returns))
    assert evaluated.metrics["std_return"] == pytest.approx(np.std(returns))
    bundle = session.rl.save_bundle(tmp_path / "native-policy")
    restored = _session().rl.load_bundle(bundle, trusted=True)
    config = restored.rl_plan.config
    assert type(config["n_episodes"]) is type(config["max_steps"]) is int
    assert (config["n_episodes"], config["max_steps"]) == (episodes, steps)
    replay = restored.rl.evaluate(n_episodes=episodes, random_state=seed, deterministic=True)
    assert replay.metrics == evaluated.metrics


@pytest.mark.parametrize("mode", ("gym_reinforce", "gym_sb3"))
@pytest.mark.parametrize(
    ("env_id", "message"),
    (
        ("BuildMLMissingEnvironment-v0", "Failed to create Gymnasium env_id"),
        ("MountainCarContinuous-v0", "requires a discrete action space"),
        ("FrozenLake-v1", "requires a Box-like observation space"),
    ),
)
def test_incompatible_real_environment_is_rejected_at_the_fit_boundary(
    gym, request, mode, env_id, message
):
    if mode == "gym_sb3":
        request.getfixturevalue("sb3")
    session = _session()
    history = session.history
    with pytest.raises(ValidationError, match=message):
        session.rl.fit(mode=mode, env_id=env_id, n_episodes=2, total_timesteps=8, max_steps=4)
    assert session.rl_plan is None and session.history == history


@pytest.mark.parametrize("algorithm", ("a2c", "dqn"))
@pytest.mark.parametrize("replay", (False, True), ids=("fitted", "reloaded"))
def test_matrix_observations_reach_real_sb3_after_fit_and_replay(
    matrix_env, algorithm, replay, tmp_path
):
    seed = 19
    session = _session()
    fit = session.rl.fit(
        algorithm=algorithm, env_id=MATRIX_ENV, total_timesteps=10, max_steps=8, random_state=seed
    )
    assert (fit.mode, fit.algorithm) == ("gym_sb3", algorithm)
    if replay:
        bundle = session.rl.save_bundle(tmp_path / "model")
        session = _session()
        session.rl.load_bundle(bundle, trusted=True)
    plan = session.rl_plan
    assert plan.obs_dim == 4 and plan.policy_.model.observation_space.shape == (2, 2)
    observations = np.array(
        [[[0.02, -0.05], [0.1, 0.04]], [[0.13, -0.16], [-0.15, 0.01]]], dtype=np.float32
    )
    predictions, _ = plan.policy_.model.predict(observations, deterministic=True)
    result = session.rl.act(observations=observations.reshape(-1, 4), deterministic=True)
    assert result.actions == tuple(predictions)
    np.testing.assert_array_equal(result.scores, np.eye(2)[predictions])
    single = session.rl.act(observations=observations[0].reshape(-1), deterministic=True)
    assert single.actions == result.actions[:1]

    env = matrix_env.make(MATRIX_ENV)
    returns = []
    try:
        for ep in range(3):
            obs, _ = env.reset(seed=seed + 10_000 + ep)
            total = 0.0
            for _ in range(7):
                action, _ = plan.policy_.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
    metrics = session.rl.evaluate(
        n_episodes=3, max_steps=7, random_state=seed, deterministic=True
    ).metrics
    assert metrics["n_eval_episodes"] == 3
    assert metrics["mean_return"] == pytest.approx(np.mean(returns))
    assert metrics["std_return"] == pytest.approx(np.std(returns))


@pytest.fixture(scope="module", params=("tabular_q", "gym_sb3"))
def other_rollout_policy(request, gym):
    mode = request.param
    if mode == "gym_sb3":
        request.getfixturevalue("sb3")
    session = _session()
    session.rl.fit(
        mode=mode,
        algorithm="a2c" if mode == "gym_sb3" else "q_learning",
        env_id="CartPole-v1" if mode == "gym_sb3" else "FrozenLake-v1",
        n_episodes=2,
        total_timesteps=10,
        max_steps=4,
    )
    return session


@pytest.mark.parametrize("parameter", ("n_episodes", "max_steps"))
@pytest.mark.parametrize("value", (True, 1.5, "3"))
def test_other_rollout_modes_validate_caller_counts_before_coercion(
    other_rollout_policy, parameter, value
):
    session = other_rollout_policy
    history = session.history
    options = {"n_episodes": 2, "max_steps": 4, parameter: value}
    with pytest.raises(ValidationError, match=rf"{parameter} (?:must be|to be) a positive integer"):
        session.rl.evaluate(**options)
    assert session.history == history
