"""Tabular TD control tests (Q-learning / SARSA family) for buildml.rl.tabular."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.explain.concepts import CONCEPT_NOTES
from buildml.rl.catalog import (
    TABULAR_ALGORITHMS,
    list_rl_algorithms,
    resolve_rl_backend_mode_algorithm,
    rl_capability_matrix,
)
from buildml.rl.extras import gymnasium_available
from buildml.rl.tabular import (
    MAX_TABULAR_STATES,
    ObservationDiscretizer,
    TabularValuePolicy,
    act_tabular_observation,
    epsilon_greedy_probabilities,
    evaluate_tabular_policy,
    resolve_tabular_algorithm,
    train_tabular_control,
)

requires_gym = pytest.mark.skipif(
    not gymnasium_available(), reason="buildml[rl] (gymnasium) not installed"
)


def _tiny_session() -> Session:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})
    return (
        Session.ingest(frame)
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.5, random_state=0)
    )


# --------------------------------------------------------------------------
# Pure-numpy units (no gymnasium required)
# --------------------------------------------------------------------------


def test_resolve_tabular_algorithm_normalises_and_refuses() -> None:
    assert resolve_tabular_algorithm(None) == "q_learning"
    assert resolve_tabular_algorithm("Double-Q-Learning") == "double_q_learning"
    with pytest.raises(ValidationError, match="tabular TD-control"):
        resolve_tabular_algorithm("ppo")


def test_epsilon_greedy_probabilities_sum_to_one_and_split_ties() -> None:
    probs = epsilon_greedy_probabilities(np.array([1.0, 0.0, 0.0]), epsilon=0.3)
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] == pytest.approx(0.7 + 0.1)
    assert probs[1] == pytest.approx(0.1)

    # An all-zero row means every action is greedy; mass must be shared.
    tied = epsilon_greedy_probabilities(np.zeros(4), epsilon=0.2)
    assert np.allclose(tied, 0.25)


def test_discrete_discretizer_is_identity_and_bounds_checked() -> None:
    disc = ObservationDiscretizer(kind="discrete", n_states=16, obs_dim=1)
    assert disc.index(0) == 0
    assert disc.index(np.array([15])) == 15
    with pytest.raises(ValidationError, match="outside"):
        disc.index(16)


def test_box_discretizer_bins_and_clips() -> None:
    disc = ObservationDiscretizer(
        kind="box",
        n_states=9,
        obs_dim=2,
        n_bins=3,
        bin_edges=((-1.0, 1.0), (-1.0, 1.0)),
        low=(-2.0, -2.0),
        high=(2.0, 2.0),
        bound_sources=("space_bounds", "space_bounds"),
    )
    assert disc.index([-1.5, -1.5]) == 0
    assert disc.index([1.5, 1.5]) == 8
    # Out-of-range and non-finite values clamp instead of silently mis-binning.
    assert disc.index([99.0, 99.0]) == 8
    assert disc.index([np.nan, np.nan]) == disc.index([0.0, 0.0])
    with pytest.raises(ValidationError, match="Observation dim"):
        disc.index([0.0, 0.0, 0.0])


def test_policy_tie_breaking_and_double_table_averaging() -> None:
    disc = ObservationDiscretizer(kind="discrete", n_states=3, obs_dim=1)
    policy = TabularValuePolicy(
        n_actions=4,
        n_states=3,
        algorithm="q_learning",
        discretizer=disc,
    )
    # A freshly initialised table is all-zero; tie-breaking must not lock on 0.
    rng = np.random.default_rng(0)
    chosen = {policy.greedy_action_for_state(0, rng=rng) for _ in range(50)}
    assert len(chosen) > 1

    policy.q_table[1] = np.array([0.0, 5.0, 0.0, 0.0])
    assert policy.greedy_action_for_state(1, rng=rng) == 1

    double = TabularValuePolicy(
        n_actions=2,
        n_states=2,
        algorithm="double_q_learning",
        discretizer=ObservationDiscretizer(kind="discrete", n_states=2, obs_dim=1),
    )
    assert double.q_table_b is not None
    double.q_table[0] = np.array([2.0, 0.0])
    double.q_table_b[0] = np.array([0.0, 4.0])
    assert np.allclose(double.q_values_for_state(0), [1.0, 2.0])
    assert double.greedy_action_for_state(0) == 1


def test_policy_rejects_mismatched_table_shape() -> None:
    disc = ObservationDiscretizer(kind="discrete", n_states=3, obs_dim=1)
    with pytest.raises(ValidationError, match="q_table shape"):
        TabularValuePolicy(
            n_actions=2,
            n_states=3,
            algorithm="q_learning",
            discretizer=disc,
            q_table=np.zeros((5, 2)),
        )


def test_act_tabular_observation_returns_q_values() -> None:
    disc = ObservationDiscretizer(kind="discrete", n_states=2, obs_dim=1)
    policy = TabularValuePolicy(
        n_actions=3,
        n_states=2,
        algorithm="q_learning",
        discretizer=disc,
    )
    policy.q_table[1] = np.array([0.1, 0.9, 0.2])
    action, q_values = act_tabular_observation(policy, 1, deterministic=True)
    assert action == 1
    assert q_values == pytest.approx((0.1, 0.9, 0.2))


def test_greedy_and_value_tables_are_inspectable() -> None:
    disc = ObservationDiscretizer(kind="discrete", n_states=2, obs_dim=1)
    policy = TabularValuePolicy(
        n_actions=2,
        n_states=2,
        algorithm="q_learning",
        discretizer=disc,
    )
    policy.q_table[0] = np.array([3.0, 1.0])
    policy.q_table[1] = np.array([1.0, 7.0])
    assert policy.greedy_policy_table().tolist() == [0, 1]
    assert policy.state_value_table().tolist() == [3.0, 7.0]


# --------------------------------------------------------------------------
# Catalog / teaching-surface wiring
# --------------------------------------------------------------------------


def test_catalog_exposes_tabular_algorithms() -> None:
    matrix = rl_capability_matrix()
    native = matrix["rl_backends"]["native"]
    assert "tabular_q" in native["modes"]
    for algo in TABULAR_ALGORITHMS:
        assert algo in native["algorithms"]
    assert native["algorithms_by_mode"]["tabular_q"] == list(TABULAR_ALGORITHMS)
    if gymnasium_available():
        assert list_rl_algorithms(backend="native", mode="tabular_q") == list(
            TABULAR_ALGORITHMS
        )
        assert list_rl_algorithms(backend="native", mode="gym_reinforce") == [
            "reinforce_linear_softmax"
        ]


def test_concept_notes_and_catalog_links_cover_q_learning() -> None:
    for key in (
        "rl-tabular-q-learning",
        "rl-sarsa-on-policy",
        "rl-state-discretization",
    ):
        assert key in CONCEPT_NOTES
        note = CONCEPT_NOTES[key]
        assert note.summary
        for related in note.related_concepts:
            assert related in CONCEPT_NOTES
    assert "rl-tabular-q-learning" in OPERATION_CATALOG["fit_rl"].concept_links


@pytest.mark.skipif(not gymnasium_available(), reason="gymnasium not installed")
def test_resolver_defaults_for_tabular_paths() -> None:
    # mode only → q_learning default (the shared 'linucb' default is ignored).
    backend, mode, algo = resolve_rl_backend_mode_algorithm(
        backend=None, mode="tabular_q", algorithm="linucb"
    )
    assert (backend, mode, algo) == ("native", "tabular_q", "q_learning")

    # algorithm only → routed to the native tabular backend/mode.
    backend, mode, algo = resolve_rl_backend_mode_algorithm(
        backend=None, mode=None, algorithm="sarsa"
    )
    assert (backend, mode, algo) == ("native", "tabular_q", "sarsa")

    # REINFORCE still resolves unchanged.
    backend, mode, algo = resolve_rl_backend_mode_algorithm(
        backend="native", mode="gym_reinforce", algorithm="linucb"
    )
    assert algo == "reinforce_linear_softmax"

    with pytest.raises(ValidationError, match="invalid for tabular_q"):
        resolve_rl_backend_mode_algorithm(
            backend="native", mode="tabular_q", algorithm="ppo"
        )
    with pytest.raises(ValidationError, match="tabular TD control"):
        resolve_rl_backend_mode_algorithm(
            backend="native", mode="gym_reinforce", algorithm="q_learning"
        )
    with pytest.raises(ValidationError, match="contextual_bandit"):
        resolve_rl_backend_mode_algorithm(
            backend="sklearn", mode="tabular_q", algorithm="q_learning"
        )


def test_tabular_q_requires_the_rl_extra_when_absent() -> None:
    if gymnasium_available():
        pytest.skip("gymnasium installed")
    with pytest.raises(MissingExtraError, match="rl"):
        _tiny_session().fit_rl(mode="tabular_q", n_episodes=5)


# --------------------------------------------------------------------------
# Learning behaviour (requires gymnasium)
# --------------------------------------------------------------------------


@requires_gym
def test_hyperparameter_validation_refuses_bad_schedules() -> None:
    with pytest.raises(ValidationError, match="learning_rate"):
        train_tabular_control(env_id="FrozenLake-v1", learning_rate=0.0, n_episodes=1)
    with pytest.raises(ValidationError, match="epsilon_min <= epsilon"):
        train_tabular_control(
            env_id="FrozenLake-v1", epsilon=0.1, epsilon_min=0.5, n_episodes=1
        )
    with pytest.raises(ValidationError, match="gamma"):
        train_tabular_control(env_id="FrozenLake-v1", gamma=1.5, n_episodes=1)


@requires_gym
def test_q_learning_solves_deterministic_frozenlake() -> None:
    policy, metrics, disclosures, _warnings = train_tabular_control(
        env_id="FrozenLake-v1",
        algorithm="q_learning",
        n_episodes=1_500,
        max_steps=100,
        learning_rate=0.2,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        random_state=0,
    )
    assert policy.n_states == 16
    assert policy.n_actions == 4
    assert metrics["mean_return_last_20"] > metrics["mean_return_first_20"]
    assert any("Q[s, a]" in note for note in disclosures)

    scores = evaluate_tabular_policy(
        policy, n_episodes=100, max_steps=100, random_state=0
    )
    # Slippery FrozenLake caps out near 0.74; a random policy scores ~0.01.
    assert scores["mean_return"] > 0.3
    assert scores["unseen_state_rate"] == pytest.approx(0.0)


@requires_gym
@pytest.mark.parametrize("algorithm", list(TABULAR_ALGORITHMS))
def test_every_tabular_algorithm_learns_the_cliff(algorithm: str) -> None:
    policy, metrics, _disclosures, _warnings = train_tabular_control(
        env_id="CliffWalking-v0",
        algorithm=algorithm,
        n_episodes=400,
        max_steps=200,
        learning_rate=0.5,
        gamma=1.0,
        epsilon=0.3,
        epsilon_min=0.1,
        epsilon_decay=1.0,
        random_state=0,
    )
    assert policy.algorithm == algorithm
    assert (policy.q_table_b is not None) == (algorithm == "double_q_learning")
    # Every method must beat the -200 timeout floor of an untrained walker.
    assert metrics["mean_return_last_20"] > -200.0
    assert metrics["state_coverage"] > 0.0
    assert np.isfinite(metrics["mean_abs_td_error"])


@requires_gym
def test_q_learning_recovers_the_optimal_cliff_path() -> None:
    """Off-policy control learns the -13 cliff-edge path its behaviour avoids."""
    policy, _metrics, _disclosures, _warnings = train_tabular_control(
        env_id="CliffWalking-v0",
        algorithm="q_learning",
        n_episodes=500,
        max_steps=200,
        learning_rate=0.5,
        gamma=1.0,
        epsilon=0.3,
        epsilon_min=0.1,
        epsilon_decay=1.0,
        random_state=0,
    )
    greedy = evaluate_tabular_policy(
        policy, n_episodes=3, max_steps=200, random_state=0
    )
    assert greedy["mean_return"] == pytest.approx(-13.0)


@requires_gym
def test_box_observations_are_discretized_with_disclosed_bounds() -> None:
    policy, metrics, disclosures, _warnings = train_tabular_control(
        env_id="CartPole-v1",
        algorithm="q_learning",
        n_episodes=60,
        max_steps=100,
        learning_rate=0.15,
        n_bins=5,
        random_state=0,
    )
    disc = policy.discretizer
    assert disc.kind == "box"
    assert disc.obs_dim == 4
    assert disc.n_states == 5**4 == policy.n_states
    # CartPole declares two finite and two infinite dims.
    assert set(disc.bound_sources) == {"space_bounds", "random_policy_probe"}
    assert all(np.isfinite(disc.low)) and all(np.isfinite(disc.high))
    assert any("uniform bins" in note for note in disclosures)
    assert metrics["n_states"] == float(policy.n_states)


@requires_gym
def test_state_space_guard_refuses_oversized_tables() -> None:
    with pytest.raises(ValidationError, match=str(MAX_TABULAR_STATES)):
        train_tabular_control(
            env_id="CartPole-v1",
            n_bins=64,
            n_episodes=1,
            max_steps=1,
        )


@requires_gym
def test_continuous_action_envs_are_refused() -> None:
    with pytest.raises(ValidationError, match="discrete action space"):
        train_tabular_control(env_id="Pendulum-v1", n_episodes=1, max_steps=1)


# --------------------------------------------------------------------------
# Session surface end-to-end
# --------------------------------------------------------------------------


@requires_gym
def test_session_tabular_q_fit_act_evaluate_bundle(tmp_path: Path) -> None:
    session = _tiny_session()
    fit = session.fit_rl(
        mode="tabular_q",
        algorithm="q_learning",
        env_id="FrozenLake-v1",
        n_episodes=600,
        max_steps=100,
        learning_rate=0.2,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        random_state=0,
    )
    assert fit.mode == "tabular_q"
    assert fit.backend == "native"
    assert fit.algorithm == "q_learning"
    assert fit.n_arms == 4

    plan = session.rl_plan
    assert plan is not None
    assert plan.config["discretizer"]["kind"] == "discrete"
    assert plan.config["n_bins"] == 8
    assert plan.config["epsilon_decay"] == 0.99

    ev = session.evaluate_rl(n_episodes=10, max_steps=100)
    assert ev.offline is False
    assert "mean_return" in ev.metrics
    assert "unseen_state_rate" in ev.metrics

    act = session.act_rl(observations=[0, 1, 2])
    assert act.n_rows == 3
    assert all(0 <= int(a) < 4 for a in act.actions)
    assert len(act.scores[0]) == 4

    out = tmp_path / "tabular_bundle"
    session.save_rl_bundle(out)
    assert (out / "meta.json").is_file()

    other = _tiny_session()
    other.load_rl_bundle(out, trusted=True)
    assert other.rl_plan is not None
    assert other.rl_plan.algorithm == "q_learning"
    reloaded = other.evaluate_rl(n_episodes=5, max_steps=100)
    assert reloaded.offline is False


@requires_gym
def test_session_algorithm_alone_routes_to_tabular() -> None:
    session = _tiny_session()
    fit = session.fit_rl(
        algorithm="expected_sarsa",
        env_id="FrozenLake-v1",
        n_episodes=50,
        max_steps=50,
        random_state=0,
    )
    assert fit.mode == "tabular_q"
    assert fit.algorithm == "expected_sarsa"


@requires_gym
def test_walkthrough_reports_tabular_mode() -> None:
    session = _tiny_session()
    session.fit_rl(
        mode="tabular_q",
        env_id="FrozenLake-v1",
        n_episodes=30,
        max_steps=50,
        random_state=0,
    )
    payload = session.walkthrough().to_dict()
    status = payload["rl_status"]
    assert status["mode"] == "tabular_q"
    assert status["algorithm"] == "q_learning"
    assert any("tabular_q requires buildml[rl]" in note for note in status["disclosures"])
