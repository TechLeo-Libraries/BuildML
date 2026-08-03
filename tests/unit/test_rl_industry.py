"""Industry-depth tests for imitation + RL (SB3 / imitation extras)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.rl.catalog import (
    list_imitation_methods,
    list_rl_algorithms,
    resolve_imitation_backend_method,
    resolve_rl_backend_mode_algorithm,
    rl_capability_matrix,
)
from buildml.rl.extras import gymnasium_available, rl_industry_available


def test_capability_matrix_shape() -> None:
    matrix = rl_capability_matrix()
    assert "imitation_backends" in matrix
    assert "rl_backends" in matrix
    assert matrix["imitation_backends"]["sklearn"]["available"] is True
    assert matrix["rl_backends"]["sklearn"]["available"] is True
    assert "tabular_q" in matrix["rl_backends"]["native"]["modes"]
    assert list_imitation_methods(backend="sklearn")
    assert list_rl_algorithms(backend="sklearn")


def test_session_rl_capability_matrix() -> None:
    matrix = Session.rl_capability_matrix()
    assert matrix["rl_backends"]["native"]["algorithms_by_mode"]["tabular_q"]


def test_resolve_sklearn_defaults() -> None:
    backend, est = resolve_imitation_backend_method(
        backend="sklearn",
        estimator=None,
        method=None,
        task="classification",
    )
    assert backend == "sklearn"
    assert est == "logistic_regression"

    rb, mode, algo = resolve_rl_backend_mode_algorithm(
        backend="sklearn",
        mode="contextual_bandit",
        algorithm="linucb",
    )
    assert rb == "sklearn" and mode == "contextual_bandit" and algo == "linucb"


@pytest.mark.skipif(not rl_industry_available(), reason="buildml[rl-industry] not installed")
def test_sb3_cartpole_smoke() -> None:
    session = (
        Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.5, random_state=0)
    )
    fit = session.fit_rl(
        backend="industry",
        mode="gym_sb3",
        algorithm="ppo",
        env_id="CartPole-v1",
        total_timesteps=5_000,
        max_steps=200,
        random_state=0,
    )
    assert fit.mode == "gym_sb3"
    assert fit.backend == "industry"
    assert session.rl_plan is not None
    ev = session.evaluate_rl(n_episodes=3, max_steps=200)
    assert ev.offline is False
    assert "mean_return" in ev.metrics


@pytest.mark.skipif(not rl_industry_available(), reason="buildml[rl-industry] not installed")
def test_industry_bc_mlp_smoke() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(180, 3))
    action = (x[:, 0] > 0).astype(int)
    frame = pd.DataFrame(
        {"f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2], "action": action}
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "f0": "feature",
                "f1": "feature",
                "f2": "feature",
                "action": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.15, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_imitation(
        backend="industry",
        method="bc_mlp",
        n_epochs=8,
        random_state=0,
    )
    assert fit.backend == "industry"
    assert fit.method == "bc_mlp"
    ev = session.evaluate_imitation(partition="test")
    assert "accuracy" in ev.metrics


def test_industry_missing_extra() -> None:
    if rl_industry_available():
        pytest.skip("rl-industry installed")
    session = (
        Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.5, random_state=0)
    )
    with pytest.raises(MissingExtraError, match="rl-industry"):
        session.fit_rl(backend="industry", mode="gym_sb3", algorithm="ppo")


@pytest.mark.skipif(not gymnasium_available(), reason="gymnasium not installed")
def test_native_reinforce_still_works() -> None:
    session = (
        Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.5, random_state=0)
    )
    fit = session.fit_rl(
        backend="native",
        mode="gym_reinforce",
        env_id="CartPole-v1",
        n_episodes=10,
        max_steps=100,
    )
    assert fit.mode == "gym_reinforce"
    assert fit.backend == "native"
