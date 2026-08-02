"""Algorithm-depth tests for imitation + RL (outside Session facade)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.rl.bandit import LinUCBPolicy, RewardModelBandit, offline_bandit_metrics
from buildml.rl.features import infer_imitation_task, softmax
from buildml.rl.fit import fit_rl
from buildml.rl.imitation import evaluate_imitation, fit_imitation


def test_linucb_prefers_better_arm() -> None:
    rng = np.random.default_rng(0)
    # Arm 1 is better when x[0] > 0
    policy = LinUCBPolicy(n_arms=2, dim=2, alpha=0.5)
    for _ in range(80):
        x = rng.normal(size=2)
        best = 1 if x[0] > 0 else 0
        # Mostly log optimal arm with reward 1
        policy.update(x, best, 1.0)
        other = 1 - best
        policy.update(x, other, 0.0)
    x_pos = np.array([1.5, 0.0])
    assert policy.select(x_pos) == 1
    x_neg = np.array([-1.5, 0.0])
    assert policy.select(x_neg) == 0


def test_offline_metrics_shapes() -> None:
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    logged = np.array([0, 1, 0])
    rewards = np.array([1.0, 0.0, 1.0])
    policy = np.array([0, 1, 1])
    pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    prop = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
    metrics = offline_bandit_metrics(
        x=x,
        logged_arms=logged,
        logged_rewards=rewards,
        policy_arms=policy,
        predicted_rewards=pred,
        propensity=prop,
    )
    assert metrics["n_rows"] == 3.0
    assert 0.0 <= metrics["action_match_rate"] <= 1.0
    assert np.isfinite(metrics["direct_method"])
    assert np.isfinite(metrics["ips"])


def test_reward_model_bandit_fit() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 3))
    arms = (x[:, 0] > 0).astype(int)
    rewards = np.where(arms == 1, 1.0, 0.2) + rng.normal(scale=0.01, size=100)
    bandit = RewardModelBandit(
        n_arms=2, dim=3, algorithm="epsilon_greedy", epsilon=0.0, random_state=0
    )
    bandit.fit_logged(x, arms, rewards)
    # Deterministic epsilon=0 → argmax predicted reward
    for row in x[:10]:
        a = bandit.select(row, rng=np.random.default_rng(0))
        assert a in (0, 1)


def test_imitation_regression_depth() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(160, 2))
    y = 0.7 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.05, size=160)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    assert infer_imitation_task(frame["y"]) == "regression"
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    plan, fit = fit_imitation(
        session.dataset,
        session._split_plan,
        task="regression",
        estimator="ridge",
        random_state=0,
    )
    assert plan.task == "regression"
    assert fit.train_score is not None
    ev = evaluate_imitation(
        session.dataset, plan, session._split_plan, partition="test"
    )
    assert "rmse" in ev.metrics
    assert "r2" in ev.metrics


def test_bandit_requires_reward_column() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=80),
            "b": rng.normal(size=80),
            "arm": rng.integers(0, 2, size=80),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "arm": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    with pytest.raises(ValidationError, match="reward_column"):
        fit_rl(
            session.dataset,
            session._split_plan,
            action_column="arm",
        )


def test_softmax_stable() -> None:
    probs = softmax(np.array([1000.0, 1000.0, 1000.0]))
    assert np.allclose(probs.sum(), 1.0)
    assert np.all(np.isfinite(probs))
