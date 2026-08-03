"""Session-facing slice tests for imitation learning + RL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, MissingExtraError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.rl.extras import gymnasium_available


def _demo_session() -> Session:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(220, 2))
    action = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
    frame = pd.DataFrame({"s0": x[:, 0], "s1": x[:, 1], "action": action})
    return (
        Session.ingest(frame)
        .set_roles({"s0": "feature", "s1": "feature", "action": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )


def _bandit_session() -> Session:
    rng = np.random.default_rng(7)
    n = 240
    ctx = rng.normal(size=(n, 2))
    # Optimal arm roughly depends on ctx[:, 0]
    action = (ctx[:, 0] > 0).astype(int)
    reward = np.where(action == (ctx[:, 0] > 0).astype(int), 1.0, 0.0)
    reward = reward + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame(
        {"c0": ctx[:, 0], "c1": ctx[:, 1], "arm": action, "reward": reward}
    )
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "c0": "feature",
                "c1": "feature",
                "arm": "target",
                "reward": "feature",
            }
        )
        .split(test_size=0.25, validation_size=0.2, random_state=0)
        .scale(method="standard", columns=["c0", "c1"])
    )


def test_core_import_and_catalog() -> None:
    import buildml.rl as rl

    assert hasattr(rl, "fit_imitation")
    assert hasattr(rl, "fit_rl")
    assert hasattr(Session, "fit_imitation")
    assert hasattr(Session, "fit_rl")
    for op in (
        "fit_imitation",
        "predict_imitation_action",
        "evaluate_imitation",
        "save_imitation_bundle",
        "load_imitation_bundle",
        "fit_rl",
        "act_rl",
        "evaluate_rl",
        "save_rl_bundle",
        "load_rl_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "imitation-behavioral-cloning" in OPERATION_CATALOG["fit_imitation"].concept_links
    assert "rl-contextual-bandit" in OPERATION_CATALOG["fit_rl"].concept_links
    assert "rl-offline-metrics" in OPERATION_CATALOG["evaluate_rl"].concept_links

    registry = build_default_registry()
    assert registry.get("fit_imitation") is not None
    assert registry.get("fit_rl") is not None
    assert registry.get("evaluate_rl") is not None


def test_imitation_fit_predict_eval_bundle(tmp_path: Path) -> None:
    session = _demo_session()
    fit = session.fit_imitation(task="classification")
    assert session.imitation_plan is not None
    assert fit.n_train_rows >= 1
    assert fit.train_score is not None

    pred = session.predict_imitation_action(partition="test")
    assert len(pred.actions) == pred.n_rows

    ev = session.evaluate_imitation(partition="validation")
    assert "accuracy" in ev.metrics
    assert "macro_f1" in ev.metrics

    out = tmp_path / "imitation_bundle"
    session.save_imitation_bundle(out)
    assert (out / "meta.json").is_file()
    assert (out / "imitation_plan.joblib").is_file()

    other = _demo_session()
    other.load_imitation_bundle(out, trusted=True)
    assert other.imitation_plan is not None
    reloaded = other.evaluate_imitation(partition="test")
    assert "accuracy" in reloaded.metrics


def test_bandit_fit_act_eval_bundle(tmp_path: Path) -> None:
    session = _bandit_session()
    fit = session.fit_rl(
        mode="contextual_bandit",
        algorithm="linucb",
        action_column="arm",
        reward_column="reward",
    )
    assert session.rl_plan is not None
    assert fit.mode == "contextual_bandit"
    assert fit.n_arms == 2

    act = session.act_rl(partition="test", deterministic=True)
    assert len(act.actions) == act.n_rows
    assert len(act.scores) == act.n_rows

    ev = session.evaluate_rl(partition="validation")
    assert ev.offline is True
    assert "direct_method" in ev.metrics
    assert "ips" in ev.metrics
    assert "action_match_rate" in ev.metrics

    out = tmp_path / "rl_bundle"
    session.save_rl_bundle(out)
    assert (out / "meta.json").is_file()
    assert (out / "rl_plan.joblib").is_file()

    other = _bandit_session()
    other.load_rl_bundle(out, trusted=True)
    assert other.rl_plan is not None
    assert other.rl_plan.algorithm == "linucb"
    reloaded = other.evaluate_rl(partition="test")
    assert reloaded.offline is True


def test_epsilon_greedy_bandit() -> None:
    session = _bandit_session()
    fit = session.fit_rl(
        mode="contextual_bandit",
        algorithm="epsilon_greedy",
        action_column="arm",
        reward_column="reward",
        epsilon=0.05,
    )
    assert fit.algorithm == "epsilon_greedy"
    ev = session.evaluate_rl(partition="test")
    assert "direct_method" in ev.metrics


def test_imitation_leakage_refuses_fit_without_split() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            "b": rng.normal(size=40),
            "y": rng.integers(0, 2, size=40),
        }
    )
    session = Session.ingest(frame).set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    with pytest.raises((ValidationError, LeakageError)):
        session.fit_imitation()


def test_bandit_leakage_refuses_fit_without_split() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            "arm": rng.integers(0, 2, size=40),
            "reward": rng.random(size=40),
        }
    )
    session = Session.ingest(frame).set_roles(
        {"a": "feature", "arm": "target", "reward": "feature"}
    )
    with pytest.raises((ValidationError, LeakageError)):
        session.fit_rl(action_column="arm", reward_column="reward")


def test_walkthrough_exposes_imitation_and_rl_status() -> None:
    session = _demo_session()
    session.fit_imitation()
    report = session.walkthrough()
    payload = report.to_dict()
    assert "imitation_status" in payload
    assert payload["imitation_status"]["enabled"] is True

    bandit = _bandit_session()
    bandit.fit_rl(action_column="arm", reward_column="reward")
    payload2 = bandit.walkthrough().to_dict()
    assert "rl_status" in payload2
    assert payload2["rl_status"]["enabled"] is True
    assert payload2["rl_status"]["mode"] == "contextual_bandit"


def test_gym_reinforce_optional() -> None:
    session = _demo_session()
    if not gymnasium_available():
        with pytest.raises(MissingExtraError, match="rl"):
            session.fit_rl(mode="gym_reinforce", n_episodes=5, max_steps=50)
        return
    fit = session.fit_rl(
        mode="gym_reinforce",
        env_id="CartPole-v1",
        n_episodes=15,
        max_steps=100,
        learning_rate=0.02,
        random_state=0,
    )
    assert fit.mode == "gym_reinforce"
    assert session.rl_plan is not None
    assert session.rl_plan.obs_dim is not None
    ev = session.evaluate_rl(n_episodes=3, max_steps=100)
    assert ev.offline is False
    assert "mean_return" in ev.metrics
    obs = np.zeros(session.rl_plan.obs_dim, dtype=float)
    act = session.act_rl(observations=obs, deterministic=True)
    assert act.n_rows == 1
