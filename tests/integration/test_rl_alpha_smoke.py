"""End-to-end Session smoke for imitation + contextual bandit (+ optional gym)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.rl.extras import gymnasium_available


def test_imitation_and_bandit_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    action = (x[:, 0] > 0).astype(int)
    reward = np.where(action == (x[:, 0] > 0).astype(int), 1.0, 0.0)
    frame = pd.DataFrame(
        {"s0": x[:, 0], "s1": x[:, 1], "action": action, "reward": reward}
    )

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "s0": "feature",
                "s1": "feature",
                "action": "target",
                "reward": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard", columns=["s0", "s1"])
    )

    bc = session.fit_imitation()
    assert bc.n_train_rows > 0
    assert session.evaluate_imitation(partition="validation").metrics["accuracy"] >= 0.0
    session.save_imitation_bundle(tmp_path / "imitation")

    rl = session.fit_rl(
        mode="contextual_bandit",
        algorithm="softmax",
        action_column="action",
        reward_column="reward",
        temperature=0.5,
    )
    assert rl.n_arms == 2
    assert session.act_rl(partition="test").n_rows > 0
    ev = session.evaluate_rl(partition="test")
    assert ev.offline is True
    session.save_rl_bundle(tmp_path / "rl")

    other = (
        Session.ingest(frame)
        .set_roles(
            {
                "s0": "feature",
                "s1": "feature",
                "action": "target",
                "reward": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard", columns=["s0", "s1"])
    )
    other.load_rl_bundle(tmp_path / "rl", trusted=True)
    assert other.evaluate_rl(partition="validation").offline is True


@pytest.mark.skipif(not gymnasium_available(), reason="buildml[rl] / gymnasium missing")
def test_gym_reinforce_smoke(tmp_path: Path) -> None:
    # Minimal Session host for the env policy
    frame = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "y": [0, 1, 0, 1]})
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    fit = session.fit_rl(
        mode="gym_reinforce",
        env_id="CartPole-v1",
        n_episodes=20,
        max_steps=100,
        learning_rate=0.02,
        random_state=0,
    )
    assert fit.mode == "gym_reinforce"
    ev = session.evaluate_rl(n_episodes=5, max_steps=100)
    assert "mean_return" in ev.metrics
    session.save_rl_bundle(tmp_path / "gym_rl")
    other = Session.ingest(frame).set_roles({"a": "feature", "y": "target"})
    other.load_rl_bundle(tmp_path / "gym_rl", trusted=True)
    assert other.rl_plan is not None
    assert other.rl_plan.mode == "gym_reinforce"
