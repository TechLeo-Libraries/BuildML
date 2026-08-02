"""Imitation + RL Session loop (mirrors quickstart-imitation-rl)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.rl.extras import gymnasium_available


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(220, 2))
    action = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
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
    print("imitation", bc.task, bc.train_score)
    print("imitation_eval", session.evaluate_imitation(partition="validation").metrics)

    rl = session.fit_rl(
        mode="contextual_bandit",
        algorithm="linucb",
        action_column="action",
        reward_column="reward",
    )
    print("bandit", rl.n_arms, rl.train_metrics)
    print("act0", session.act_rl(partition="test").actions[:3])
    print("bandit_eval", session.evaluate_rl(partition="validation").metrics)

    out_il = Path("artifacts") / "imitation_demo_bundle"
    out_rl = Path("artifacts") / "rl_bandit_demo_bundle"
    session.save_imitation_bundle(out_il)
    session.save_rl_bundle(out_rl)
    print("saved", out_il, out_rl)

    if gymnasium_available():
        gym_session = (
            Session.ingest(frame)
            .set_roles(
                {
                    "s0": "feature",
                    "s1": "feature",
                    "action": "target",
                    "reward": "feature",
                }
            )
            .split(test_size=0.2, validation_size=0.2, random_state=0)
        )
        gfit = gym_session.fit_rl(
            mode="gym_reinforce",
            env_id="CartPole-v1",
            n_episodes=80,
            max_steps=200,
            learning_rate=0.02,
            random_state=0,
        )
        print("gym", gfit.train_metrics)
        print("gym_eval", gym_session.evaluate_rl(n_episodes=10).metrics)
        gym_session.save_rl_bundle(Path("artifacts") / "rl_gym_demo_bundle")
    else:
        print("gymnasium not installed; skip gym_reinforce (pip install 'buildml[rl]')")


if __name__ == "__main__":
    main()
