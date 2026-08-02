"""RL policy return benchmark — SB3 vs BC baseline on CartPole."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.rl.catalog import rl_capability_matrix
from buildml.rl.extras import gymnasium_available, rl_industry_available


def _cartpole_demo_frame(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Synthetic CartPole-like 4D state demos with heuristic expert actions."""
    if not gymnasium_available():
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(n, 4))
        action = (obs[:, 2] > 0).astype(int)
        return pd.DataFrame(
            {
                "s0": obs[:, 0],
                "s1": obs[:, 1],
                "s2": obs[:, 2],
                "s3": obs[:, 3],
                "action": action,
            }
        )
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    try:
        for i in range(n):
            reset_out = env.reset(seed=int(seed) + i)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            flat = np.asarray(obs, dtype=float).reshape(-1)
            # Weak expert: push cart toward pole lean (teaching signal, not optimal).
            action = 1 if flat[2] > 0 else 0
            rows.append(
                {
                    "s0": float(flat[0]),
                    "s1": float(flat[1]),
                    "s2": float(flat[2]),
                    "s3": float(flat[3]),
                    "action": int(action),
                }
            )
    finally:
        env.close()
    return pd.DataFrame(rows)


def _bc_baseline(*, seed: int = 0) -> dict[str, object]:
    session = (
        Session.ingest(_cartpole_demo_frame(seed=seed))
        .set_roles(
            {
                "s0": "feature",
                "s1": "feature",
                "s2": "feature",
                "s3": "feature",
                "action": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.15, random_state=seed)
        .scale(method="standard")
    )
    fit = session.fit_imitation(backend="sklearn", task="classification")
    ev = session.evaluate_imitation(partition="test")
    return {
        "backend": "sklearn",
        "method": "behavioral_cloning",
        "train_score": fit.train_score,
        "eval_accuracy": ev.metrics.get("accuracy"),
        "eval_macro_f1": ev.metrics.get("macro_f1"),
        "n_train_rows": fit.n_train_rows,
    }


def _sb3_ppo(*, total_timesteps: int = 15_000, seed: int = 0) -> dict[str, object]:
    session = (
        Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.5, random_state=seed)
    )
    fit = session.fit_rl(
        backend="industry",
        mode="gym_sb3",
        algorithm="ppo",
        env_id="CartPole-v1",
        total_timesteps=total_timesteps,
        max_steps=500,
        random_state=seed,
    )
    ev = session.evaluate_rl(n_episodes=15, max_steps=500, random_state=seed)
    return {
        "backend": "industry",
        "method": "ppo",
        "mode": fit.mode,
        "train_mean_return": fit.train_metrics.get("mean_return"),
        "eval_mean_return": ev.metrics.get("mean_return"),
        "eval_std_return": ev.metrics.get("std_return"),
        "total_timesteps": total_timesteps,
    }


def _reinforce_native(*, n_episodes: int = 120, seed: int = 0) -> dict[str, object]:
    session = (
        Session.ingest(pd.DataFrame({"a": [0.0, 1.0], "y": [0, 1]}))
        .set_roles({"a": "feature", "y": "target"})
        .split(test_size=0.5, random_state=seed)
    )
    fit = session.fit_rl(
        backend="native",
        mode="gym_reinforce",
        env_id="CartPole-v1",
        n_episodes=n_episodes,
        max_steps=200,
        random_state=seed,
    )
    ev = session.evaluate_rl(n_episodes=10, max_steps=200, random_state=seed)
    return {
        "backend": "native",
        "method": "reinforce_linear_softmax",
        "train_mean_return": fit.train_metrics.get("mean_return"),
        "eval_mean_return": ev.metrics.get("mean_return"),
        "n_episodes": n_episodes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML RL policy return benchmark (BC vs SB3 on CartPole)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/rl/results/policy_return.json"),
    )
    parser.add_argument("--total-timesteps", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(_bc_baseline(seed=args.seed))

    if gymnasium_available():
        try:
            runs.append(_reinforce_native(seed=args.seed))
        except Exception as exc:  # noqa: BLE001
            if "rl" not in str(exc).lower() and "gymnasium" not in str(exc).lower():
                raise

    if rl_industry_available():
        try:
            runs.append(
                _sb3_ppo(total_timesteps=args.total_timesteps, seed=args.seed)
            )
        except Exception as exc:  # noqa: BLE001
            if "rl-industry" not in str(exc).lower() and "stable_baselines3" not in str(
                exc
            ).lower():
                raise

    bc = next((r for r in runs if r.get("method") == "behavioral_cloning"), None)
    sb3 = next((r for r in runs if r.get("method") == "ppo"), None)
    payload = {
        "benchmark": "rl_policy_return",
        "capability_matrix": rl_capability_matrix(),
        "env_id": "CartPole-v1",
        "results": runs,
        "bc_eval_accuracy": None if bc is None else bc.get("eval_accuracy"),
        "sb3_eval_mean_return": None if sb3 is None else sb3.get("eval_mean_return"),
        "floor_note": (
            "SB3 PPO should exceed sklearn BC demonstration accuracy as an env "
            "return metric when buildml[rl-industry] is installed — different "
            "metrics (offline BC accuracy vs online env return)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if sb3 and bc:
        ret = float(sb3.get("eval_mean_return") or 0.0)
        if ret < 20.0 and args.total_timesteps < 30_000:
            print(
                "WARN: SB3 mean return still low; increase --total-timesteps for "
                "CartPole teaching benchmark.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
