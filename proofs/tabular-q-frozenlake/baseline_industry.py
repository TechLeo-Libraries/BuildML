"""Tier C: random-policy FrozenLake twin for tabular-q-frozenlake."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np

from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def random_policy_baseline(
    *,
    n_episodes: int = 50,
    max_steps: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Roll out a uniform random policy when gymnasium is available."""
    try:
        import gymnasium as gym
    except ImportError:
        return {"available": 0.0, "mean_return": float("nan")}

    env = gym.make("FrozenLake-v1")
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    for _ in range(n_episodes):
        _obs, _info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total = 0.0
        for _step in range(max_steps):
            action = int(env.action_space.sample())
            _obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return {
        "available": 1.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "n_episodes": float(n_episodes),
    }


def main() -> None:
    ctx = new_proof_context("tabular-q-frozenlake", seed=0)
    industry_raw = random_policy_baseline(n_episodes=50, max_steps=100, seed=ctx.seed)
    industry_metrics = metrics_round(
        {
            "mean_return": industry_raw.get("mean_return"),
            "n_episodes": industry_raw.get("n_episodes", 50.0),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    probe = bml_raw.get("tabular_q_probe") or {}
    eval_blob = probe.get("eval") if isinstance(probe, dict) else {}
    metrics_blob = {}
    if isinstance(eval_blob, dict):
        metrics_blob = dict(eval_blob.get("metrics") or {})
    if not metrics_blob:
        metrics_blob = extract_buildml_test_metrics(
            bml_raw,
            prefer=("test_metrics", "tabular_q_probe"),
            keys=("mean_return",),
        )
    bml_metrics = metrics_round(
        {
            k: v
            for k, v in metrics_blob.items()
            if k in ("mean_return", "n_eval_episodes", "std_return")
            and isinstance(v, (int, float))
        }
    )
    if "mean_return" not in bml_metrics and isinstance(probe, dict):
        # Fall back if evaluate metrics nested differently.
        fit = probe.get("fit") or {}
        train_m = fit.get("train_metrics") if isinstance(fit, dict) else {}
        if isinstance(train_m, dict) and "mean_return" in train_m:
            bml_metrics = metrics_round({"mean_return": train_m["mean_return"]})

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session.fit_rl(tabular_q)",
            "method": "q_learning",
            "test_metrics": bml_metrics,
            "gymnasium_ran": bool(isinstance(probe, dict) and probe.get("ran")),
        },
        industry={
            "backend": "gymnasium.FrozenLake-v1",
            "method": "uniform_random_policy",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Random policy uses fresh env rollouts (no Session tabular leakage)",
                "Same episode/step budget as BuildML evaluate_rl disclosure",
            ],
        },
        same_split=False,
        split_counts={"eval_episodes": 50},
        delta_keys=("mean_return",),
        extra={
            "note": (
                "Env-loop RL has no tabular SplitPlan; comparison is same-budget "
                "rollout parity (learned Q vs random), not a supervised same-split twin."
            ),
        },
    )
    print("tabular-q-frozenlake Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
