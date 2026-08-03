"""Baseline: random policy on FrozenLake for tabular-q proof comparison."""

from __future__ import annotations

import numpy as np


def random_policy_baseline(*, n_episodes: int = 50, max_steps: int = 100, seed: int = 0) -> dict[str, float]:
    """Roll out a uniform random policy when gymnasium is available."""
    try:
        import gymnasium as gym
    except ImportError:
        return {"available": 0.0, "mean_return": float("nan")}

    env = gym.make("FrozenLake-v1")
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    for _ in range(n_episodes):
        obs, _info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total = 0.0
        for _step in range(max_steps):
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, _info = env.step(action)
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
