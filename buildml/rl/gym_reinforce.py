"""Optional Gymnasium REINFORCE-lite (linear softmax policy) — behind buildml[rl]."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rl.extras import require_gymnasium
from buildml.rl.features import softmax


@dataclass
class LinearSoftmaxPolicy:
    """Linear softmax policy π(a|s) = softmax(W s) for discrete-action envs."""

    n_actions: int
    obs_dim: int
    learning_rate: float = 0.01
    gamma: float = 0.99
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.weights.size == 0:
            # Small init helps early exploration without torch.
            rng = np.random.default_rng(0)
            self.weights = rng.normal(
                scale=0.01, size=(self.n_actions, self.obs_dim)
            ).astype(float)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=float).reshape(-1)
        return self.weights @ x

    def probs(self, obs: np.ndarray) -> np.ndarray:
        return softmax(self.logits(obs))

    def act(
        self, obs: np.ndarray, *, rng: np.random.Generator, deterministic: bool = False
    ) -> int:
        probs = self.probs(obs)
        if deterministic:
            return int(np.argmax(probs))
        return int(rng.choice(self.n_actions, p=probs))

    def update_episode(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
    ) -> float:
        """REINFORCE with returns-to-go; returns episode return."""
        if not rewards:
            return 0.0
        returns: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = float(r) + self.gamma * g
            returns.insert(0, g)
        ret_arr = np.asarray(returns, dtype=float)
        # Baseline = mean return (variance reduction lite).
        ret_arr = ret_arr - float(np.mean(ret_arr))
        for obs, action, gt in zip(observations, actions, ret_arr, strict=True):
            x = np.asarray(obs, dtype=float).reshape(-1)
            probs = self.probs(x)
            # ∇ log π(a|s) = x * (1_a - π)
            grad = np.zeros_like(self.weights)
            for a in range(self.n_actions):
                indicator = 1.0 if a == int(action) else 0.0
                grad[a] = x * (indicator - probs[a])
            self.weights = self.weights + self.learning_rate * float(gt) * grad
        return float(sum(rewards))


def train_gym_reinforce(
    *,
    env_id: str = "CartPole-v1",
    n_episodes: int = 200,
    max_steps: int = 500,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    random_state: int | None = 0,
) -> tuple[LinearSoftmaxPolicy, dict[str, float], list[str], list[str]]:
    """Train a linear REINFORCE policy on a discrete Gymnasium env."""
    gymnasium = require_gymnasium(feature="fit_rl(mode='gym_reinforce')")
    disclosures = [
        "Gymnasium REINFORCE-lite trains a linear softmax policy in an env loop.",
        "This path requires buildml[rl] (gymnasium). Core BC/bandit paths do not.",
        "Honesty: small discrete-action env teaching loop — not MuJoCo/robotics.",
        f"env_id={env_id!r}; n_episodes={n_episodes}; gamma={gamma}.",
    ]
    warnings: list[str] = []
    try:
        env = gymnasium.make(env_id)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"Failed to create Gymnasium env_id={env_id!r}: {exc}"
        ) from exc

    try:
        obs_space = env.observation_space
        act_space = env.action_space
        if not hasattr(act_space, "n"):
            raise ValidationError(
                "gym_reinforce requires a discrete action space (action_space.n)."
            )
        obs_shape = getattr(obs_space, "shape", None)
        if not obs_shape:
            raise ValidationError(
                "gym_reinforce requires a Box-like observation space with a shape."
            )
        obs_dim = int(np.prod(obs_shape))
        n_actions = int(act_space.n)
        policy = LinearSoftmaxPolicy(
            n_actions=n_actions,
            obs_dim=obs_dim,
            learning_rate=learning_rate,
            gamma=gamma,
        )
        # Re-seed weights with caller random_state.
        rng = np.random.default_rng(random_state)
        policy.weights = rng.normal(
            scale=0.01, size=(n_actions, obs_dim)
        ).astype(float)

        returns: list[float] = []
        for ep in range(int(n_episodes)):
            reset_out = env.reset(seed=None if random_state is None else int(random_state) + ep)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            observations: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            for _ in range(int(max_steps)):
                flat = np.asarray(obs, dtype=float).reshape(-1)
                action = policy.act(flat, rng=rng, deterministic=False)
                step_out = env.step(action)
                if len(step_out) == 5:
                    next_obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    next_obs, reward, done, _info = step_out
                observations.append(flat)
                actions.append(int(action))
                rewards.append(float(reward))
                obs = next_obs
                if done:
                    break
            ep_return = policy.update_episode(observations, actions, rewards)
            returns.append(ep_return)
    finally:
        env.close()

    metrics = {
        "n_episodes": float(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "last_return": float(returns[-1]) if returns else float("nan"),
        "mean_return_last_20": float(np.mean(returns[-20:])) if returns else float("nan"),
    }
    if metrics["mean_return_last_20"] < 20 and env_id.startswith("CartPole"):
        warnings.append(
            "CartPole mean return (last 20) is still low; try more episodes "
            "or a different learning_rate — this is a lite teaching loop."
        )
    return policy, metrics, disclosures, warnings


def evaluate_gym_policy(
    policy: LinearSoftmaxPolicy,
    *,
    env_id: str,
    n_episodes: int = 20,
    max_steps: int = 500,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> dict[str, float]:
    """Roll out a fitted Gymnasium policy; return mean episode return."""
    gymnasium = require_gymnasium(feature="evaluate_rl(mode='gym_reinforce')")
    env = gymnasium.make(env_id)
    rng = np.random.default_rng(random_state)
    returns: list[float] = []
    try:
        for ep in range(int(n_episodes)):
            reset_out = env.reset(
                seed=None if random_state is None else int(random_state) + 10_000 + ep
            )
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            total = 0.0
            for _ in range(int(max_steps)):
                flat = np.asarray(obs, dtype=float).reshape(-1)
                action = policy.act(flat, rng=rng, deterministic=deterministic)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, _info = step_out
                total += float(reward)
                if done:
                    break
            returns.append(total)
    finally:
        env.close()
    return {
        "n_eval_episodes": float(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "std_return": float(np.std(returns)) if returns else float("nan"),
        "min_return": float(np.min(returns)) if returns else float("nan"),
        "max_return": float(np.max(returns)) if returns else float("nan"),
    }


def act_gym_observation(
    policy: LinearSoftmaxPolicy,
    observation: Any,
    *,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> tuple[int, tuple[float, ...]]:
    """Choose an action for a single observation vector."""
    rng = np.random.default_rng(random_state)
    flat = np.asarray(observation, dtype=float).reshape(-1)
    if flat.size != policy.obs_dim:
        raise ValidationError(
            f"Observation dim {flat.size} != policy.obs_dim={policy.obs_dim}."
        )
    probs = policy.probs(flat)
    action = policy.act(flat, rng=rng, deterministic=deterministic)
    return int(action), tuple(float(p) for p in probs.tolist())
