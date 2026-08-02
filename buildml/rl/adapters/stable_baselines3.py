"""Stable-Baselines3 adapter for PPO / DQN / A2C on Gymnasium envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rl.extras import require_gymnasium, require_stable_baselines3

Sb3Algorithm = Literal["ppo", "dqn", "a2c"]


@dataclass
class SB3PolicyWrapper:
    """Checkpoint-friendly wrapper around an SB3 model."""

    model: Any
    env_id: str
    algorithm: Sb3Algorithm
    obs_dim: int
    n_actions: int
    total_timesteps: int = 0
    disclosures: tuple[str, ...] = ()

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[int, np.ndarray | None]:
        obs = np.asarray(observation, dtype=float).reshape(1, -1)
        action, state = self.model.predict(obs, deterministic=deterministic)
        return int(action), state


def _make_sb3_model(
    algorithm: Sb3Algorithm,
    env: Any,
    *,
    learning_rate: float,
    gamma: float,
    seed: int | None,
) -> Any:
    sb3 = require_stable_baselines3(feature=f"fit_rl SB3 {algorithm}")
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 0,
        "seed": seed,
        "gamma": gamma,
    }
    if algorithm == "ppo":
        return sb3.PPO(learning_rate=learning_rate, **common)
    if algorithm == "dqn":
        return sb3.DQN(learning_rate=learning_rate, **common)
    if algorithm == "a2c":
        return sb3.A2C(learning_rate=learning_rate, **common)
    raise ValidationError(
        f"Unknown SB3 algorithm={algorithm!r}. Supported: ppo, dqn, a2c."
    )


def train_sb3_policy(
    *,
    env_id: str = "CartPole-v1",
    algorithm: Sb3Algorithm = "ppo",
    total_timesteps: int = 20_000,
    max_steps: int = 500,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    random_state: int | None = 0,
) -> tuple[SB3PolicyWrapper, dict[str, float], list[str], list[str]]:
    """Train an SB3 policy on a discrete Gymnasium env."""
    gymnasium = require_gymnasium(feature="fit_rl(mode='gym_sb3')")
    require_stable_baselines3(feature="fit_rl(mode='gym_sb3')")
    disclosures = [
        "SB3 industry path trains PPO/DQN/A2C on a Gymnasium env loop.",
        "Requires buildml[rl-industry] (stable-baselines3 + gymnasium).",
        "Honesty: small discrete-action env teaching — not MuJoCo/robotics/AV.",
        "Offline RL / batch-constrained Q-learning are out of scope here.",
        f"env_id={env_id!r}; algorithm={algorithm}; "
        f"total_timesteps={total_timesteps}.",
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
                "gym_sb3 requires a discrete action space (action_space.n)."
            )
        obs_shape = getattr(obs_space, "shape", None)
        if not obs_shape:
            raise ValidationError(
                "gym_sb3 requires a Box-like observation space with a shape."
            )
        obs_dim = int(np.prod(obs_shape))
        n_actions = int(act_space.n)

        model = _make_sb3_model(
            algorithm,
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            seed=random_state,
        )
        model.learn(total_timesteps=int(total_timesteps))

        # Quick in-env return probe for train metrics.
        eval_metrics = evaluate_sb3_policy(
            SB3PolicyWrapper(
                model=model,
                env_id=env_id,
                algorithm=algorithm,
                obs_dim=obs_dim,
                n_actions=n_actions,
                total_timesteps=total_timesteps,
            ),
            n_episodes=10,
            max_steps=max_steps,
            random_state=random_state,
            deterministic=True,
        )
    finally:
        env.close()

    wrapper = SB3PolicyWrapper(
        model=model,
        env_id=env_id,
        algorithm=algorithm,
        obs_dim=obs_dim,
        n_actions=n_actions,
        total_timesteps=int(total_timesteps),
        disclosures=tuple(disclosures),
    )
    metrics = {
        "total_timesteps": float(total_timesteps),
        "n_eval_episodes": eval_metrics.get("n_eval_episodes", 0.0),
        "mean_return": eval_metrics.get("mean_return", float("nan")),
        "std_return": eval_metrics.get("std_return", float("nan")),
    }
    if (
        env_id.startswith("CartPole")
        and metrics["mean_return"] < 50.0
        and total_timesteps < 50_000
    ):
        warnings.append(
            "CartPole mean return is still low; increase total_timesteps — "
            "this is an honest small-env teaching loop, not a robotics product."
        )
    return wrapper, metrics, disclosures, warnings


def evaluate_sb3_policy(
    policy: SB3PolicyWrapper,
    *,
    env_id: str | None = None,
    n_episodes: int = 20,
    max_steps: int = 500,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> dict[str, float]:
    """Roll out an SB3 policy; return mean episode return."""
    gymnasium = require_gymnasium(feature="evaluate_rl(mode='gym_sb3')")
    resolved_env = env_id or policy.env_id
    env = gymnasium.make(resolved_env)
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
                action, _state = policy.model.predict(
                    flat.reshape(1, -1),
                    deterministic=deterministic,
                )
                step_out = env.step(int(action))
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


def act_sb3_observation(
    policy: SB3PolicyWrapper,
    observation: Any,
    *,
    deterministic: bool = True,
) -> tuple[int, tuple[float, ...]]:
    """Choose an action for a single observation vector."""
    flat = np.asarray(observation, dtype=float).reshape(-1)
    if flat.size != policy.obs_dim:
        raise ValidationError(
            f"Observation dim {flat.size} != policy.obs_dim={policy.obs_dim}."
        )
    action, _state = policy.predict(flat, deterministic=deterministic)
    # SB3 does not expose action probs for all algos; return one-hot-ish scores.
    scores = np.zeros(policy.n_actions, dtype=float)
    scores[int(action)] = 1.0
    return int(action), tuple(float(v) for v in scores.tolist())
