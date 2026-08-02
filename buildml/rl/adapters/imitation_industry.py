"""Industry imitation adapters — BC MLP + GAIL-lite via imitation / SB3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rl.extras import require_gymnasium, require_imitation, require_stable_baselines3


@dataclass
class TabularMlpPolicy:
    """Tabular BC / GAIL MLP policy trained via imitation + SB3."""

    model: Any
    obs_dim: int
    n_actions: int
    method: str
    classes_: tuple[Any, ...] | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        obs = np.asarray(x, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        actions, _states = self.model.predict(obs, deterministic=True)
        return np.asarray(actions).reshape(-1)


def _build_spaces(obs_dim: int, n_actions: int) -> tuple[Any, Any]:
    gymnasium = require_gymnasium(feature="imitation industry BC/GAIL")
    spaces = gymnasium.spaces
    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32,
    )
    act_space = spaces.Discrete(n_actions)
    return obs_space, act_space


def fit_tabular_bc_mlp(
    x: np.ndarray,
    y_codes: np.ndarray,
    *,
    n_actions: int,
    n_epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    random_state: int | None = 0,
) -> tuple[TabularMlpPolicy, float, list[str], list[str]]:
    """Train an MLP BC policy on tabular (state, action) demonstrations."""
    require_imitation(feature="fit_imitation(backend='industry')")
    require_stable_baselines3(feature="fit_imitation(backend='industry')")
    from imitation.algorithms import bc
    from imitation.data.types import Transitions

    obs_dim = int(x.shape[1])
    n = int(x.shape[0])
    obs_space, act_space = _build_spaces(obs_dim, n_actions)

    transitions = Transitions(
        obs=np.asarray(x, dtype=np.float32),
        acts=np.asarray(y_codes, dtype=np.int64),
        infos=np.array([{}] * n, dtype=object),
        next_obs=np.asarray(x, dtype=np.float32),
        dones=np.ones(n, dtype=bool),
    )

    rng = np.random.default_rng(random_state)
    bc_trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=transitions,
        rng=rng,
        policy_kwargs={"net_arch": [64, 64]},
    )
    bc_trainer.train(
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
    )
    policy_model = bc_trainer.policy
    pred = policy_model.predict(np.asarray(x, dtype=np.float32), deterministic=True)[0]
    pred = np.asarray(pred).reshape(-1)
    train_score = float(np.mean(pred == y_codes))

    disclosures = [
        "Industry imitation BC trains an MLP policy via imitation+SB3 on train demos.",
        "Requires buildml[rl-industry] (imitation + stable-baselines3 + gymnasium).",
        "Honesty: tabular BC from tables — not inverse RL, not DAgger, not robotics.",
        "Offline RL / batch RL are out of scope; this is supervised cloning depth.",
        f"method=bc_mlp; n_epochs={n_epochs}; obs_dim={obs_dim}; n_actions={n_actions}.",
    ]
    warnings: list[str] = []
    wrapper = TabularMlpPolicy(
        model=policy_model,
        obs_dim=obs_dim,
        n_actions=n_actions,
        method="bc_mlp",
    )
    return wrapper, train_score, disclosures, warnings


def fit_tabular_gail_lite(
    x: np.ndarray,
    y_codes: np.ndarray,
    *,
    env_id: str,
    n_actions: int,
    total_timesteps: int = 8_000,
    random_state: int | None = 0,
) -> tuple[TabularMlpPolicy, float, list[str], list[str]]:
    """GAIL-lite: adversarial imitation with very small budgets (honest lite path)."""
    require_imitation(feature="fit_imitation(method='gail_lite')")
    sb3 = require_stable_baselines3(feature="fit_imitation(method='gail_lite')")
    gymnasium = require_gymnasium(feature="fit_imitation(method='gail_lite')")

    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.data.types import Transitions
    from imitation.rewards.reward_nets import BasicRewardNet

    obs_dim = int(x.shape[1])
    n = int(x.shape[0])

    try:
        env = gymnasium.make(env_id)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"gail_lite requires a valid Gymnasium env_id={env_id!r}: {exc}"
        ) from exc

    try:
        env_obs_dim = int(np.prod(env.observation_space.shape))
        if env_obs_dim != obs_dim:
            raise ValidationError(
                f"gail_lite demo obs_dim={obs_dim} must match env obs dim={env_obs_dim}."
            )
        if not hasattr(env.action_space, "n"):
            raise ValidationError(
                "gail_lite requires a discrete-action Gymnasium env."
            )
        if int(env.action_space.n) != int(n_actions):
            raise ValidationError(
                f"gail_lite demo n_actions={n_actions} must match env "
                f"n_actions={env.action_space.n}."
            )

        transitions = Transitions(
            obs=np.asarray(x, dtype=np.float32),
            acts=np.asarray(y_codes, dtype=np.int64),
            infos=np.array([{}] * n, dtype=object),
            next_obs=np.asarray(x, dtype=np.float32),
            dones=np.ones(n, dtype=bool),
        )

        reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        gail = GAIL(
            demonstrations=transitions,
            demo_batch_size=min(32, max(1, n)),
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=2,
            venv=env,
            reward_net=reward_net,
            gen_algo=sb3.PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=random_state,
                n_steps=128,
                batch_size=64,
            ),
        )
        gail.train(total_timesteps=int(total_timesteps))

        policy_model = gail.gen_algo
        pred_raw = policy_model.predict(
            np.asarray(x, dtype=np.float32), deterministic=True
        )[0]
        pred = np.asarray(pred_raw).reshape(-1)
        train_score = float(np.mean(pred == y_codes))

        disclosures = [
            "GAIL-lite runs a small-budget adversarial imitation loop (honest lite).",
            "Requires buildml[rl-industry] and env-compatible demonstration rows.",
            "Honesty: teaching-depth GAIL — not robotics / AV / multi-agent sims.",
            "Offline RL disclosures: imitation from demos, not batch offline RL.",
            f"method=gail_lite; env_id={env_id!r}; total_timesteps={total_timesteps}.",
        ]
        warnings = [
            "GAIL-lite uses minimal timesteps; expect variance vs sklearn BC.",
        ]
        wrapper = TabularMlpPolicy(
            model=policy_model,
            obs_dim=obs_dim,
            n_actions=n_actions,
            method="gail_lite",
        )
        return wrapper, train_score, disclosures, warnings
    finally:
        env.close()
