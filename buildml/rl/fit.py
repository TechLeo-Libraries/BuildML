"""Fit RL policies: contextual bandit (core) or Gymnasium REINFORCE (optional)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.rl.bandit import LinUCBPolicy, RewardModelBandit, fit_propensity_model
from buildml.rl.catalog import resolve_rl_backend_mode_algorithm
from buildml.rl.features import encode_discrete_actions, matrix_from_frame, resolve_rl_columns
from buildml.rl.gym_reinforce import train_gym_reinforce
from buildml.rl.results import RlFitResult, RlPlan
from buildml.rl.types import BanditAlgorithm, RlBackend, RlConfig, RlMode, Sb3Algorithm


def fit_rl(
    dataset: Dataset | None,
    split_plan: SplitPlan | None,
    *,
    backend: RlBackend | None = None,
    mode: RlMode | None = None,
    algorithm: BanditAlgorithm | Sb3Algorithm | str = "linucb",
    columns: list[str] | None = None,
    action_column: str | None = None,
    reward_column: str | None = None,
    alpha: float = 1.0,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
    env_id: str = "CartPole-v1",
    n_episodes: int = 200,
    max_steps: int = 500,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    total_timesteps: int = 20_000,
) -> tuple[RlPlan, RlFitResult]:
    """Fit a Session-shaped RL policy.

    Backends
    --------
    sklearn (default for bandits):
        Contextual bandit on logged train rows.
    native (``buildml[rl]``):
        REINFORCE-lite linear softmax Gymnasium loop.
    industry (``buildml[rl-industry]``):
        Stable-Baselines3 PPO/DQN/A2C on Gymnasium envs.

    Modes
    -----
    contextual_bandit:
        Train-only offline learning from logged (context, action, reward) rows.
    gym_reinforce:
        Optional Gymnasium REINFORCE-lite env loop (requires ``buildml[rl]``).
    gym_sb3:
        SB3 industry env loop (requires ``buildml[rl-industry]``).
    """
    resolved_backend, resolved_mode, resolved_algo = resolve_rl_backend_mode_algorithm(
        backend=backend,
        mode=mode,
        algorithm=str(algorithm),
    )
    if resolved_mode == "gym_sb3":
        return _fit_gym_sb3(
            env_id=env_id,
            algorithm=resolved_algo,  # type: ignore[arg-type]
            total_timesteps=total_timesteps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            random_state=random_state,
            backend=resolved_backend,
        )
    if resolved_mode == "gym_reinforce":
        return _fit_gym_reinforce(
            env_id=env_id,
            n_episodes=n_episodes,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            random_state=random_state,
            algorithm=resolved_algo,
            backend=resolved_backend,
        )
    if resolved_mode != "contextual_bandit":
        raise ValidationError(
            f"Unknown RL mode={resolved_mode!r}. "
            "Supported: contextual_bandit, gym_reinforce, gym_sb3."
        )
    if dataset is None or split_plan is None:
        raise ValidationError(
            "contextual_bandit requires a Dataset and SplitPlan (train-only fit)."
        )
    return _fit_contextual_bandit(
        dataset,
        split_plan,
        backend=resolved_backend,
        algorithm=resolved_algo,  # type: ignore[arg-type]
        columns=columns,
        action_column=action_column,
        reward_column=reward_column,
        alpha=alpha,
        epsilon=epsilon,
        temperature=temperature,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=reduce_plan,
    )


def _fit_contextual_bandit(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    backend: RlBackend,
    algorithm: BanditAlgorithm,
    columns: list[str] | None,
    action_column: str | None,
    reward_column: str | None,
    alpha: float,
    epsilon: float,
    temperature: float,
    random_state: int | None,
    prefer_reduce_components: bool,
    reduce_plan: Any | None,
) -> tuple[RlPlan, RlFitResult]:
    assert_fit_partition(split_plan, "train")
    target = dataset.require_target()
    train = frame_for_partition(dataset, split_plan, "train")

    action_col = action_column or target
    if action_col not in train.columns:
        raise ValidationError(
            f"action_column={action_col!r} missing from the train partition."
        )
    reward_col = reward_column
    if reward_col is None:
        # Prefer an explicit 'reward' column; else refuse silent misuse of target
        # when action already consumed the target.
        if "reward" in train.columns:
            reward_col = "reward"
        elif action_col != target and pd.api.types.is_numeric_dtype(train[target]):
            reward_col = target
        else:
            raise ValidationError(
                "contextual_bandit requires reward_column=... (or a numeric "
                "'reward' column). When the Dataset target is used as the action, "
                "pass an explicit reward_column."
            )
    if reward_col not in train.columns:
        raise ValidationError(
            f"reward_column={reward_col!r} missing from the train partition."
        )
    if not pd.api.types.is_numeric_dtype(train[reward_col]):
        raise ValidationError("reward_column must be numeric.")
    if train[reward_col].isna().any():
        raise ValidationError("reward_column has nulls; impute or drop before fit_rl.")

    exclude = {action_col, reward_col}
    cols, used_reduce, disclosures = resolve_rl_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
        exclude_columns=tuple(exclude),
    )
    x = matrix_from_frame(train, cols)
    arms, encoder, arm_labels = encode_discrete_actions(train[action_col])
    rewards = train[reward_col].to_numpy(dtype=float)
    n_arms = len(arm_labels)
    dim = int(x.shape[1])
    n_train = int(x.shape[0])
    warnings: list[str] = []

    disclosures.extend(
        [
            "Contextual bandit fits on logged train (context, action, reward) only.",
            "Validation/test are never used to update the bandit policy.",
            "Holdout evaluation uses offline estimators (DM / IPS) — not online A/B.",
            "Honesty: Session tabular bandits — not a multi-agent / robotics platform.",
            f"algorithm={algorithm}; n_arms={n_arms}; action={action_col!r}; "
            f"reward={reward_col!r}.",
        ]
    )

    if algorithm == "linucb":
        policy: Any = LinUCBPolicy(n_arms=n_arms, dim=dim, alpha=float(alpha))
        policy.fit_logged(x, arms, rewards)
    elif algorithm in {"epsilon_greedy", "softmax"}:
        policy = RewardModelBandit(
            n_arms=n_arms,
            dim=dim,
            algorithm=algorithm,
            epsilon=float(epsilon),
            temperature=float(temperature),
            random_state=random_state,
        )
        policy.fit_logged(x, arms, rewards)
    else:
        raise ValidationError(
            f"Unknown bandit algorithm={algorithm!r}. "
            "Supported: linucb, epsilon_greedy, softmax."
        )

    try:
        propensity = fit_propensity_model(x, arms, random_state=random_state)
    except ValidationError as exc:
        warnings.append(str(exc))
        propensity = None

    train_metrics = {
        "n_train_rows": float(n_train),
        "n_arms": float(n_arms),
        "mean_logged_reward": float(np.mean(rewards)),
    }
    config = RlConfig(
        mode="contextual_bandit",
        backend=backend,
        algorithm=algorithm,
        columns=tuple(cols),
        action_column=action_col,
        reward_column=reward_col,
        alpha=alpha,
        epsilon=epsilon,
        temperature=temperature,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
    )
    plan = RlPlan(
        mode="contextual_bandit",
        backend=backend,
        algorithm=algorithm,
        columns=tuple(cols),
        action_column=action_col,
        reward_column=reward_col,
        n_train_rows=n_train,
        n_arms=n_arms,
        arms_=arm_labels,
        label_encoder_=encoder,
        policy_=policy,
        propensity_model_=propensity,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
        train_metrics=train_metrics,
    )
    result = RlFitResult(
        mode="contextual_bandit",
        backend=backend,
        algorithm=algorithm,
        n_train_rows=n_train,
        n_arms=n_arms,
        columns=tuple(cols),
        action_column=action_col,
        reward_column=reward_col,
        train_metrics=train_metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _fit_gym_reinforce(
    *,
    env_id: str,
    n_episodes: int,
    max_steps: int,
    learning_rate: float,
    gamma: float,
    random_state: int | None,
    algorithm: str,
    backend: RlBackend,
) -> tuple[RlPlan, RlFitResult]:
    policy, metrics, disclosures, warnings = train_gym_reinforce(
        env_id=env_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        random_state=random_state,
    )
    disclosures = list(disclosures) + [
        "gym_reinforce does not fit on Session tabular partitions; "
        "the Session hosts the checkpointed env policy for workflow continuity.",
    ]
    config = RlConfig(
        mode="gym_reinforce",
        backend=backend,
        algorithm="linucb" if algorithm == "linucb" else algorithm,  # unused
        env_id=env_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        random_state=random_state,
    )
    # Record algorithm field as reinforce for honesty in meta.
    config_dict = config.to_dict()
    config_dict["algorithm"] = "reinforce_linear_softmax"
    plan = RlPlan(
        mode="gym_reinforce",
        backend=backend,
        algorithm="reinforce_linear_softmax",
        columns=(),
        action_column=None,
        reward_column=None,
        n_train_rows=int(metrics.get("n_episodes", 0)),
        n_arms=int(policy.n_actions),
        arms_=tuple(range(int(policy.n_actions))),
        policy_=policy,
        env_id=env_id,
        obs_dim=int(policy.obs_dim),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config_dict,
        train_metrics=metrics,
    )
    result = RlFitResult(
        mode="gym_reinforce",
        backend=backend,
        algorithm="reinforce_linear_softmax",
        n_train_rows=int(metrics.get("n_episodes", 0)),
        n_arms=int(policy.n_actions),
        columns=(),
        env_id=env_id,
        train_metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _fit_gym_sb3(
    *,
    env_id: str,
    algorithm: Sb3Algorithm,
    total_timesteps: int,
    max_steps: int,
    learning_rate: float,
    gamma: float,
    random_state: int | None,
    backend: RlBackend,
) -> tuple[RlPlan, RlFitResult]:
    from buildml.rl.adapters.stable_baselines3 import train_sb3_policy

    policy, metrics, disclosures, warnings = train_sb3_policy(
        env_id=env_id,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        random_state=random_state,
    )
    disclosures = list(disclosures) + [
        "gym_sb3 does not fit on Session tabular partitions; "
        "the Session hosts the checkpointed SB3 policy for workflow continuity.",
    ]
    config = RlConfig(
        mode="gym_sb3",
        backend=backend,
        algorithm=algorithm,  # type: ignore[arg-type]
        env_id=env_id,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        random_state=random_state,
        total_timesteps=total_timesteps,
    )
    config_dict = config.to_dict()
    config_dict["algorithm"] = algorithm
    plan = RlPlan(
        mode="gym_sb3",
        backend=backend,
        algorithm=algorithm,
        columns=(),
        action_column=None,
        reward_column=None,
        n_train_rows=int(metrics.get("total_timesteps", 0)),
        n_arms=int(policy.n_actions),
        arms_=tuple(range(int(policy.n_actions))),
        policy_=policy,
        env_id=env_id,
        obs_dim=int(policy.obs_dim),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config_dict,
        train_metrics=metrics,
    )
    result = RlFitResult(
        mode="gym_sb3",
        backend=backend,
        algorithm=algorithm,
        n_train_rows=int(metrics.get("total_timesteps", 0)),
        n_arms=int(policy.n_actions),
        columns=(),
        env_id=env_id,
        train_metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result
