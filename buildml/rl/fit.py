"""Learn a decision policy from rewards rather than from labels.

Supervised learning is told the right answer. Reinforcement learning is not: it
chooses an action, sees a reward, and has to work out which action was
responsible. That gap is what makes it hard, and it is why this module offers
four modes rather than one algorithm — they differ in what the world lets you
do.

**Contextual bandits** are the mode most tabular work needs, and the only one
that runs without extra dependencies. Each row of a log records a situation, the
action taken, and the reward observed. There is no sequence: the action does not
change what happens next, so credit assignment is one step deep. Recommendation
slots, offer selection, and treatment choice usually fit this shape. Learning is
offline, from a fixed log, and holdout scoring is therefore an *estimate* of how
an alternative policy would have performed — see :mod:`buildml.rl.evaluate` for
what that estimate can and cannot support.

The other three modes need an environment to interact with, because they handle
sequential problems where an action changes the next situation.
``'gym_reinforce'`` runs a linear-softmax policy gradient, ``'tabular_q'`` runs
TD control over a discretised state table, and ``'gym_sb3'`` hands off to
Stable-Baselines3. All three need ``buildml[rl]`` or ``buildml[rl-industry]``,
and none of them read the Session's tabular partitions — they learn in the
environment, and the Session merely holds the resulting policy so it can be
checkpointed and resumed alongside the rest of your work.

Two things are deliberately out of scope. This is not batch offline RL: CQL,
IQL, and Decision Transformers learn a sequential policy from logged
trajectories without an environment, and nothing here does that. Nor is it a
robotics or multi-agent platform.

See Also
--------
buildml.rl.evaluate : Offline estimators, and why they are estimates.
buildml.rl.act : Choosing actions once a policy is fitted.
buildml.rl.imitation : Copying a demonstrator when you have no reward signal.
"""

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
from buildml.rl.tabular import train_tabular_control
from buildml.rl.types import (
    BanditAlgorithm,
    RlBackend,
    RlConfig,
    RlMode,
    Sb3Algorithm,
    TabularAlgorithm,
)


def fit_rl(
    dataset: Dataset | None,
    split_plan: SplitPlan | None,
    *,
    backend: RlBackend | None = None,
    mode: RlMode | None = None,
    algorithm: BanditAlgorithm | TabularAlgorithm | Sb3Algorithm | str = "linucb",
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
    n_bins: int = 8,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
) -> tuple[RlPlan, RlFitResult]:
    """Learn a policy that chooses actions to earn reward.

    One entry point for four different situations. Which one applies depends on
    what you have: a log of past decisions and their outcomes, or an environment
    you can interact with.

    Parameters
    ----------
    dataset:
        The logged table for ``'contextual_bandit'``. Ignored by the
        environment modes, which learn from the environment instead — pass
        ``None`` there.
    split_plan:
        Required for ``'contextual_bandit'``, so the policy is fitted on train
        rows and can be estimated honestly on holdout ones. Ignored by the
        environment modes.
    backend:
        ``'sklearn'`` for bandits, ``'native'`` for BuildML's own environment
        loops, ``'industry'`` for Stable-Baselines3. Left unset, it follows
        from the mode and algorithm.
    mode:
        Which problem you are solving. Inferred from the algorithm when not
        given, so ``algorithm='q_learning'`` selects ``'tabular_q'`` on its own.
    algorithm:
        Within bandits: ``'linucb'`` (default) keeps an uncertainty estimate per
        arm and explores the ones it is least sure about, which is the
        principled choice when contexts are informative;
        ``'epsilon_greedy'`` explores at a fixed rate, simpler and easier to
        reason about; ``'softmax'`` explores in proportion to estimated reward.
        Within tabular control: ``'q_learning'`` and ``'double_q_learning'``
        learn the greedy policy's value while behaving exploratorily, whereas
        ``'sarsa'`` and ``'expected_sarsa'`` learn the value of the policy they
        actually follow — the latter matter when exploration itself is costly.
        Within Stable-Baselines3: ``'ppo'``, ``'dqn'``, or ``'a2c'``.
    columns:
        Bandit context features. Defaults to the usable columns with the action
        and reward columns excluded.
    action_column:
        The action taken in each logged row. Defaults to the Dataset target.
    reward_column:
        The reward observed. Falls back to a column literally named ``reward``,
        or to a numeric target when the action came from elsewhere. When the
        target *is* the action, this must be passed explicitly — quietly reusing
        the target as its own reward would produce a meaningless policy.
    alpha:
        LinUCB exploration width. Higher values try under-explored arms more
        readily.
    epsilon:
        Random-action probability for epsilon-greedy and for tabular
        exploration.
    temperature:
        Softmax sharpness. Low values concentrate on the best arm.
    random_state:
        Seed. RL results vary widely across seeds; a single run is an anecdote.
    prefer_reduce_components:
        Use an attached dimensionality reduction for the context.
    reduce_plan:
        An explicit reduction plan.
    env_id:
        The Gymnasium environment for the environment modes.
    n_episodes:
        Training episodes for ``'gym_reinforce'`` and ``'tabular_q'``.
    max_steps:
        Per-episode step cap, so a non-terminating episode cannot hang the run.
    learning_rate:
        Step size for the environment modes.
    gamma:
        Discount factor. Near 1.0 values long-run reward; lower values make the
        policy short-sighted.
    total_timesteps:
        Interaction budget for Stable-Baselines3.
    n_bins:
        Bins per observation dimension for ``'tabular_q'``. The state table
        grows as ``n_bins ** obs_dim``, so this is what decides whether tabular
        control is viable for a given environment.
    epsilon_min / epsilon_decay:
        Exploration schedule for tabular control: start at ``epsilon``, multiply
        by ``epsilon_decay`` each episode, never fall below ``epsilon_min``.

    Returns
    -------
    RlPlan
        The fitted policy, ready for :func:`~buildml.rl.act.act_rl` and
        :func:`~buildml.rl.evaluate.evaluate_rl`.
    RlFitResult
        What the fit saw: rows or episodes, arms, columns, and per-mode
        training metrics.

    Raises
    ------
    LeakageError
        If a bandit fit is attempted without a train partition.
    ValidationError
        If the mode or algorithm is unknown, if a bandit fit is missing its
        dataset, split plan, action column, or reward column, or if the reward
        column is non-numeric or contains nulls.
    MissingExtraError
        If an environment mode is requested without ``buildml[rl]`` or
        ``buildml[rl-industry]``.

    Notes
    -----
    **A bandit's training metrics are not a score.** ``mean_logged_reward`` is
    what the *logging* policy earned, which is the baseline the new policy has
    to beat — not evidence that it does. Use
    :func:`~buildml.rl.evaluate.evaluate_rl` for that.

    **A propensity model is fitted alongside the bandit** to estimate how likely
    the logging policy was to take each action. Inverse propensity scoring needs
    it. If it cannot be fitted, the fit still succeeds and records a warning,
    and holdout evaluation falls back to direct-method estimates alone.

    **The environment modes ignore your data entirely.** They learn in the
    environment; the Session stores the policy so it travels with your
    checkpoints. If you expected your tabular rows to influence a
    ``'gym_reinforce'`` fit, they did not.

    Examples
    --------
    >>> plan, result = fit_rl(  # doctest: +SKIP
    ...     dataset, split_plan, action_column="offer", reward_column="revenue"
    ... )
    >>> result.n_arms, result.train_metrics["mean_logged_reward"]  # doctest: +SKIP
    (4, 12.7)

    See Also
    --------
    buildml.rl.evaluate.evaluate_rl : Estimate holdout performance offline.
    buildml.rl.act.act_rl : Choose actions with the fitted policy.
    buildml.rl.catalog.rl_capability_matrix : Every mode and algorithm as data.
    """
    resolved_backend, resolved_mode, resolved_algo = resolve_rl_backend_mode_algorithm(
        backend=backend,
        mode=mode,
        algorithm=str(algorithm),
    )
    if resolved_mode == "tabular_q":
        return _fit_tabular_q(
            env_id=env_id,
            algorithm=resolved_algo,
            n_episodes=n_episodes,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            n_bins=n_bins,
            random_state=random_state,
            backend=resolved_backend,
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
            "Supported: contextual_bandit, gym_reinforce, tabular_q, gym_sb3."
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


def _fit_tabular_q(
    *,
    env_id: str,
    algorithm: str,
    n_episodes: int,
    max_steps: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    n_bins: int,
    random_state: int | None,
    backend: RlBackend,
) -> tuple[RlPlan, RlFitResult]:
    policy, metrics, disclosures, warnings = train_tabular_control(
        env_id=env_id,
        algorithm=algorithm,
        n_episodes=n_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        n_bins=n_bins,
        random_state=random_state,
    )
    disclosures = list(disclosures) + [
        "tabular_q does not fit on Session tabular partitions; "
        "the Session hosts the checkpointed Q-table policy for workflow continuity.",
        "Off-policy TD control here is still an ONLINE env loop — it is not "
        "batch offline RL (CQL / IQL / Decision Transformer remain out of scope).",
    ]
    config = RlConfig(
        mode="tabular_q",
        backend=backend,
        algorithm=algorithm,
        env_id=env_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        n_bins=n_bins,
        random_state=random_state,
    )
    config_dict = config.to_dict()
    config_dict["discretizer"] = policy.discretizer.to_dict()
    plan = RlPlan(
        mode="tabular_q",
        backend=backend,
        algorithm=algorithm,
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
        mode="tabular_q",
        backend=backend,
        algorithm=algorithm,
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
