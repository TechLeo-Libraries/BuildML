"""Evaluate RL policies (offline bandit metrics or Gymnasium rollouts)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.rl.adapters.stable_baselines3 import SB3PolicyWrapper, evaluate_sb3_policy
from buildml.rl.act import act_rl
from buildml.rl.bandit import LinUCBPolicy, RewardModelBandit, offline_bandit_metrics
from buildml.rl.features import encode_discrete_actions, matrix_from_frame
from buildml.rl.gym_reinforce import LinearSoftmaxPolicy, evaluate_gym_policy
from buildml.rl.results import RlEvalResult, RlPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_rl(
    dataset: Dataset | None,
    plan: RlPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "validation",
    n_episodes: int | None = None,
    max_steps: int | None = None,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> RlEvalResult:
    """Evaluate a fitted RL policy.

    contextual_bandit:
        Offline DM / IPS / match-rate on a holdout partition (disclosed offline).
    gym_reinforce:
        Online env rollouts (mean return); not offline.
    """
    if plan.mode == "gym_reinforce":
        return _eval_gym(
            plan,
            n_episodes=n_episodes,
            max_steps=max_steps,
            random_state=random_state,
            deterministic=deterministic,
        )
    if plan.mode == "gym_sb3":
        return _eval_sb3(
            plan,
            n_episodes=n_episodes,
            max_steps=max_steps,
            random_state=random_state,
            deterministic=deterministic,
        )
    if plan.mode != "contextual_bandit":
        raise ValidationError(f"Unsupported RL mode for evaluate_rl: {plan.mode!r}.")
    if dataset is None:
        raise ValidationError("contextual_bandit evaluate_rl requires a Dataset.")
    return _eval_bandit(
        dataset,
        plan,
        split_plan,
        partition=partition,
        random_state=random_state,
        deterministic=deterministic,
    )


def _eval_bandit(
    dataset: Dataset,
    plan: RlPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll,
    random_state: int | None,
    deterministic: bool,
) -> RlEvalResult:
    if plan.action_column is None or plan.reward_column is None:
        raise ValidationError("Bandit plan is missing action_column / reward_column.")
    if partition == "all":
        frame = dataset._ensure_pandas()
    else:
        if split_plan is None:
            raise ValidationError(
                "A SplitPlan is required for partition-scoped evaluate_rl."
            )
        frame = frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]
    if frame.empty:
        return RlEvalResult(
            partition=str(partition),
            mode=plan.mode,
            n_rows=0,
            metrics={},
            offline=True,
            disclosures=("Empty partition; no offline bandit metrics.",),
            warnings=("Empty evaluation partition.",),
        )
    for col in (plan.action_column, plan.reward_column, *plan.columns):
        if col not in frame.columns:
            raise ValidationError(
                f"Column {col!r} missing from partition={partition!r}."
            )

    x = matrix_from_frame(frame, list(plan.columns))
    logged_arms, _, _ = encode_discrete_actions(
        frame[plan.action_column],
        classes=plan.arms_,
    )
    logged_rewards = frame[plan.reward_column].to_numpy(dtype=float)
    act = act_rl(
        dataset,
        plan,
        split_plan,
        partition=partition,
        deterministic=deterministic,
        random_state=random_state,
    )
    policy_arms, _, _ = encode_discrete_actions(
        pd.Series(list(act.actions)),
        classes=plan.arms_,
    )

    predicted = None
    policy = plan.policy_
    if isinstance(policy, RewardModelBandit):
        predicted = policy.predicted_rewards(x)
    elif isinstance(policy, LinUCBPolicy):
        # Use expected reward θ·x (without exploration bonus) as DM proxy.
        predicted = np.zeros((x.shape[0], plan.n_arms), dtype=float)
        for i in range(x.shape[0]):
            row = x[i]
            for a in range(plan.n_arms):
                a_inv = np.linalg.inv(policy.A[a])
                theta = a_inv @ policy.b[a]
                predicted[i, a] = float(theta @ row)

    propensity = None
    if plan.propensity_model_ is not None:
        propensity = np.asarray(
            plan.propensity_model_.predict_proba(x), dtype=float
        )

    metrics = offline_bandit_metrics(
        x=x,
        logged_arms=logged_arms,
        logged_rewards=logged_rewards,
        policy_arms=policy_arms,
        predicted_rewards=predicted,
        propensity=propensity,
    )
    disclosures = [
        "OFFLINE metrics: direct_method / ips / action_match_rate are not online A/B results.",
        "IPS uses a train-fitted propensity model π_b(a|x); treat with caution under confounding.",
        "Holdout rows never update the bandit policy.",
    ]
    warnings: list[str] = []
    if propensity is None:
        warnings.append("Propensity model missing; ips metric is NaN.")
    return RlEvalResult(
        partition=str(partition),
        mode=plan.mode,
        n_rows=int(x.shape[0]),
        metrics=metrics,
        offline=True,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _eval_gym(
    plan: RlPlan,
    *,
    n_episodes: int | None,
    max_steps: int | None,
    random_state: int | None,
    deterministic: bool,
) -> RlEvalResult:
    policy = plan.policy_
    if not isinstance(policy, LinearSoftmaxPolicy):
        raise ValidationError("gym_reinforce plan is missing a LinearSoftmaxPolicy.")
    if plan.env_id is None:
        raise ValidationError("gym_reinforce plan is missing env_id.")
    cfg = plan.config or {}
    metrics = evaluate_gym_policy(
        policy,
        env_id=plan.env_id,
        n_episodes=int(n_episodes if n_episodes is not None else 20),
        max_steps=int(max_steps if max_steps is not None else cfg.get("max_steps", 500)),
        random_state=random_state,
        deterministic=deterministic,
    )
    return RlEvalResult(
        partition=None,
        mode=plan.mode,
        n_rows=int(metrics.get("n_eval_episodes", 0)),
        metrics=metrics,
        offline=False,
        disclosures=(
            "Gymnasium evaluation rolls out the policy in the env (online returns).",
            "Requires buildml[rl] (gymnasium).",
            "Honesty: small-env teaching loop — not MuJoCo/robotics.",
        ),
    )


def _eval_sb3(
    plan: RlPlan,
    *,
    n_episodes: int | None,
    max_steps: int | None,
    random_state: int | None,
    deterministic: bool,
) -> RlEvalResult:
    policy = plan.policy_
    if not isinstance(policy, SB3PolicyWrapper):
        raise ValidationError("gym_sb3 plan is missing an SB3PolicyWrapper.")
    if plan.env_id is None:
        raise ValidationError("gym_sb3 plan is missing env_id.")
    cfg = plan.config or {}
    metrics = evaluate_sb3_policy(
        policy,
        env_id=plan.env_id,
        n_episodes=int(n_episodes if n_episodes is not None else 20),
        max_steps=int(max_steps if max_steps is not None else cfg.get("max_steps", 500)),
        random_state=random_state,
        deterministic=deterministic,
    )
    return RlEvalResult(
        partition=None,
        mode=plan.mode,
        n_rows=int(metrics.get("n_eval_episodes", 0)),
        metrics=metrics,
        offline=False,
        disclosures=(
            "SB3 evaluation rolls out the policy in the env (online returns).",
            "Requires buildml[rl-industry] (stable-baselines3 + gymnasium + imitation).",
            "Honesty: small-env teaching loop — not MuJoCo/robotics/AV.",
            "Offline RL (batch RL) is out of scope; bandit IPS/DM are separate.",
        ),
    )
