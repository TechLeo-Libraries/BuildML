"""Estimate what a policy would earn, and be clear about how good the estimate is.

Evaluating a policy is harder than evaluating a classifier, and the reason is
worth understanding before reading any number this module produces.

A classifier can be checked against holdout labels because the right answer was
recorded. A policy cannot: your log records what reward followed the action that
*was* taken, and says nothing about the reward that would have followed the
action your new policy would take instead. That missing quantity — the
counterfactual — is the whole difficulty.

Two estimators approach it from opposite directions, which is why both are
reported.

**The direct method** fits a model of reward given context and action, then asks
it what the new policy's choices would have earned. It uses every row, so it is
low-variance, but it inherits every error in the reward model — and the reward
model is least reliable precisely for the context-action pairs the log rarely
contains, which are often the ones a new policy favours.

**Inverse propensity scoring** takes the opposite tack. It keeps only rows where
the new policy agrees with the log, and reweights each by how unlikely the
logging policy was to have taken that action. It is unbiased when the propensity
estimates are right, but a rare action produces a large weight, and a handful of
large weights can dominate the estimate.

**When the two disagree, believe neither.** Agreement is weak evidence the
estimate is sound; disagreement is strong evidence it is not.

``action_match_rate`` is the sanity check to read first. If the new policy picks
the logged action almost every time, it has barely changed anything and the
estimates are trivially reliable. If it almost never agrees, both estimates rest
on very little overlapping data and should not drive a decision.

None of this applies to the environment modes. There the policy is actually run,
so the returns are measured rather than estimated — ``offline`` on the result is
``False``, and the number means what it says.

See Also
--------
buildml.rl.bandit : The estimator implementations.
buildml.rl.fit : Fitting the policy and its propensity model.
"""

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
from buildml.rl.tabular import TabularValuePolicy, evaluate_tabular_policy

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
    """Score a fitted policy, offline for bandits and by rollout otherwise.

    Bandit plans are scored against a holdout partition using the two offline
    estimators described in the module docstring. Environment plans are scored
    by actually running the policy for a number of episodes.

    Parameters
    ----------
    dataset:
        The holdout table, for ``'contextual_bandit'``. Not used by the
        environment modes.
    plan:
        The fitted policy from :func:`~buildml.rl.fit.fit_rl`.
    split_plan:
        Required when a bandit plan is scored on a partition other than
        ``'all'``.
    partition:
        Which rows to score, for bandit plans. Defaults to ``'validation'``.
    n_episodes:
        Rollout episodes for the environment modes. Defaults to 20. Returns
        vary substantially between episodes, so a small number gives a noisy
        mean.
    max_steps:
        Per-episode step cap. Falls back to the value the plan was fitted with.
    random_state:
        Seed for rollouts and for stochastic acting.
    deterministic:
        Score the greedy policy (default) or the exploring one. Greedy is what
        you would deploy; stochastic is what generated the training data.

    Returns
    -------
    RlEvalResult
        For bandits: ``direct_method``, ``ips``, ``action_match_rate``, and the
        logged-reward baseline, with ``offline=True``. For environment modes:
        mean and standard deviation of episode return, with ``offline=False``.

    Raises
    ------
    ValidationError
        If the mode is unsupported, if a bandit plan lacks its dataset, split
        plan, action column, reward column, or context columns, if an
        environment plan lacks its ``env_id``, or if the stored policy does not
        match the declared mode.
    MissingExtraError
        If an environment rollout is requested without ``buildml[rl]`` or
        ``buildml[rl-industry]``.

    Notes
    -----
    **Compare every bandit estimate against ``mean_logged_reward``.** A direct
    method of 14.2 means nothing on its own; against a logged baseline of 12.7
    it suggests an improvement, and against 15.1 it suggests the new policy is
    worse than what you already run.

    **``ips`` is ``NaN`` when no propensity model could be fitted**, and a
    warning says so. That leaves the direct method unchecked, which is exactly
    the situation where a single estimate should not be trusted alone.

    **Tabular evaluation reports ``unseen_state_rate``.** A tabular policy has no
    way to generalise to a state it never visited, so its action there comes
    from an untouched Q-row — effectively arbitrary. Above 20% a warning is
    raised, because the mean return is then substantially a measure of luck.

    Examples
    --------
    >>> result = evaluate_rl(dataset, plan, split_plan)  # doctest: +SKIP
    >>> result.offline, result.metrics["action_match_rate"]  # doctest: +SKIP
    (True, 0.34)

    See Also
    --------
    buildml.rl.act.act_rl : Run the policy rather than score it.
    """
    if plan.mode == "gym_reinforce":
        return _eval_gym(
            plan,
            n_episodes=n_episodes,
            max_steps=max_steps,
            random_state=random_state,
            deterministic=deterministic,
        )
    if plan.mode == "tabular_q":
        return _eval_tabular(
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


def _eval_tabular(
    plan: RlPlan,
    *,
    n_episodes: int | None,
    max_steps: int | None,
    random_state: int | None,
    deterministic: bool,
) -> RlEvalResult:
    policy = plan.policy_
    if not isinstance(policy, TabularValuePolicy):
        raise ValidationError("tabular_q plan is missing a TabularValuePolicy.")
    if plan.env_id is None:
        raise ValidationError("tabular_q plan is missing env_id.")
    cfg = plan.config or {}
    metrics = evaluate_tabular_policy(
        policy,
        env_id=plan.env_id,
        n_episodes=int(n_episodes if n_episodes is not None else 20),
        max_steps=int(max_steps if max_steps is not None else cfg.get("max_steps", 200)),
        random_state=random_state,
        deterministic=deterministic,
    )
    warnings: list[str] = []
    unseen = float(metrics.get("unseen_state_rate", float("nan")))
    if np.isfinite(unseen) and unseen > 0.2:
        warnings.append(
            f"{unseen:.0%} of evaluation steps landed in states never visited "
            "during training; those actions come from an untrained Q-row."
        )
    return RlEvalResult(
        partition=None,
        mode=plan.mode,
        n_rows=int(metrics.get("n_eval_episodes", 0)),
        metrics=metrics,
        offline=False,
        disclosures=(
            "Tabular evaluation rolls out the greedy Q-table policy in the env "
            "(online returns).",
            "Requires buildml[rl] (gymnasium).",
            "unseen_state_rate reports how often evaluation reached a state the "
            "Q-table never updated — the honest generalization limit of tabular RL.",
            "Honesty: small discrete-control teaching loop — not MuJoCo/robotics.",
        ),
        warnings=tuple(warnings),
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
