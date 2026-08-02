"""Act / choose actions under a fitted RL policy."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.rl.adapters.stable_baselines3 import SB3PolicyWrapper, act_sb3_observation
from buildml.rl.bandit import LinUCBPolicy, RewardModelBandit
from buildml.rl.features import decode_discrete_actions, matrix_from_frame
from buildml.rl.gym_reinforce import LinearSoftmaxPolicy, act_gym_observation
from buildml.rl.results import RlActResult, RlPlan

PartitionOrAll = PartitionName | Literal["all"]


def act_rl(
    dataset: Dataset | None,
    plan: RlPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    observations: Sequence[Any] | np.ndarray | None = None,
    deterministic: bool = True,
    random_state: int | None = 0,
) -> RlActResult:
    """Choose actions for a tabular partition (bandit) or raw observations (gym)."""
    if plan.mode == "gym_reinforce":
        return _act_gym(
            plan,
            observations=observations,
            deterministic=deterministic,
            random_state=random_state,
        )
    if plan.mode == "gym_sb3":
        return _act_sb3(
            plan,
            observations=observations,
            deterministic=deterministic,
        )
    if plan.mode != "contextual_bandit":
        raise ValidationError(f"Unsupported RL mode for act_rl: {plan.mode!r}.")
    if dataset is None:
        raise ValidationError("contextual_bandit act_rl requires a Dataset.")
    return _act_bandit(
        dataset,
        plan,
        split_plan,
        partition=partition,
        deterministic=deterministic,
        random_state=random_state,
    )


def _act_bandit(
    dataset: Dataset,
    plan: RlPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll,
    deterministic: bool,
    random_state: int | None,
) -> RlActResult:
    if partition == "all":
        frame = dataset._ensure_pandas()
    else:
        if split_plan is None:
            raise ValidationError(
                "A SplitPlan is required for partition-scoped act_rl."
            )
        frame = frame_for_partition(dataset, split_plan, partition)  # type: ignore[arg-type]
    if frame.empty:
        return RlActResult(
            partition=str(partition),
            mode=plan.mode,
            n_rows=0,
            actions=(),
            disclosures=("Empty partition; no actions chosen.",),
        )
    x = matrix_from_frame(frame, list(plan.columns))
    rng = np.random.default_rng(random_state)
    policy = plan.policy_
    arm_codes: list[int] = []
    score_rows: list[tuple[float, ...]] = []
    for i in range(x.shape[0]):
        row = x[i]
        if isinstance(policy, LinUCBPolicy):
            scores = policy.scores(row)
            arm = int(np.argmax(scores)) if deterministic else policy.select(row, rng=rng)
        elif isinstance(policy, RewardModelBandit):
            scores = policy.scores_row(row)
            if deterministic:
                arm = int(np.argmax(scores))
            else:
                arm = policy.select(row, rng=rng)
        else:
            raise ValidationError(
                f"Unsupported bandit policy type: {type(policy)!r}."
            )
        arm_codes.append(arm)
        score_rows.append(tuple(float(s) for s in np.asarray(scores).tolist()))
    actions = tuple(
        decode_discrete_actions(np.asarray(arm_codes), plan.label_encoder_)
    )
    return RlActResult(
        partition=str(partition),
        mode=plan.mode,
        n_rows=int(x.shape[0]),
        actions=actions,
        scores=tuple(score_rows),
        disclosures=(
            "Actions chosen by a train-fitted contextual bandit policy.",
            "Scores are UCB values (linucb) or predicted rewards (epsilon_greedy/softmax).",
        ),
    )


def _act_gym(
    plan: RlPlan,
    *,
    observations: Sequence[Any] | np.ndarray | None,
    deterministic: bool,
    random_state: int | None,
) -> RlActResult:
    if observations is None:
        raise ValidationError(
            "gym_reinforce act_rl requires observations=... "
            "(array or sequence of observation vectors)."
        )
    policy = plan.policy_
    if not isinstance(policy, LinearSoftmaxPolicy):
        raise ValidationError("gym_reinforce plan is missing a LinearSoftmaxPolicy.")
    arr = np.asarray(observations, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    actions: list[Any] = []
    scores: list[tuple[float, ...]] = []
    for i, obs in enumerate(arr):
        action, probs = act_gym_observation(
            policy,
            obs,
            random_state=None if random_state is None else int(random_state) + i,
            deterministic=deterministic,
        )
        actions.append(action)
        scores.append(probs)
    return RlActResult(
        partition=None,
        mode=plan.mode,
        n_rows=len(actions),
        actions=tuple(actions),
        scores=tuple(scores),
        disclosures=(
            "Actions chosen by a Gymnasium REINFORCE-lite linear softmax policy.",
        ),
    )


def _act_sb3(
    plan: RlPlan,
    *,
    observations: Sequence[Any] | np.ndarray | None,
    deterministic: bool,
) -> RlActResult:
    if observations is None:
        raise ValidationError(
            "gym_sb3 act_rl requires observations=... "
            "(array or sequence of observation vectors)."
        )
    policy = plan.policy_
    if not isinstance(policy, SB3PolicyWrapper):
        raise ValidationError("gym_sb3 plan is missing an SB3PolicyWrapper.")
    arr = np.asarray(observations, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    actions: list[Any] = []
    scores: list[tuple[float, ...]] = []
    for obs in arr:
        action, probs = act_sb3_observation(
            policy,
            obs,
            deterministic=deterministic,
        )
        actions.append(action)
        scores.append(probs)
    return RlActResult(
        partition=None,
        mode=plan.mode,
        n_rows=len(actions),
        actions=tuple(actions),
        scores=tuple(scores),
        disclosures=(
            "Actions chosen by a Stable-Baselines3 industry policy.",
            "Honesty: small-env teaching loop — not MuJoCo/robotics.",
        ),
    )
