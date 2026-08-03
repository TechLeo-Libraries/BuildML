"""Ask a fitted policy what to do.

Fitting produces a policy; this is where you use it. What you pass in depends on
the mode, and the difference is not arbitrary. A contextual bandit reads
situations off a table, so you give it a partition. An environment policy reads
observation vectors, which have no tabular equivalent, so you give it
observations directly.

Every action comes back with the scores behind it, and reading those is usually
more informative than reading the action. Four arms scoring 0.51, 0.50, 0.50,
0.49 mean the policy has essentially no preference, which the chosen action
alone would not tell you.

The ``deterministic`` flag is the one real decision here. Deterministic acting
always takes the highest-scoring action and is what you want when serving:
reproducible, and it exploits what has been learned. Stochastic acting samples,
which keeps exploring — necessary if the log you collect will be used to train
the next policy, because a deterministic policy generates data about one action
per context and nothing about the alternatives.

See Also
--------
buildml.rl.fit : Producing the policy.
buildml.rl.evaluate : Scoring the policy rather than running it.
"""

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
from buildml.rl.tabular import TabularValuePolicy, act_tabular_observation

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
    """Choose an action for each situation you give the policy.

    Dispatches on the plan's mode. Bandit plans read situations from a dataset
    partition; environment plans read observation vectors from
    ``observations``.

    Parameters
    ----------
    dataset:
        The table of situations, for ``'contextual_bandit'``. Not used by the
        environment modes.
    plan:
        The fitted policy from :func:`~buildml.rl.fit.fit_rl`.
    split_plan:
        Required when a bandit plan is scoped to a partition other than
        ``'all'``.
    partition:
        Which rows to act on, for bandit plans.
    observations:
        Observation vectors for the environment modes: a 2-D array of rows, or
        a single 1-D vector, which is treated as one observation. For a
        ``Discrete`` observation space under ``'tabular_q'``, pass the state
        indices themselves.
    deterministic:
        ``True`` (default) always takes the best-scoring action. ``False``
        samples, which keeps exploring — use it when the resulting log will
        train the next policy.
    random_state:
        Seed for stochastic acting. Ignored when ``deterministic`` is ``True``.

    Returns
    -------
    RlActResult
        The chosen actions plus their per-action scores. Bandit actions come
        back as the original action labels rather than internal codes.

    Raises
    ------
    ValidationError
        If the mode is unsupported, if a bandit plan is given no dataset, if a
        partition is requested without a split plan, if an environment plan is
        given no observations, or if the plan's stored policy does not match
        its declared mode.

    Notes
    -----
    **Scores mean different things per mode, and are not comparable across
    them.** LinUCB reports upper confidence bounds — an optimistic estimate,
    deliberately inflated for arms it has seen little of. Epsilon-greedy and
    softmax report predicted rewards. Policy-gradient modes report action
    probabilities summing to 1.0. Tabular control reports Q-values, which are
    discounted future-return estimates.

    An empty bandit partition returns an empty result rather than raising.

    Examples
    --------
    >>> result = act_rl(dataset, plan, split_plan, partition="test")  # doctest: +SKIP
    >>> result.actions[:3], result.scores[0]  # doctest: +SKIP
    (('offer_b', 'offer_a', 'offer_b'), (0.41, 0.62, 0.33, 0.29))

    See Also
    --------
    buildml.rl.evaluate.evaluate_rl : What this policy is estimated to earn.
    """
    if plan.mode == "gym_reinforce":
        return _act_gym(
            plan,
            observations=observations,
            deterministic=deterministic,
            random_state=random_state,
        )
    if plan.mode == "tabular_q":
        return _act_tabular(
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


def _act_tabular(
    plan: RlPlan,
    *,
    observations: Sequence[Any] | np.ndarray | None,
    deterministic: bool,
    random_state: int | None,
) -> RlActResult:
    if observations is None:
        raise ValidationError(
            "tabular_q act_rl requires observations=... "
            "(state indices for Discrete envs, or observation vectors for Box envs)."
        )
    policy = plan.policy_
    if not isinstance(policy, TabularValuePolicy):
        raise ValidationError("tabular_q plan is missing a TabularValuePolicy.")
    arr = np.asarray(observations, dtype=float)
    if policy.discretizer.kind == "discrete":
        rows: list[Any] = [float(v) for v in arr.reshape(-1).tolist()]
    else:
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows = list(arr)
    actions: list[Any] = []
    scores: list[tuple[float, ...]] = []
    for i, obs in enumerate(rows):
        action, q_values = act_tabular_observation(
            policy,
            obs,
            random_state=None if random_state is None else int(random_state) + i,
            deterministic=deterministic,
        )
        actions.append(action)
        scores.append(q_values)
    return RlActResult(
        partition=None,
        mode=plan.mode,
        n_rows=len(actions),
        actions=tuple(actions),
        scores=tuple(scores),
        disclosures=(
            f"Actions chosen greedily from a tabular {policy.algorithm} Q-table."
            if deterministic
            else f"Actions sampled epsilon-greedily from a tabular "
            f"{policy.algorithm} Q-table.",
            "Scores are Q(s, a) action values for the discretized state.",
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
