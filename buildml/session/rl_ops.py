"""Thin Session facades over buildml.rl (imitation + reinforcement learning)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.rl.act import act_rl
from buildml.rl.checkpoint import (
    load_imitation_bundle,
    load_rl_bundle,
    save_imitation_bundle,
    save_rl_bundle,
)
from buildml.rl.evaluate import evaluate_rl
from buildml.rl.explain_hooks import (
    imitation_eval_summary,
    imitation_fit_summary,
    imitation_predict_summary,
    rl_act_summary,
    rl_eval_summary,
    rl_fit_summary,
)
from buildml.rl.catalog import rl_capability_matrix
from buildml.rl.fit import fit_rl
from buildml.rl.imitation import (
    evaluate_imitation,
    fit_imitation,
    predict_imitation_action,
)
from buildml.rl.types import (
    BanditAlgorithm,
    ImitationEstimator,
    ImitationTask,
    RlMode,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_imitation_op(
    session,
    *,
    backend: str | None = None,
    task: ImitationTask | None = None,
    estimator: ImitationEstimator | None = None,
    method: str | None = None,
    columns: list[str] | None = None,
    action_column: str | None = None,
    env_id: str | None = None,
    n_epochs: int = 40,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
) -> Any:
    """Fit behavioral cloning on Session train demonstrations.

    Delegates to :func:`buildml.rl.imitation.fit_imitation`, stores the
    :class:`~buildml.rl.results.ImitationPlan` on Session, and records the
    fit. Follow with :func:`predict_imitation_action_op` or
    :func:`evaluate_imitation_op`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    backend:
        Optional backend override (sklearn, torch).
    task:
        Optional task override (classification/regression).
    estimator:
        Optional BC estimator identifier.
    method:
        Optional method alias for the resolved backend.
    columns:
        Optional explicit state feature columns.
    action_column:
        Optional action column override.
    env_id:
        Optional Gymnasium environment id for env-backed demos.
    n_epochs:
        Training epochs for torch BC backend.
    random_state:
        Seed for stochastic training steps.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.

    Returns
    -------
    ImitationFitResult
        Serializable fit summary including action-space disclosures.

    Notes
    -----
    **Leakage:** Requires a split. Policy uses train only. Honesty: BC from
    tables — not inverse RL / DAgger / robotics.
    """
    session.assert_can_fit("train")
    plan, result = fit_imitation(
        session.dataset,
        session._split_plan,
        backend=backend,  # type: ignore[arg-type]
        task=task,
        estimator=estimator,
        method=method,  # type: ignore[arg-type]
        columns=columns,
        action_column=action_column,
        env_id=env_id,
        n_epochs=n_epochs,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._imitation_plan = plan
    session._imitation_fit_result = result
    session._imitation_eval_result = None
    session._imitation_predict_result = None
    session._record(
        "fit_imitation",
        {
            "backend": backend,
            "task": task,
            "estimator": estimator,
            "method": method,
            "columns": columns,
            "action_column": action_column,
            "env_id": env_id,
            "n_epochs": n_epochs,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
        },
        warnings=tuple(result.warnings),
        result_summary=imitation_fit_summary(result),
    )
    return result


def predict_imitation_action_op(
    session,
    *,
    partition: PartitionOrAll = "test",
) -> Any:
    """Predict actions under the fitted BC policy.

    Delegates to :func:`buildml.rl.imitation.predict_imitation_action` on a
    named partition without refitting the policy.

    Parameters
    ----------
    session:
        Active Session with an imitation plan from :func:`fit_imitation_op`.
    partition:
        Partition to predict on (``test`` by default).

    Returns
    -------
    ImitationPredictResult
        Predicted actions and optional quality disclosures.

    Raises
    ------
    ValidationError
        When no imitation plan exists on the Session.
    """
    plan = getattr(session, "_imitation_plan", None)
    if plan is None:
        raise ValidationError("No ImitationPlan. Call fit_imitation(...) first.")
    result = predict_imitation_action(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
    )
    session._imitation_predict_result = result
    session._record(
        "predict_imitation_action",
        {"partition": partition},
        warnings=tuple(result.warnings),
        result_summary=imitation_predict_summary(result),
    )
    return result


def evaluate_imitation_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
) -> Any:
    """Evaluate BC against held-out demonstration actions.

    Delegates to :func:`buildml.rl.imitation.evaluate_imitation` on a holdout
    partition. Falls back to ``test`` when validation is empty.

    Parameters
    ----------
    session:
        Active Session with an imitation plan from :func:`fit_imitation_op`.
    partition:
        Holdout partition for evaluation (``validation`` by default).

    Returns
    -------
    ImitationEvalResult
        Held-out action prediction metrics and disclosures.

    Raises
    ------
    ValidationError
        When no imitation plan exists on the Session.
    """
    plan = getattr(session, "_imitation_plan", None)
    if plan is None:
        raise ValidationError("No ImitationPlan. Call fit_imitation(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_imitation(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
    )
    session._imitation_eval_result = result
    session._record(
        "evaluate_imitation",
        {"partition": resolved},
        warnings=tuple(result.warnings),
        result_summary=imitation_eval_summary(result),
    )
    return result


def save_imitation_bundle_op(session, path: str | Path) -> Path:
    """Persist the active ImitationPlan as ``buildml.imitation_bundle.v1``.

    Delegates to :func:`buildml.rl.checkpoint.save_imitation_bundle`.
    Reload with :func:`load_imitation_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with an imitation plan from :func:`fit_imitation_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no imitation plan exists on the Session.
    """
    plan = getattr(session, "_imitation_plan", None)
    if plan is None:
        raise ValidationError("No ImitationPlan. Call fit_imitation(...) first.")
    out = save_imitation_bundle(
        path,
        plan,
        fit_result=getattr(session, "_imitation_fit_result", None),
        eval_result=getattr(session, "_imitation_eval_result", None),
    )
    session._record(
        "save_imitation_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "format": "buildml.imitation_bundle.v1"},
    )
    return out


def load_imitation_bundle_op(session, path: str | Path):
    """Load an imitation bundle into this Session.

    Delegates to :func:`buildml.rl.checkpoint.load_imitation_bundle` and
    clears prior eval/predict results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded imitation plan.
    path:
        Path to a ``buildml.imitation_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with imitation plan attached for chaining.
    """
    plan = load_imitation_bundle(path)
    session._imitation_plan = plan
    session._imitation_fit_result = None
    session._imitation_eval_result = None
    session._imitation_predict_result = None
    session._record(
        "load_imitation_bundle",
        {"path": str(path)},
        result_summary={
            "path": str(path),
            "kind": "imitation",
            "task": plan.task,
            "estimator": plan.estimator,
        },
    )
    return session


def fit_rl_op(
    session,
    *,
    backend: str | None = None,
    mode: RlMode | None = None,
    algorithm: BanditAlgorithm | str = "linucb",
    columns: list[str] | None = None,
    action_column: str | None = None,
    reward_column: str | None = None,
    alpha: float = 1.0,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    env_id: str = "CartPole-v1",
    n_episodes: int = 200,
    max_steps: int = 500,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    total_timesteps: int = 20_000,
    n_bins: int = 8,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
) -> Any:
    """Fit a contextual bandit (core) or a Gymnasium env policy (``buildml[rl]``).

    Delegates to :func:`buildml.rl.fit.fit_rl`, stores the
    :class:`~buildml.rl.results.RlPlan` on Session, and records the fit.
    Follow with :func:`act_rl_op` or :func:`evaluate_rl_op`.

    Parameters
    ----------
    session:
        Active Session with dataset attached (bandit mode) or env config.
    backend:
        Optional backend override.
    mode:
        RL mode (``contextual_bandit`` or gym-style modes).
    algorithm:
        Bandit or policy algorithm identifier.
    columns:
        Optional state feature columns for bandit mode.
    action_column:
        Logged action column for bandit mode.
    reward_column:
        Logged reward column for bandit mode.
    alpha:
        Exploration/strength parameter for LinUCB-style bandits.
    epsilon:
        Epsilon for epsilon-greedy exploration.
    temperature:
        Softmax temperature for stochastic action selection.
    random_state:
        Seed for stochastic training and exploration.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.
    env_id:
        Gymnasium environment id for env-loop modes.
    n_episodes:
        Number of training episodes for tabular/env modes.
    max_steps:
        Maximum steps per episode.
    learning_rate:
        Optimizer learning rate for policy updates.
    gamma:
        Discount factor for temporal-difference methods.
    total_timesteps:
        Total timesteps for SB3-style trainers.
    n_bins:
        Discretization bins for tabular Q-learning.
    epsilon_min:
        Minimum epsilon for decay schedules.
    epsilon_decay:
        Per-episode epsilon decay multiplier.

    Returns
    -------
    RlFitResult
        Serializable fit summary including mode and algorithm disclosures.

    Notes
    -----
    **Leakage (bandit):** Requires a split; updates use train logged data only.
    **gym_reinforce / tabular_q / gym_sb3:** Env loop; does not fit on Session
    tabular partitions. Honesty: not MuJoCo / robotics / multi-agent.
    """
    from buildml.rl.catalog import resolve_rl_backend_mode_algorithm

    _backend, resolved_mode, _algo = resolve_rl_backend_mode_algorithm(
        backend=backend,  # type: ignore[arg-type]
        mode=mode,
        algorithm=str(algorithm),
    )
    if resolved_mode == "contextual_bandit":
        session.assert_can_fit("train")
        plan, result = fit_rl(
            session.dataset,
            session._split_plan,
            backend=backend,  # type: ignore[arg-type]
            mode=mode,
            algorithm=algorithm,
            columns=columns,
            action_column=action_column,
            reward_column=reward_column,
            alpha=alpha,
            epsilon=epsilon,
            temperature=temperature,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            reduce_plan=getattr(session, "_reduce_plan", None),
        )
    else:
        plan, result = fit_rl(
            getattr(session, "_dataset", None),
            getattr(session, "_split_plan", None),
            backend=backend,  # type: ignore[arg-type]
            mode=mode,
            algorithm=algorithm,
            env_id=env_id,
            n_episodes=n_episodes,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            random_state=random_state,
            total_timesteps=total_timesteps,
            n_bins=n_bins,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
    session._rl_plan = plan
    session._rl_fit_result = result
    session._rl_eval_result = None
    session._rl_act_result = None
    session._record(
        "fit_rl",
        {
            "backend": backend,
            "mode": mode,
            "algorithm": algorithm,
            "columns": columns,
            "action_column": action_column,
            "reward_column": reward_column,
            "alpha": alpha,
            "epsilon": epsilon,
            "temperature": temperature,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
            "env_id": env_id,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "total_timesteps": total_timesteps,
            "n_bins": n_bins,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        },
        warnings=tuple(result.warnings),
        result_summary=rl_fit_summary(result),
    )
    return result


def act_rl_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    observations: Sequence[Any] | np.ndarray | None = None,
    deterministic: bool = True,
    random_state: int | None = 0,
) -> Any:
    """Choose actions under the fitted RL policy.

    Delegates to :func:`buildml.rl.act.act_rl` for bandit rows or env
    observations without refitting the policy.

    Parameters
    ----------
    session:
        Active Session with an RL plan from :func:`fit_rl_op`.
    partition:
        Partition for bandit action selection (``test`` by default).
    observations:
        Optional explicit observation batch for env/bandit modes.
    deterministic:
        When True, disable exploratory sampling where supported.
    random_state:
        Seed for stochastic action selection.

    Returns
    -------
    RlActResult
        Selected actions and policy disclosures.

    Raises
    ------
    ValidationError
        When no RL plan exists on the Session.
    """
    plan = getattr(session, "_rl_plan", None)
    if plan is None:
        raise ValidationError("No RlPlan. Call fit_rl(...) first.")
    result = act_rl(
        getattr(session, "_dataset", None),
        plan,
        getattr(session, "_split_plan", None),
        partition=partition,
        observations=observations,
        deterministic=deterministic,
        random_state=random_state,
    )
    session._rl_act_result = result
    session._record(
        "act_rl",
        {
            "partition": partition,
            "n_observations": None if observations is None else len(np.asarray(observations)),
            "deterministic": deterministic,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=rl_act_summary(result),
    )
    return result


def evaluate_rl_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    n_episodes: int | None = None,
    max_steps: int | None = None,
    random_state: int | None = 0,
    deterministic: bool = True,
) -> Any:
    """Evaluate RL (offline bandit metrics or Gymnasium rollouts).

    Delegates to :func:`buildml.rl.evaluate.evaluate_rl` on a holdout
    partition or env rollouts. Falls back to ``test`` for bandits when
    validation is empty.

    Parameters
    ----------
    session:
        Active Session with an RL plan from :func:`fit_rl_op`.
    partition:
        Holdout partition for bandit evaluation (``validation`` by default).
    n_episodes:
        Optional episode override for env evaluation.
    max_steps:
        Optional per-episode step cap for env evaluation.
    random_state:
        Seed for stochastic rollouts.
    deterministic:
        When True, disable exploratory sampling during evaluation.

    Returns
    -------
    RlEvalResult
        Offline or env evaluation metrics and disclosures.

    Raises
    ------
    ValidationError
        When no RL plan exists on the Session.
    """
    plan = getattr(session, "_rl_plan", None)
    if plan is None:
        raise ValidationError("No RlPlan. Call fit_rl(...) first.")
    resolved: PartitionOrAll = partition
    split = getattr(session, "_split_plan", None)
    if (
        plan.mode == "contextual_bandit"
        and partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_rl(
        getattr(session, "_dataset", None),
        plan,
        split,
        partition=resolved,
        n_episodes=n_episodes,
        max_steps=max_steps,
        random_state=random_state,
        deterministic=deterministic,
    )
    session._rl_eval_result = result
    session._record(
        "evaluate_rl",
        {
            "partition": resolved,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "random_state": random_state,
            "deterministic": deterministic,
        },
        warnings=tuple(result.warnings),
        result_summary=rl_eval_summary(result),
    )
    return result


def save_rl_bundle_op(session, path: str | Path) -> Path:
    """Persist the active RlPlan as ``buildml.rl_bundle.v1``.

    Delegates to :func:`buildml.rl.checkpoint.save_rl_bundle`.
    Reload with :func:`load_rl_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with an RL plan from :func:`fit_rl_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no RL plan exists on the Session.
    """
    plan = getattr(session, "_rl_plan", None)
    if plan is None:
        raise ValidationError("No RlPlan. Call fit_rl(...) first.")
    out = save_rl_bundle(
        path,
        plan,
        fit_result=getattr(session, "_rl_fit_result", None),
        eval_result=getattr(session, "_rl_eval_result", None),
    )
    session._record(
        "save_rl_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "format": "buildml.rl_bundle.v1"},
    )
    return out


def load_rl_bundle_op(session, path: str | Path):
    """Load an RL bundle into this Session.

    Delegates to :func:`buildml.rl.checkpoint.load_rl_bundle` and clears
    prior eval/act results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded RL plan.
    path:
        Path to a ``buildml.rl_bundle.v1`` directory.

    Returns
    -------
    Session
        ``session`` with RL plan attached for chaining.
    """
    plan = load_rl_bundle(path)
    session._rl_plan = plan
    session._rl_fit_result = None
    session._rl_eval_result = None
    session._rl_act_result = None
    session._record(
        "load_rl_bundle",
        {"path": str(path)},
        result_summary={
            "path": str(path),
            "kind": "rl",
            "mode": plan.mode,
            "algorithm": plan.algorithm,
        },
    )
    return session


def rl_capability_matrix_op() -> dict[str, Any]:
    """Return the RL / imitation capability matrix for this installation.

    Delegates to :func:`buildml.rl.catalog.rl_capability_matrix`. Use before
    :func:`fit_rl_op` or :func:`fit_imitation_op` to confirm available
    backends, modes, and algorithms for the current extras install.

    Returns
    -------
    dict
        Nested map of backend identifiers to supported modes and methods.
    """
    return rl_capability_matrix()
