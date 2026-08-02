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
    """Predict actions under the fitted BC policy."""
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
    """Evaluate BC against held-out demonstration actions."""
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
) -> Any:
    """Fit a contextual bandit (core) or Gymnasium REINFORCE-lite (``buildml[rl]``).

    Notes
    -----
    **Leakage (bandit):** Requires a split; updates use train logged data only.
    **gym_reinforce / gym_sb3:** Env loop; does not fit on Session tabular partitions.
    Honesty: not MuJoCo / robotics / multi-agent.
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
            random_state=random_state,
            total_timesteps=total_timesteps,
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
    """Choose actions under the fitted RL policy."""
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
    """Evaluate RL (offline bandit metrics or Gymnasium rollouts)."""
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
