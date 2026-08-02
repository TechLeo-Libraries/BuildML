"""Thin Session facades over buildml.metalearning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.metalearning.adapt import adapt_to_task
from buildml.metalearning.checkpoint import (
    load_metalearning_bundle,
    save_metalearning_bundle,
)
from buildml.metalearning.evaluate import evaluate_metalearning
from buildml.metalearning.explain_hooks import (
    adapt_result_summary,
    eval_result_summary,
    fit_result_summary,
)
from buildml.metalearning.fit import fit_metalearning
from buildml.metalearning.types import MetaLearningBaseEstimator, MetaLearningMethod

PartitionOrAll = PartitionName | Literal["all"]


def fit_metalearning_op(
    session,
    *,
    method: MetaLearningMethod = "prototypical",
    task_column: str | None = None,
    columns: list[str] | None = None,
    n_way: int | None = None,
    k_shot: int = 5,
    n_query: int = 10,
    n_episodes: int = 20,
    base_estimator: MetaLearningBaseEstimator | str = "logistic_regression",
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    task_holdout_fraction: float = 0.25,
) -> Any:
    """Meta-train on episodic tasks carved from the train partition only.

    Notes
    -----
    **Leakage:** Requires a split. Meta-train uses train only. Validation/test
    are never used for meta-training. Needs a task/group column (role or
    ``task_column=``) and exactly one ``role='target'``. Honesty: tabular
    few-shot / episodic protocols — not foundation-model meta-learning.
    """
    session.assert_can_fit("train")
    plan, result = fit_metalearning(
        session.dataset,
        session._split_plan,
        method=method,
        task_column=task_column,
        columns=columns,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        n_episodes=n_episodes,
        base_estimator=base_estimator,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        task_holdout_fraction=task_holdout_fraction,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._metalearning_plan = plan
    session._metalearning_fit_result = result
    session._metalearning_adapt_result = None
    session._metalearning_eval_result = None
    session._record(
        "fit_metalearning",
        {
            "method": method,
            "task_column": task_column,
            "columns": columns,
            "n_way": n_way,
            "k_shot": k_shot,
            "n_query": n_query,
            "n_episodes": n_episodes,
            "base_estimator": base_estimator,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
            "task_holdout_fraction": task_holdout_fraction,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def adapt_to_task_op(
    session,
    *,
    task_id: Any | None = None,
    partition: PartitionName = "train",
    support_frame: pd.DataFrame | None = None,
    max_support_per_class: int | None = None,
    random_state: int | None = 0,
) -> Any:
    """Fast-adapt the meta-learner to one task's labeled support set."""
    plan = getattr(session, "_metalearning_plan", None)
    if plan is None:
        raise ValidationError(
            "No meta-learning plan. Call fit_metalearning(...) first."
        )
    result = adapt_to_task(
        session.dataset,
        plan,
        session._split_plan,
        task_id=task_id,
        partition=partition,
        support_frame=support_frame,
        max_support_per_class=max_support_per_class,
        random_state=random_state,
    )
    session._metalearning_adapt_result = result
    session._record(
        "adapt_to_task",
        {
            "task_id": task_id,
            "partition": partition,
            "has_support_frame": support_frame is not None,
            "max_support_per_class": max_support_per_class,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=adapt_result_summary(result),
    )
    return result


def evaluate_metalearning_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    k_shot: int | None = None,
    n_query: int | None = None,
    n_way: int | None = None,
    prefer_novel_tasks: bool = True,
    random_state: int | None = 0,
) -> Any:
    """Episodic holdout evaluation (never for meta-train)."""
    plan = getattr(session, "_metalearning_plan", None)
    if plan is None:
        raise ValidationError(
            "No meta-learning plan. Call fit_metalearning(...) first."
        )
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_metalearning(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        k_shot=k_shot,
        n_query=n_query,
        n_way=n_way,
        prefer_novel_tasks=prefer_novel_tasks,
        random_state=random_state,
    )
    session._metalearning_eval_result = result
    session._record(
        "evaluate_metalearning",
        {
            "partition": resolved,
            "k_shot": k_shot,
            "n_query": n_query,
            "n_way": n_way,
            "prefer_novel_tasks": prefer_novel_tasks,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_metalearning_bundle_op(session, path: str | Path) -> Path:
    """Persist the active MetaLearningPlan as ``buildml.metalearning_bundle.v1``."""
    plan = getattr(session, "_metalearning_plan", None)
    if plan is None:
        raise ValidationError(
            "No meta-learning plan. Call fit_metalearning(...) first."
        )
    out = save_metalearning_bundle(
        path,
        plan,
        fit_result=getattr(session, "_metalearning_fit_result", None),
        eval_result=getattr(session, "_metalearning_eval_result", None),
        adapt_result=getattr(session, "_metalearning_adapt_result", None),
    )
    session._record(
        "save_metalearning_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "method": plan.method,
            "task_column": plan.task_column,
            "n_meta_train_tasks": len(plan.train_task_ids),
        },
    )
    return out


def load_metalearning_bundle_op(session, path: str | Path) -> Any:
    """Load a meta-learning bundle into this Session."""
    plan = load_metalearning_bundle(path)
    session._metalearning_plan = plan
    session._metalearning_fit_result = None
    session._metalearning_adapt_result = None
    session._metalearning_eval_result = None
    session._record(
        "load_metalearning_bundle",
        {
            "path": str(path),
            "method": plan.method,
            "task_column": plan.task_column,
            "n_meta_train_tasks": len(plan.train_task_ids),
        },
        result_summary=plan.to_dict(),
    )
    return session
