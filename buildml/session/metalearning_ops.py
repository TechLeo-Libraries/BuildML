"""Thin Session facades over buildml.metalearning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
from buildml.metalearning.types import MetaLearningBaseEstimator

PartitionOrAll = PartitionName | Literal["all"]


def fit_metalearning_op(
    session,
    *,
    backend: str | None = None,
    method: str = "prototypical",
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
    meta_epochs: int = 40,
    inner_lr: float = 0.05,
    inner_steps: int = 5,
    meta_lr: float = 1e-3,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> Any:
    """Meta-train on episodic tasks carved from the train partition only.

    Delegates to :func:`buildml.metalearning.fit.fit_metalearning`, stores the
    :class:`~buildml.metalearning.results.MetaLearningPlan` on Session, and
    records the fit. Follow with :func:`adapt_to_task_op` or
    :func:`evaluate_metalearning_op`.

    Parameters
    ----------
    session:
        Active Session with task/group column and split plan.
    backend:
        Optional backend override (``native`` or ``torch``).
    method:
        Meta-learning method (``prototypical``, ``maml``, etc.).
    task_column:
        Column identifying tasks/episodes; inferred from roles when omitted.
    columns:
        Explicit feature columns for episodic sampling.
    n_way:
        Classes per episode; inferred from data when ``None``.
    k_shot:
        Support examples per class in each episode.
    n_query:
        Query examples per class in each episode.
    n_episodes:
        Number of meta-training episodes per epoch.
    base_estimator:
        Fallback sklearn estimator for non-torch backends.
    random_state:
        Seed for episode sampling and initialization.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    task_holdout_fraction:
        Fraction of train tasks held out during meta-training.
    meta_epochs:
        Number of meta-training epochs (torch backends).
    inner_lr:
        Inner-loop learning rate for MAML-style methods.
    inner_steps:
        Inner-loop gradient steps per episode.
    meta_lr:
        Outer/meta learning rate.
    embed_dim:
        Embedding dimension for torch encoders.
    hidden_dim:
        Hidden layer width for torch encoders.
    device:
        Torch device string (``cpu`` or ``cuda``).

    Returns
    -------
    MetaLearningFitResult
        Serializable fit summary including task counts and disclosures.

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
        backend=backend,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
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
        meta_epochs=meta_epochs,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        meta_lr=meta_lr,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        device=device,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._metalearning_plan = plan
    session._metalearning_fit_result = result
    session._metalearning_adapt_result = None
    session._metalearning_eval_result = None
    session._record(
        "fit_metalearning",
        {
            "backend": backend,
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
            "meta_epochs": meta_epochs,
            "inner_lr": inner_lr,
            "inner_steps": inner_steps,
            "meta_lr": meta_lr,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "device": device,
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
    """Fast-adapt the meta-learner to one task's labeled support set.

    Delegates to :func:`buildml.metalearning.adapt.adapt_to_task` using the
    plan from :func:`fit_metalearning_op`. No meta-training occurs here.

    Parameters
    ----------
    session:
        Active Session with a MetaLearningPlan from :func:`fit_metalearning_op`.
    task_id:
        Task identifier to adapt to; required when multiple tasks exist.
    partition:
        Partition containing support labels (default ``train``).
    support_frame:
        Optional explicit support DataFrame instead of a partition slice.
    max_support_per_class:
        Cap on support rows sampled per class.
    random_state:
        Seed for support sampling.

    Returns
    -------
    MetaLearningAdaptResult
        Adapted predictions and support-set summary for the task.

    Raises
    ------
    ValidationError
        When no meta-learning plan exists on the Session.
    """
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
    """Run episodic holdout evaluation without meta-training on holdout.

    Delegates to :func:`buildml.metalearning.evaluate.evaluate_metalearning`.
    Falls back to ``test`` when no validation partition exists.

    Parameters
    ----------
    session:
        Active Session with a MetaLearningPlan from :func:`fit_metalearning_op`.
    partition:
        Holdout partition for episodic evaluation (default ``validation``).
    k_shot:
        Support examples per class override for evaluation episodes.
    n_query:
        Query examples per class override for evaluation episodes.
    n_way:
        Classes per episode override.
    prefer_novel_tasks:
        When True, prefer tasks not seen during meta-training.
    random_state:
        Seed for episode construction.

    Returns
    -------
    MetaLearningEvalResult
        Episodic accuracy metrics on the holdout partition.

    Raises
    ------
    ValidationError
        When no meta-learning plan exists on the Session.
    """
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
    """Persist the active MetaLearningPlan as ``buildml.metalearning_bundle.v1``.

    Delegates to :func:`buildml.metalearning.checkpoint.save_metalearning_bundle`.
    Reload with :func:`load_metalearning_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a MetaLearningPlan from :func:`fit_metalearning_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no meta-learning plan exists on the Session.
    """
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


def load_metalearning_bundle_op(session, path: str | Path, *, trusted: bool = False) -> Any:
    """Load a meta-learning bundle into this Session.

    Delegates to :func:`buildml.metalearning.checkpoint.load_metalearning_bundle`
    and clears prior adapt/eval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded MetaLearningPlan.
    path:
        Path to a ``buildml.metalearning_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with MetaLearningPlan attached for chaining.
    """
    plan = load_metalearning_bundle(path, trusted=trusted)
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
    return cast("Session", session)