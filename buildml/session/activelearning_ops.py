"""Thin Session facades over buildml.activelearning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.activelearning.checkpoint import (
    load_active_learning_bundle,
    save_active_learning_bundle,
)
from buildml.activelearning.evaluate import evaluate_active_learning
from buildml.activelearning.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    label_result_summary,
    query_result_summary,
)
from buildml.activelearning.fit import fit_active_learner
from buildml.activelearning.label import label_rows
from buildml.activelearning.query import suggest_query
from buildml.activelearning.types import (
    ActiveLearningBackend,
    ActiveLearningEstimator,
    ActiveLearningStrategy,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_active_learner_op(
    session,
    *,
    backend: ActiveLearningBackend | None = None,
    strategy: ActiveLearningStrategy = "margin",
    base_estimator: ActiveLearningEstimator = "logistic_regression",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    batch_size: int = 5,
    label_budget: int | None = 50,
    unlabeled_marker: Any = None,
    prefer_reduce_components: bool = True,
    committee_size: int = 5,
    auto_refit: bool = True,
    epochs: int = 60,
    learning_rate: float = 1e-3,
    mc_samples: int = 20,
    device: str = "cpu",
) -> Any:
    """Fit / initialize the active learner on labeled train rows only.

    Notes
    -----
    **Leakage:** Requires a split. Fit uses labeled train rows only. The
    unlabeled pool is train target missingness (NaN by default). Validation/test
    are never the query pool. Labels come from the user — no oracle in core.
    """
    session.assert_can_fit("train")
    prior = getattr(session, "_activelearning_plan", None)
    plan, result = fit_active_learner(
        session.dataset,
        session._split_plan,
        backend=backend,
        strategy=strategy,
        base_estimator=base_estimator,
        columns=columns,
        random_state=random_state,
        batch_size=batch_size,
        label_budget=label_budget,
        unlabeled_marker=unlabeled_marker,
        prefer_reduce_components=prefer_reduce_components,
        committee_size=committee_size,
        auto_refit=auto_refit,
        epochs=epochs,
        learning_rate=learning_rate,
        mc_samples=mc_samples,
        device=device,
        reduce_plan=getattr(session, "_reduce_plan", None),
        prior_plan=prior,
    )
    session._activelearning_plan = plan
    session._activelearning_fit_result = result
    session._activelearning_query_result = None
    session._activelearning_label_result = None
    session._activelearning_eval_result = None
    session._record(
        "fit_active_learner",
        {
            "backend": backend,
            "strategy": strategy,
            "base_estimator": base_estimator,
            "columns": columns,
            "random_state": random_state,
            "batch_size": batch_size,
            "label_budget": label_budget,
            "unlabeled_marker": unlabeled_marker,
            "prefer_reduce_components": prefer_reduce_components,
            "committee_size": committee_size,
            "auto_refit": auto_refit,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "mc_samples": mc_samples,
            "device": device,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def suggest_query_op(
    session,
    *,
    batch_size: int | None = None,
    strategy: ActiveLearningStrategy | None = None,
) -> Any:
    """Suggest unlabeled train-pool indices for human labeling (no oracle)."""
    plan = getattr(session, "_activelearning_plan", None)
    if plan is None:
        raise ValidationError("No active-learning plan. Call fit_active_learner(...) first.")
    result = suggest_query(
        session.dataset,
        plan,
        session._split_plan,
        batch_size=batch_size,
        strategy=strategy,
    )
    session._activelearning_query_result = result
    session._record(
        "suggest_query",
        {"batch_size": batch_size, "strategy": strategy},
        warnings=tuple(result.warnings),
        result_summary=query_result_summary(result),
    )
    return result


def label_rows_op(
    session,
    *,
    indices: Sequence[Any],
    labels: Sequence[Any],
    refit: bool | None = None,
) -> Any:
    """Incorporate user-provided labels on train-pool rows; optionally refit."""
    plan = getattr(session, "_activelearning_plan", None)
    if plan is None:
        raise ValidationError("No active-learning plan. Call fit_active_learner(...) first.")
    new_dataset, new_plan, result, fit_result = label_rows(
        session.dataset,
        plan,
        session._split_plan,
        indices=indices,
        labels=labels,
        refit=refit,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._dataset = new_dataset
    session._activelearning_plan = new_plan
    session._activelearning_label_result = result
    if fit_result is not None:
        session._activelearning_fit_result = fit_result
    session._record(
        "label_rows",
        {
            "n_indices": len(list(indices)),
            "indices": list(indices),
            "refit": result.refit,
        },
        warnings=tuple(result.warnings),
        result_summary=label_result_summary(result),
    )
    return result


def evaluate_active_learning_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    unlabeled_marker: Any = None,
) -> Any:
    """Evaluate the active learner on labeled rows of a partition."""
    plan = getattr(session, "_activelearning_plan", None)
    if plan is None:
        raise ValidationError("No active-learning plan. Call fit_active_learner(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_active_learning(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        unlabeled_marker=unlabeled_marker,
    )
    session._activelearning_eval_result = result
    session._record(
        "evaluate_active_learning",
        {"partition": resolved, "unlabeled_marker": unlabeled_marker},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_active_learning_bundle_op(session, path: str | Path) -> Path:
    """Persist the active ActiveLearningPlan as ``buildml.activelearning_bundle.v1``."""
    plan = getattr(session, "_activelearning_plan", None)
    if plan is None:
        raise ValidationError("No active-learning plan. Call fit_active_learner(...) first.")
    out = save_active_learning_bundle(
        path,
        plan,
        fit_result=getattr(session, "_activelearning_fit_result", None),
        eval_result=getattr(session, "_activelearning_eval_result", None),
    )
    session._record(
        "save_active_learning_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "strategy": plan.strategy,
            "n_labeled_train": plan.n_labeled_train,
            "n_unlabeled_pool": plan.n_unlabeled_pool,
            "n_queries_used": plan.n_queries_used,
        },
    )
    return out


def load_active_learning_bundle_op(session, path: str | Path) -> Any:
    """Load an active-learning bundle into this Session."""
    plan = load_active_learning_bundle(path)
    session._activelearning_plan = plan
    session._activelearning_fit_result = None
    session._activelearning_query_result = None
    session._activelearning_label_result = None
    session._activelearning_eval_result = None
    session._record(
        "load_active_learning_bundle",
        {"path": str(path), "strategy": plan.strategy},
        result_summary=plan.to_dict(),
    )
    return session
