"""Thin Session facades over buildml.selfsupervised."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.selfsupervised.checkpoint import load_ssl_bundle, save_ssl_bundle
from buildml.selfsupervised.evaluate import evaluate_ssl
from buildml.selfsupervised.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    head_fit_result_summary,
    transform_result_summary,
)
from buildml.selfsupervised.finetune import finetune_ssl_head
from buildml.selfsupervised.fit import fit_ssl_pretext
from buildml.selfsupervised.transform import transform_ssl
from buildml.selfsupervised.types import SelfSupervisedMethod, SSLHeadEstimator

PartitionOrAll = PartitionName | Literal["all"]


def fit_ssl_pretext_op(
    session,
    *,
    method: SelfSupervisedMethod = "masked_tabular",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    latent_dim: int = 16,
    hidden: tuple[int, ...] | list[int] = (64,),
    mask_ratio: float = 0.15,
    n_mask_views: int = 3,
    max_iter: int = 200,
    prefer_reduce_components: bool = True,
    representation_prefix: str = "ssl_emb",
) -> Any:
    """Fit a self-supervised pretext encoder on the train partition only."""
    session.assert_can_fit("train")
    plan, result = fit_ssl_pretext(
        session.dataset,
        session._split_plan,
        method=method,
        columns=columns,
        random_state=random_state,
        latent_dim=latent_dim,
        hidden=hidden,
        mask_ratio=mask_ratio,
        n_mask_views=n_mask_views,
        max_iter=max_iter,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
        representation_prefix=representation_prefix,
    )
    session._ssl_plan = plan
    session._ssl_fit_result = result
    session._ssl_transform_result = None
    session._ssl_head_plan = None
    session._ssl_head_fit_result = None
    session._ssl_eval_result = None
    session._record(
        "fit_ssl_pretext",
        {
            "method": method,
            "columns": columns,
            "latent_dim": latent_dim,
            "hidden": list(hidden),
            "mask_ratio": mask_ratio,
            "n_mask_views": n_mask_views,
            "max_iter": max_iter,
            "prefer_reduce_components": prefer_reduce_components,
            "representation_prefix": representation_prefix,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def transform_ssl_op(
    session,
    *,
    partition: PartitionOrAll = "train",
    attach: bool = False,
) -> Any:
    """Export SSL representations with the train-fitted pretext (no refit)."""
    plan = getattr(session, "_ssl_plan", None)
    if plan is None:
        raise ValidationError("No SSL plan. Call fit_ssl_pretext(...) first.")
    new_dataset, result, _emb = transform_ssl(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        attach=attach,
    )
    if new_dataset is not None:
        session._dataset = new_dataset
    session._ssl_transform_result = result
    session._record(
        "transform_ssl",
        {"partition": partition, "attach": attach},
        result_summary=transform_result_summary(result),
    )
    return result


def finetune_ssl_head_op(
    session,
    *,
    estimator: SSLHeadEstimator = "logistic_regression",
    random_state: int | None = 0,
    unlabeled_marker: Any = None,
) -> Any:
    """Fit a supervised head on frozen SSL embeddings (labeled train only)."""
    plan = getattr(session, "_ssl_plan", None)
    if plan is None:
        raise ValidationError("No SSL plan. Call fit_ssl_pretext(...) first.")
    session.assert_can_fit("train")
    head_plan, result = finetune_ssl_head(
        session.dataset,
        session._split_plan,
        plan,
        estimator=estimator,
        random_state=random_state,
        unlabeled_marker=unlabeled_marker,
    )
    session._ssl_head_plan = head_plan
    session._ssl_head_fit_result = result
    session._ssl_eval_result = None
    session._record(
        "finetune_ssl_head",
        {
            "estimator": estimator,
            "random_state": random_state,
            "unlabeled_marker": unlabeled_marker,
        },
        warnings=tuple(result.warnings),
        result_summary=head_fit_result_summary(result),
    )
    return result


def evaluate_ssl_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    unlabeled_marker: Any = None,
) -> Any:
    """Evaluate frozen SSL pretext + head on labeled partition rows."""
    ssl_plan = getattr(session, "_ssl_plan", None)
    head_plan = getattr(session, "_ssl_head_plan", None)
    if ssl_plan is None or head_plan is None:
        raise ValidationError(
            "evaluate_ssl requires fit_ssl_pretext(...) and finetune_ssl_head(...)."
        )
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_ssl(
        session.dataset,
        ssl_plan,
        head_plan,
        session._split_plan,
        partition=resolved,
        unlabeled_marker=unlabeled_marker,
    )
    session._ssl_eval_result = result
    session._record(
        "evaluate_ssl",
        {"partition": resolved, "unlabeled_marker": unlabeled_marker},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_ssl_bundle_op(session, path: str | Path) -> Path:
    """Persist the active SSL plan (+ optional head) as ``buildml.selfsupervised_bundle.v1``."""
    plan = getattr(session, "_ssl_plan", None)
    if plan is None:
        raise ValidationError("No SSL plan. Call fit_ssl_pretext(...) first.")
    out = save_ssl_bundle(
        path,
        plan,
        fit_result=getattr(session, "_ssl_fit_result", None),
        head_plan=getattr(session, "_ssl_head_plan", None),
        head_fit_result=getattr(session, "_ssl_head_fit_result", None),
        eval_result=getattr(session, "_ssl_eval_result", None),
    )
    session._record(
        "save_ssl_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "method": plan.method,
            "latent_dim": plan.latent_dim,
            "has_head": getattr(session, "_ssl_head_plan", None) is not None,
        },
    )
    return out


def load_ssl_bundle_op(session, path: str | Path) -> Any:
    """Load a self-supervised bundle into this Session."""
    plan, head = load_ssl_bundle(path)
    session._ssl_plan = plan
    session._ssl_head_plan = head
    session._ssl_fit_result = None
    session._ssl_transform_result = None
    session._ssl_head_fit_result = None
    session._ssl_eval_result = None
    session._record(
        "load_ssl_bundle",
        {
            "path": str(path),
            "method": plan.method,
            "has_head": head is not None,
        },
        result_summary=plan.to_dict(),
    )
    return session
