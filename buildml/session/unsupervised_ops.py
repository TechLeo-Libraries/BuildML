"""Thin Session facades over buildml.unsupervised."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.unsupervised.checkpoint import load_unsupervised_bundle, save_unsupervised_bundle
from buildml.unsupervised.cluster import assign_clusters, fit_clusterer
from buildml.unsupervised.evaluate import evaluate_clustering
from buildml.unsupervised.explain_hooks import (
    assign_result_summary,
    eval_result_summary,
    fit_result_summary,
)
from buildml.unsupervised.types import ClusterMethod

PartitionOrAll = PartitionName | Literal["all"]


def fit_clusters(
    session,
    *,
    method: ClusterMethod = "kmeans",
    n_clusters: int | None = 8,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    n_init: int | str = "auto",
    max_iter: int = 300,
    linkage: str = "ward",
    eps: float = 0.5,
    min_samples: int = 5,
    prefer_reduce_components: bool = True,
    label_column: str = "cluster_id",
) -> Any:
    """Fit a clusterer on the train partition only.

    Integrates with ``Session.reduce_dimensions``: when a ReducePlan's component
    columns are on the frame and ``prefer_reduce_components=True``, those
    components are used. Clustering does **not** refit PCA.

    Notes
    -----
    **Leakage:** Requires a split. Geometry is learned on train only.
    Scale numeric inputs first for distance-based methods. EDA IsolationForest
    / correlation-cluster screens are not this API.
    """
    session.assert_can_fit("train")
    plan, result = fit_clusterer(
        session.dataset,
        session._split_plan,
        method=method,
        n_clusters=n_clusters,
        columns=columns,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        linkage=linkage,
        eps=eps,
        min_samples=min_samples,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
        label_column=label_column,
    )
    session._cluster_plan = plan
    session._cluster_fit_result = result
    session._cluster_assign_result = None
    session._cluster_eval_result = None
    session._record(
        "fit_clusters",
        {
            "method": method,
            "n_clusters": n_clusters,
            "columns": columns,
            "prefer_reduce_components": prefer_reduce_components,
            "label_column": label_column,
            "eps": eps,
            "min_samples": min_samples,
            "linkage": linkage,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def assign_clusters_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
) -> Any:
    """Assign cluster labels with the train-fitted plan (no refit)."""
    plan = getattr(session, "_cluster_plan", None)
    if plan is None:
        raise ValidationError("No cluster plan. Call fit_clusters(...) first.")
    new_dataset, result = assign_clusters(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        attach=attach,
    )
    if new_dataset is not None:
        session._dataset = new_dataset
    session._cluster_assign_result = result
    session._record(
        "assign_clusters",
        {"partition": partition, "attach": attach},
        result_summary=assign_result_summary(result),
    )
    return result


def evaluate_clusters(
    session,
    *,
    partition: PartitionOrAll = "validation",
    external_label_column: str | None = None,
    sample_size: int | None = 2000,
    random_state: int | None = 0,
) -> Any:
    """Evaluate train-fitted clusters on a partition (internal + optional external)."""
    plan = getattr(session, "_cluster_plan", None)
    if plan is None:
        raise ValidationError("No cluster plan. Call fit_clusters(...) first.")
    # Prefer validation; fall back to test when validation was not carved.
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_clustering(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
        external_label_column=external_label_column,
        sample_size=sample_size,
        random_state=random_state,
    )
    session._cluster_eval_result = result
    session._record(
        "evaluate_clusters",
        {
            "partition": resolved,
            "external_label_column": external_label_column,
            "sample_size": sample_size,
        },
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_unsupervised_bundle_op(session, path: str | Path) -> Path:
    """Persist the active ClusterPlan as ``buildml.unsupervised_bundle.v1``."""
    plan = getattr(session, "_cluster_plan", None)
    if plan is None:
        raise ValidationError("No cluster plan. Call fit_clusters(...) first.")
    out = save_unsupervised_bundle(
        path,
        plan,
        fit_result=getattr(session, "_cluster_fit_result", None),
        eval_result=getattr(session, "_cluster_eval_result", None),
    )
    session._record(
        "save_unsupervised_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "method": plan.method, "n_clusters": plan.n_clusters},
    )
    return out


def load_unsupervised_bundle_op(session, path: str | Path) -> Any:
    """Load an unsupervised bundle into this Session."""
    plan = load_unsupervised_bundle(path)
    session._cluster_plan = plan
    session._cluster_fit_result = None
    session._cluster_assign_result = None
    session._cluster_eval_result = None
    session._record(
        "load_unsupervised_bundle",
        {"path": str(path), "method": plan.method, "n_clusters": plan.n_clusters},
        result_summary=plan.to_dict(),
    )
    return session
