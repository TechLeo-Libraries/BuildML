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
    gmm_covariance_type: str = "full",
    gmm_max_components: int = 10,
    gmm_select_by: str = "bic",
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int | None = None,
    spectral_affinity: str = "nearest_neighbors",
    spectral_n_neighbors: int = 10,
    optics_min_samples: int = 5,
    optics_xi: float = 0.05,
    optics_min_cluster_size: float | None = None,
    bandwidth: float | None = None,
    latent_dim: int = 10,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    prefer_reduce_components: bool = True,
    label_column: str = "cluster_id",
    auto_k: bool = False,
    auto_k_min: int = 2,
    auto_k_max: int = 10,
) -> Any:
    """Fit a clusterer on the train partition only."""
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
        gmm_covariance_type=gmm_covariance_type,
        gmm_max_components=gmm_max_components,
        gmm_select_by=gmm_select_by,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        spectral_affinity=spectral_affinity,
        spectral_n_neighbors=spectral_n_neighbors,
        optics_min_samples=optics_min_samples,
        optics_xi=optics_xi,
        optics_min_cluster_size=optics_min_cluster_size,
        bandwidth=bandwidth,
        latent_dim=latent_dim,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
        label_column=label_column,
        auto_k=auto_k,
        auto_k_min=auto_k_min,
        auto_k_max=auto_k_max,
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
            "auto_k": auto_k,
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
    compute_stability: bool = False,
    stability_runs: int = 10,
    stability_sample_fraction: float = 0.8,
    compute_elbow: bool = False,
    elbow_k_min: int = 2,
    elbow_k_max: int = 10,
) -> Any:
    """Evaluate train-fitted clusters on a partition (internal + optional external)."""
    plan = getattr(session, "_cluster_plan", None)
    if plan is None:
        raise ValidationError("No cluster plan. Call fit_clusters(...) first.")
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
        compute_stability=compute_stability,
        stability_runs=stability_runs,
        stability_sample_fraction=stability_sample_fraction,
        compute_elbow=compute_elbow,
        elbow_k_min=elbow_k_min,
        elbow_k_max=elbow_k_max,
    )
    session._cluster_eval_result = result
    session._record(
        "evaluate_clusters",
        {
            "partition": resolved,
            "external_label_column": external_label_column,
            "sample_size": sample_size,
            "compute_stability": compute_stability,
            "compute_elbow": compute_elbow,
        },
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def save_unsupervised_bundle_op(session, path: str | Path) -> Path:
    """Persist the active ClusterPlan as ``buildml.unsupervised_bundle.v2``."""
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
