"""Train-fitted clustering with leakage-safe assign on holdout partitions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import (
    PartitionName,
    SplitPlan,
    assert_fit_partition,
    frame_for_partition,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.unsupervised.backends import fit_backend, predict_backend
from buildml.unsupervised.features import matrix_from_frame, resolve_cluster_columns
from buildml.unsupervised.results import ClusterAssignResult, ClusterFitResult, ClusterPlan
from buildml.unsupervised.types import ClusterConfig, ClusterMethod

PartitionOrAll = PartitionName | Literal["all"]


def fit_clusterer(
    dataset: Dataset,
    split_plan: SplitPlan | None,
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
    reduce_plan: Any | None = None,
    label_column: str = "cluster_id",
    auto_k: bool = False,
    auto_k_min: int = 2,
    auto_k_max: int = 10,
) -> tuple[ClusterPlan, ClusterFitResult]:
    """Fit a clusterer on the train partition only.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.
method:
    Method or strategy identifier for the resolved backend.
n_clusters:
    Target number of clusters for partitioning.
columns:
    Optional explicit feature column list; ``None`` auto-selects numerics.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
n_init:
    n init (int | str).
max_iter:
    max iter (int).
linkage:
    linkage (str).
eps:
    Neighborhood radius for density-based clustering.
min_samples:
    Minimum neighborhood size for density-based methods.
gmm_covariance_type:
    gmm covariance type (str).
gmm_max_components:
    gmm max components (int).
gmm_select_by:
    gmm select by (str).
hdbscan_min_cluster_size:
    hdbscan min cluster size (int).
hdbscan_min_samples:
    hdbscan min samples (int | None).
spectral_affinity:
    spectral affinity (str).
spectral_n_neighbors:
    spectral n neighbors (int).
optics_min_samples:
    optics min samples (int).
optics_xi:
    optics xi (float).
optics_min_cluster_size:
    optics min cluster size (float | None).
bandwidth:
    bandwidth (float | None).
latent_dim:
    latent dim (int).
pretrain_epochs:
    pretrain epochs (int).
finetune_epochs:
    finetune epochs (int).
batch_size:
    Number of rows to select per query or training minibatch.
learning_rate:
    Optimizer learning rate for torch training.
prefer_reduce_components:
    Prefer reduced component columns when a reduce plan exists.
reduce_plan:
    Optional preprocess reduce plan from Session.
label_column:
    label column (str).
auto_k:
    auto k (bool).
auto_k_min:
    auto k min (int).
auto_k_max:
    auto k max (int).

Returns
-------
tuple[ClusterPlan, ClusterFitResult]
    Tuple of results (tuple[ClusterPlan, ClusterFitResult]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if not label_column or not str(label_column).replace("_", "").isalnum():
        raise ValidationError("label_column must be a non-empty alphanumeric token")

    train = frame_for_partition(dataset, split_plan, "train")
    cols, used_reduce, disclosures = resolve_cluster_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
    )
    x = matrix_from_frame(train, cols)
    n_train = int(x.shape[0])

    config = ClusterConfig(
        method=method,
        n_clusters=n_clusters,
        columns=tuple(cols),
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
        label_column=label_column,
        auto_k=auto_k,
        auto_k_min=auto_k_min,
        auto_k_max=auto_k_max,
    )
    outcome = fit_backend(x, config, n_train=n_train)
    disclosures = list(disclosures) + list(outcome.disclosures)

    label_list = [int(v) for v in np.asarray(outcome.labels).tolist()]
    sizes = {int(k): int(v) for k, v in sorted(Counter(label_list).items())}
    plan = ClusterPlan(
        method=method,
        columns=tuple(cols),
        label_column=label_column,
        n_clusters=outcome.n_clusters,
        n_train_rows=n_train,
        train_labels_=tuple(label_list),
        cluster_sizes_=sizes,
        assign_strategy=outcome.assign_strategy,
        estimator_=outcome.estimator,
        centroids_=outcome.centroids,
        centroid_label_ids_=outcome.centroid_ids,
        core_sample_indices_=outcome.core_idx,
        disclosures=tuple(disclosures),
        warnings=tuple(outcome.warnings),
        used_reduce_components=used_reduce,
        config={**config.to_dict(), **outcome.extra},
    )
    result = ClusterFitResult(
        method=method,
        n_clusters=outcome.n_clusters,
        n_train_rows=n_train,
        columns=tuple(cols),
        cluster_sizes=sizes,
        assign_strategy=outcome.assign_strategy,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(outcome.warnings),
        inertia=outcome.inertia,
        diagnostics=dict(outcome.extra),
    )
    return plan, result


def assign_clusters(
    dataset: Dataset,
    plan: ClusterPlan,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    attach: bool = False,
) -> tuple[Dataset | None, ClusterAssignResult]:
    """Assign cluster labels using a train-fitted plan (no refit).

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
plan:
    Fitted plan object carrying model state and feature contract.
split_plan:
    Train/validation/test split; fit uses train partition only.
partition:
    Holdout partition name or ``all`` for the full frame.
attach:
    attach (bool).

Returns
-------
tuple[Dataset | None, ClusterAssignResult]
    Tuple of results (tuple[Dataset | None, ClusterAssignResult]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    frame, part_name = _frame_for_assign(dataset, split_plan, partition)
    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Cluster plan columns missing from dataset: {missing}")
    x = matrix_from_frame(frame, list(plan.columns))
    labels = predict_backend(plan, x)
    label_list = [int(v) for v in labels.tolist()]
    n_noise = sum(1 for v in label_list if v < 0)
    disclosures = list(plan.disclosures)
    if plan.assign_strategy != "native" and partition != "train":
        disclosures.append(
            f"Assigned partition='{part_name}' with strategy={plan.assign_strategy} "
            "(frozen train geometry; not a refit)."
        )

    attached = False
    new_dataset: Dataset | None = None
    if attach:
        if partition != "all":
            raise ValidationError(
                "attach=True requires partition='all' so label columns stay aligned "
                "with the Session frame. Assign holdout labels without attach, or "
                "call assign_clusters(..., partition='all', attach=True)."
            )
        out = dataset._ensure_pandas().copy()
        if plan.label_column in out.columns:
            raise ValidationError(
                f"label_column '{plan.label_column}' already exists on the dataset"
            )
        out[plan.label_column] = np.asarray(label_list, dtype=int)
        roles = dict(dataset.roles)
        roles[plan.label_column] = ColumnRole.FEATURE
        new_dataset = Dataset.from_transformed(
            dataset,
            out,
            schema=schema_from_dataframe(out),
            roles=roles,
        )
        attached = True

    result = ClusterAssignResult(
        partition=part_name,
        labels=tuple(label_list),
        n_rows=len(label_list),
        label_column=plan.label_column,
        method=plan.method,
        assign_strategy=plan.assign_strategy,
        attached=attached,
        n_noise=n_noise,
        disclosures=tuple(dict.fromkeys(disclosures)),
    )
    return new_dataset, result


def _frame_for_assign(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    partition: PartitionOrAll,
) -> tuple[pd.DataFrame, str]:
    if partition == "all":
        return dataset._ensure_pandas(), "all"
    if split_plan is None:
        raise ValidationError(
            f"partition='{partition}' requires a SplitPlan. "
            "Call session.split(...) first, or use partition='all'."
        )
    return frame_for_partition(dataset, split_plan, partition), str(partition)
