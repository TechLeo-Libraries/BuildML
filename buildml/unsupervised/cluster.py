"""Train-fitted clustering with leakage-safe assign on holdout partitions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

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
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
    label_column: str = "cluster_id",
) -> tuple[ClusterPlan, ClusterFitResult]:
    """Fit a clusterer on the train partition only.

    Parameters
    ----------
    method:
        ``kmeans`` (native predict), ``agglomerative`` (nearest-centroid assign
        on holdout with disclosure), or ``dbscan`` (nearest-core / noise).
    n_clusters:
        Required for kmeans/agglomerative; ignored for dbscan (density decides).
    prefer_reduce_components:
        When True and a ``ReducePlan`` is attached with component columns on
        the frame, cluster those components instead of raw features.
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
    warnings: list[str] = []

    if method == "kmeans":
        if n_clusters is None or int(n_clusters) < 2:
            raise ValidationError("kmeans requires n_clusters >= 2")
        if int(n_clusters) > n_train:
            raise ValidationError(
                f"n_clusters={n_clusters} exceeds n_train_rows={n_train}"
            )
        estimator = KMeans(
            n_clusters=int(n_clusters),
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = estimator.fit_predict(x)
        centroids = np.asarray(estimator.cluster_centers_, dtype=float)
        centroid_ids = tuple(range(int(n_clusters)))
        assign_strategy = "native"
        inertia = float(estimator.inertia_)
        core_idx: tuple[int, ...] = ()
    elif method == "agglomerative":
        if n_clusters is None or int(n_clusters) < 2:
            raise ValidationError("agglomerative requires n_clusters >= 2")
        if int(n_clusters) > n_train:
            raise ValidationError(
                f"n_clusters={n_clusters} exceeds n_train_rows={n_train}"
            )
        estimator = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            linkage=linkage,
        )
        labels = estimator.fit_predict(x)
        centroids, centroid_ids = _centroids_from_labels(x, labels)
        assign_strategy = "nearest_centroid"
        inertia = None
        core_idx = ()
        disclosures.append(
            "AgglomerativeClustering has no native predict for new rows; "
            "holdout assign uses nearest train-cluster centroid (disclosed approximation)."
        )
    elif method == "dbscan":
        if eps <= 0:
            raise ValidationError("dbscan eps must be > 0")
        if min_samples < 1:
            raise ValidationError("dbscan min_samples must be >= 1")
        estimator = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = estimator.fit_predict(x)
        unique = sorted({int(v) for v in labels if int(v) >= 0})
        n_clusters = len(unique)
        if n_clusters < 1:
            warnings.append(
                "DBSCAN found no non-noise clusters on train; check eps/min_samples "
                "and scaling. Holdout assign will label most points as noise (-1)."
            )
        centroids, centroid_ids = (
            _centroids_from_labels(x, labels) if unique else (None, ())
        )
        core_idx = tuple(int(i) for i in getattr(estimator, "core_sample_indices_", []))
        assign_strategy = "nearest_core"
        inertia = None
        disclosures.append(
            "DBSCAN holdout assign uses nearest train core sample within eps; "
            "points farther than eps are labeled noise (-1). This is not refitting."
        )
        disclosures.append(
            "DBSCAN cluster count is density-driven; n_clusters is observed, not requested."
        )
    else:
        raise ValidationError(f"Unsupported cluster method '{method}'")

    label_list = [int(v) for v in np.asarray(labels).tolist()]
    sizes = {int(k): int(v) for k, v in sorted(Counter(label_list).items())}
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
        prefer_reduce_components=prefer_reduce_components,
        label_column=label_column,
    )
    plan = ClusterPlan(
        method=method,
        columns=tuple(cols),
        label_column=label_column,
        n_clusters=n_clusters,
        n_train_rows=n_train,
        train_labels_=tuple(label_list),
        cluster_sizes_=sizes,
        assign_strategy=assign_strategy,
        estimator_=estimator,
        centroids_=centroids,
        centroid_label_ids_=centroid_ids,
        core_sample_indices_=core_idx,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config.to_dict(),
    )
    result = ClusterFitResult(
        method=method,
        n_clusters=n_clusters,
        n_train_rows=n_train,
        columns=tuple(cols),
        cluster_sizes=sizes,
        assign_strategy=assign_strategy,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        inertia=inertia if method == "kmeans" else None,
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

    Parameters
    ----------
    partition:
        ``train``, ``validation``, ``test``, or ``all`` (full frame).
    attach:
        When True, write ``plan.label_column`` onto a copy of the dataset and
        return the mutated Dataset as the first tuple element.
    """
    frame, part_name = _frame_for_assign(dataset, split_plan, partition)
    missing = [c for c in plan.columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"Cluster plan columns missing from dataset: {missing}")
    x = matrix_from_frame(frame, list(plan.columns))
    labels = _predict_labels(plan, x)
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


def _predict_labels(plan: ClusterPlan, x: np.ndarray) -> np.ndarray:
    method = plan.method
    if method == "kmeans":
        return np.asarray(plan.estimator_.predict(x), dtype=int)
    if method == "agglomerative":
        if plan.centroids_ is None or len(plan.centroids_) == 0:
            raise ValidationError("Agglomerative plan is missing train centroids")
        return _nearest_centroid_labels(
            x, plan.centroids_, label_ids=plan.centroid_label_ids_
        )
    if method == "dbscan":
        return _dbscan_assign(plan, x)
    raise ValidationError(f"Unsupported cluster method '{method}'")


def _centroids_from_labels(
    x: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray | None, tuple[int, ...]]:
    ids = sorted({int(v) for v in labels if int(v) >= 0})
    if not ids:
        return None, ()
    centers = []
    for label in ids:
        mask = np.asarray(labels) == label
        centers.append(x[mask].mean(axis=0))
    return np.asarray(centers, dtype=float), tuple(ids)


def _nearest_centroid_labels(
    x: np.ndarray,
    centroids: np.ndarray,
    *,
    label_ids: tuple[int, ...],
) -> np.ndarray:
    dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = np.asarray(dists.argmin(axis=1), dtype=int)
    if not label_ids:
        return nearest
    mapping = np.asarray(label_ids, dtype=int)
    return mapping[nearest]


def _dbscan_assign(plan: ClusterPlan, x: np.ndarray) -> np.ndarray:
    estimator = plan.estimator_
    eps = float(getattr(estimator, "eps", plan.config.get("eps", 0.5)))
    cores = getattr(estimator, "components_", None)
    labels_arr = getattr(estimator, "labels_", None)
    core_idx = list(plan.core_sample_indices_)
    if cores is None or len(core_idx) == 0 or labels_arr is None:
        return np.full(shape=(x.shape[0],), fill_value=-1, dtype=int)
    cores_arr = np.asarray(cores, dtype=float)
    core_lab = np.asarray(labels_arr, dtype=int)[np.asarray(core_idx, dtype=int)]
    if cores_arr.shape[0] != core_lab.shape[0]:
        if plan.centroids_ is not None and len(plan.centroids_) > 0:
            raw = _nearest_centroid_labels(
                x, plan.centroids_, label_ids=plan.centroid_label_ids_
            )
            dists = np.sqrt(
                ((x[:, None, :] - plan.centroids_[None, :, :]) ** 2).sum(axis=2)
            )
            nearest = dists.min(axis=1)
            out = raw.copy()
            out[nearest > eps] = -1
            return out.astype(int)
        return np.full(shape=(x.shape[0],), fill_value=-1, dtype=int)

    dists = np.sqrt(((x[:, None, :] - cores_arr[None, :, :]) ** 2).sum(axis=2))
    nearest_i = dists.argmin(axis=1)
    nearest_d = dists[np.arange(x.shape[0]), nearest_i]
    out = core_lab[nearest_i].astype(int)
    out[nearest_d > eps] = -1
    return out
