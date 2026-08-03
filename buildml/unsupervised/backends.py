"""Backend fitters for unsupervised clustering methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.unsupervised.types import ClusterConfig

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class FitOutcome:
    """Typed container for FitOutcome state and disclosures.

Carries fitted model handles, feature contract fields, and disclosure text for walkthrough honesty.
    """
    labels: np.ndarray
    estimator: Any
    n_clusters: int | None
    centroids: np.ndarray | None
    centroid_ids: tuple[int, ...]
    core_idx: tuple[int, ...]
    assign_strategy: str
    inertia: float | None
    warnings: list[str]
    disclosures: list[str]
    extra: dict[str, Any]


def _centroids_from_labels(
    x: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray | None, tuple[int, ...]]:
    ids = sorted({int(v) for v in labels if int(v) >= 0})
    if not ids:
        return None, ()
    centers = [x[np.asarray(labels) == label].mean(axis=0) for label in ids]
    return np.asarray(centers, dtype=float), tuple(ids)


def _elbow_k(
    x: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    random_state: int | None,
    n_init: int | str,
    max_iter: int,
) -> tuple[int, dict[int, float]]:
    n_train = x.shape[0]
    hi = min(int(k_max), n_train - 1)
    lo = max(2, int(k_min))
    if hi < lo:
        raise ValidationError("auto_k range invalid for train row count")
    inertias: dict[int, float] = {}
    for k in range(lo, hi + 1):
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        km.fit(x)
        inertias[k] = float(km.inertia_)
    # Simple elbow: max second derivative of negative inertia
    ks = sorted(inertias)
    if len(ks) == 1:
        return ks[0], inertias
    vals = np.array([inertias[k] for k in ks], dtype=float)
    diffs = np.diff(vals)
    diffs2 = np.diff(diffs)
    if len(diffs2) == 0:
        return ks[0], inertias
    elbow_i = int(np.argmax(diffs2)) + 1
    return ks[min(elbow_i, len(ks) - 1)], inertias


def fit_backend(
    x: np.ndarray,
    config: ClusterConfig,
    *,
    n_train: int,
) -> FitOutcome:
    """Fit backend on the train partition using the recorded contract.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
x:
    Feature matrix input rows.
config:
    config (ClusterConfig).
n_train:
    n train (int).

Returns
-------
FitOutcome
    Return value (FitOutcome) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    method = config.method
    warnings: list[str] = []
    disclosures: list[str] = []
    extra: dict[str, Any] = {}
    n_clusters = config.n_clusters
    rs = config.random_state

    if method == "kmeans":
        if config.auto_k:
            n_clusters, elbow = _elbow_k(
                x,
                k_min=config.auto_k_min,
                k_max=config.auto_k_max,
                random_state=rs,
                n_init=config.n_init,
                max_iter=config.max_iter,
            )
            extra["elbow_inertia"] = {str(k): v for k, v in elbow.items()}
            disclosures.append(
                f"auto_k selected n_clusters={n_clusters} via elbow on train inertia."
            )
        if n_clusters is None or int(n_clusters) < 2:
            raise ValidationError("kmeans requires n_clusters >= 2 (or auto_k=True)")
        if int(n_clusters) > n_train:
            raise ValidationError(f"n_clusters={n_clusters} exceeds n_train_rows={n_train}")
        estimator = KMeans(
            n_clusters=int(n_clusters),
            random_state=rs,
            n_init=config.n_init,
            max_iter=config.max_iter,
        )
        labels = estimator.fit_predict(x)
        centroids = np.asarray(estimator.cluster_centers_, dtype=float)
        return FitOutcome(
            labels=labels,
            estimator=estimator,
            n_clusters=int(n_clusters),
            centroids=centroids,
            centroid_ids=tuple(range(int(n_clusters))),
            core_idx=(),
            assign_strategy="native",
            inertia=float(estimator.inertia_),
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method == "agglomerative":
        if n_clusters is None or int(n_clusters) < 2:
            raise ValidationError("agglomerative requires n_clusters >= 2")
        if int(n_clusters) > n_train:
            raise ValidationError(f"n_clusters={n_clusters} exceeds n_train_rows={n_train}")
        estimator = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            linkage=config.linkage,
        )
        labels = estimator.fit_predict(x)
        centroids, cids = _centroids_from_labels(x, labels)
        disclosures.append(
            "AgglomerativeClustering has no native predict; holdout assign uses "
            "nearest train-cluster centroid (disclosed approximation)."
        )
        return FitOutcome(
            labels=labels,
            estimator=estimator,
            n_clusters=int(n_clusters),
            centroids=centroids,
            centroid_ids=cids,
            core_idx=(),
            assign_strategy="nearest_centroid",
            inertia=None,
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method == "dbscan":
        if config.eps <= 0:
            raise ValidationError("dbscan eps must be > 0")
        estimator = DBSCAN(eps=float(config.eps), min_samples=int(config.min_samples))
        labels = estimator.fit_predict(x)
        unique = sorted({int(v) for v in labels if int(v) >= 0})
        n_obs = len(unique)
        if n_obs < 1:
            warnings.append("DBSCAN found no non-noise clusters on train.")
        centroids, cids = _centroids_from_labels(x, labels) if unique else (None, ())
        core_idx = tuple(int(i) for i in getattr(estimator, "core_sample_indices_", []))
        disclosures.extend(
            [
                "DBSCAN holdout assign uses nearest train core within eps; else noise (-1).",
                "DBSCAN cluster count is density-driven; n_clusters is observed, not requested.",
            ]
        )
        return FitOutcome(
            labels=labels,
            estimator=estimator,
            n_clusters=n_obs if n_obs else None,
            centroids=centroids,
            centroid_ids=cids,
            core_idx=core_idx,
            assign_strategy="nearest_core",
            inertia=None,
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method == "gmm":
        if config.auto_k or n_clusters is None:
            lo = max(2, int(config.auto_k_min))
            hi = min(int(config.auto_k_max), int(config.gmm_max_components), n_train - 1)
        else:
            k_fixed = int(n_clusters)
            if k_fixed < 2 or k_fixed > n_train - 1:
                raise ValidationError(f"gmm n_clusters={k_fixed} invalid for n_train={n_train}")
            lo = hi = k_fixed
        if hi < lo:
            raise ValidationError("GMM component search range invalid for train size")
        best_bic = float("inf")
        best_gmm: GaussianMixture | None = None
        bic_table: dict[int, float] = {}
        for k in range(lo, hi + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=config.gmm_covariance_type,
                random_state=rs,
                n_init=1,
                max_iter=config.max_iter,
            )
            gmm.fit(x)
            bic = float(gmm.bic(x))
            bic_table[k] = bic
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        assert best_gmm is not None
        labels = best_gmm.predict(x)
        n_clusters = int(best_gmm.n_components)
        centroids = np.asarray(best_gmm.means_, dtype=float)
        extra["gmm_bic"] = {str(k): v for k, v in bic_table.items()}
        extra["gmm_selected_k"] = n_clusters
        disclosures.append(
            f"GMM selected n_components={n_clusters} by BIC over k∈[{lo},{hi}] on train."
        )
        if config.auto_k and lo != hi:
            disclosures.append("auto_k enabled: BIC search used instead of fixed n_clusters.")
        return FitOutcome(
            labels=labels,
            estimator=best_gmm,
            n_clusters=n_clusters,
            centroids=centroids,
            centroid_ids=tuple(range(n_clusters)),
            core_idx=(),
            assign_strategy="gmm_predict",
            inertia=None,
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method == "hdbscan":
        from buildml.unsupervised.extras import hdbscan_available, require_hdbscan

        if not hdbscan_available():
            raise MissingExtraError(
                "unsupervised",
                "HDBSCAN clustering (pip install 'buildml[unsupervised]')",
            )
        hdbscan = require_hdbscan()
        min_samples = config.hdbscan_min_samples
        if min_samples is None:
            min_samples = config.min_samples
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(config.hdbscan_min_cluster_size),
            min_samples=int(min_samples),
            prediction_data=True,
        )
        labels = clusterer.fit_predict(x)
        unique = sorted({int(v) for v in labels if int(v) >= 0})
        n_obs = len(unique)
        centroids, cids = _centroids_from_labels(x, labels) if unique else (None, ())
        disclosures.extend(
            [
                "HDBSCAN (hdbscan library) used as industry default when buildml[unsupervised] installed.",
                "Holdout assign uses hdbscan.approximate_predict when available; "
                "else nearest train core / centroid within disclosed limits.",
            ]
        )
        return FitOutcome(
            labels=labels,
            estimator=clusterer,
            n_clusters=n_obs if n_obs else None,
            centroids=centroids,
            centroid_ids=cids,
            core_idx=(),
            assign_strategy="nearest_core",
            inertia=None,
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method == "spectral":
        if n_clusters is None or int(n_clusters) < 2:
            raise ValidationError("spectral requires n_clusters >= 2")
        if int(n_clusters) > n_train:
            raise ValidationError(f"n_clusters={n_clusters} exceeds n_train_rows={n_train}")
        estimator = SpectralClustering(
            n_clusters=int(n_clusters),
            affinity=config.spectral_affinity,
            n_neighbors=int(config.spectral_n_neighbors),
            random_state=rs,
            assign_labels="kmeans",
        )
        labels = estimator.fit_predict(x)
        centroids, cids = _centroids_from_labels(x, labels)
        disclosures.append(
            "SpectralClustering is transductive on train; holdout assign uses "
            "nearest train-cluster centroid (disclosed approximation)."
        )
        return FitOutcome(
            labels=labels,
            estimator=estimator,
            n_clusters=int(n_clusters),
            centroids=centroids,
            centroid_ids=cids,
            core_idx=(),
            assign_strategy="nearest_centroid",
            inertia=None,
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method == "optics":
        min_cluster_size = config.optics_min_cluster_size
        if min_cluster_size is None:
            min_cluster_size = int(config.optics_min_samples)
        else:
            min_cluster_size = int(min_cluster_size)
        estimator = OPTICS(
            min_samples=int(config.optics_min_samples),
            xi=float(config.optics_xi),
            min_cluster_size=min_cluster_size,
        )
        labels = estimator.fit_predict(x)
        unique = sorted({int(v) for v in labels if int(v) >= 0})
        n_obs = len(unique)
        centroids, cids = _centroids_from_labels(x, labels) if unique else (None, ())
        disclosures.extend(
            [
                "OPTICS is transductive on train; holdout assign uses nearest train centroid.",
                "Cluster count is density/order-driven; n_clusters is observed.",
            ]
        )
        return FitOutcome(
            labels=labels,
            estimator=estimator,
            n_clusters=n_obs if n_obs else None,
            centroids=centroids,
            centroid_ids=cids,
            core_idx=(),
            assign_strategy="nearest_centroid",
            inertia=None,
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method == "mean_shift":
        estimator = MeanShift(bandwidth=config.bandwidth)
        labels = estimator.fit_predict(x)
        unique = sorted({int(v) for v in labels if int(v) >= 0})
        n_obs = len(unique)
        centroids = np.asarray(getattr(estimator, "cluster_centers_", []), dtype=float)
        cids = tuple(unique) if unique else ()
        if centroids.shape[0] != len(cids):
            centroids, cids = _centroids_from_labels(x, labels)
        disclosures.append(
            "MeanShift cluster count is bandwidth-driven; holdout assign uses nearest centroid."
        )
        return FitOutcome(
            labels=labels,
            estimator=estimator,
            n_clusters=n_obs if n_obs else None,
            centroids=centroids,
            centroid_ids=cids,
            core_idx=(),
            assign_strategy="nearest_centroid",
            inertia=None,
            warnings=warnings,
            disclosures=disclosures,
            extra=extra,
        )

    if method in {"dec", "idec"}:
        from buildml.unsupervised.torch.dec import fit_dec_idec

        if n_clusters is None or int(n_clusters) < 2:
            raise ValidationError(f"{method} requires n_clusters >= 2")
        outcome = fit_dec_idec(
            x,
            method=method,
            n_clusters=int(n_clusters),
            latent_dim=int(config.latent_dim),
            pretrain_epochs=int(config.pretrain_epochs),
            finetune_epochs=int(config.finetune_epochs),
            batch_size=int(config.batch_size),
            learning_rate=float(config.learning_rate),
            random_state=rs,
        )
        return outcome

    raise ValidationError(f"Unsupported cluster method '{method}'")


def predict_backend(plan: Any, x: np.ndarray) -> np.ndarray:
    """Assign labels for holdout rows using a fitted plan.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
plan:
    Fitted plan object carrying model state and feature contract.
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    method = plan.method
    if method == "kmeans":
        return np.asarray(plan.estimator_.predict(x), dtype=int)
    if method == "gmm":
        return np.asarray(plan.estimator_.predict(x), dtype=int)
    if method in {"dec", "idec"}:
        from buildml.unsupervised.torch.dec import predict_dec_idec

        return predict_dec_idec(plan.estimator_, x)
    if method == "hdbscan":
        from buildml.unsupervised.extras import hdbscan_available

        if hdbscan_available():
            try:
                import hdbscan as hdb

                labels, _ = hdb.approximate_predict(plan.estimator_, x)
                return np.asarray(labels, dtype=int)
            except Exception:
                # approximate_predict unsupported or failed; use nearest-core fallback below.
                logger.debug(
                    "unsupervised: hdbscan.approximate_predict failed; using assign fallback",
                    exc_info=True,
                )
        # fall through to centroid/core logic
    if plan.assign_strategy == "nearest_core":
        return _dbscan_like_assign(plan, x)
    if plan.centroids_ is None or len(plan.centroids_) == 0:
        raise ValidationError(f"{method} plan is missing train centroids for assign")
    return _nearest_centroid_labels(x, plan.centroids_, label_ids=plan.centroid_label_ids_)


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
    return np.asarray(label_ids, dtype=int)[nearest]


def _dbscan_like_assign(plan: Any, x: np.ndarray) -> np.ndarray:
    estimator = plan.estimator_
    eps = float(plan.config.get("eps", getattr(estimator, "eps", 0.5)))
    cores = getattr(estimator, "components_", None)
    labels_arr = getattr(estimator, "labels_", None)
    core_idx = list(plan.core_sample_indices_)
    if cores is not None and len(core_idx) > 0 and labels_arr is not None:
        cores_arr = np.asarray(cores, dtype=float)
        core_lab = np.asarray(labels_arr, dtype=int)[np.asarray(core_idx, dtype=int)]
        dists = np.sqrt(((x[:, None, :] - cores_arr[None, :, :]) ** 2).sum(axis=2))
        nearest_i = dists.argmin(axis=1)
        nearest_d = dists[np.arange(x.shape[0]), nearest_i]
        out = core_lab[nearest_i].astype(int)
        out[nearest_d > eps] = -1
        return out
    if plan.centroids_ is not None and len(plan.centroids_) > 0:
        raw = _nearest_centroid_labels(
            x, plan.centroids_, label_ids=plan.centroid_label_ids_
        )
        dists = np.sqrt(((x[:, None, :] - plan.centroids_[None, :, :]) ** 2).sum(axis=2))
        nearest = dists.min(axis=1)
        out = raw.copy()
        out[nearest > eps] = -1
        return out.astype(int)
    return np.full(shape=(x.shape[0],), fill_value=-1, dtype=int)
