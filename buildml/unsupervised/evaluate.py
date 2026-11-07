"""Unsupervised (and optional external) clustering evaluation."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import PartitionName, SplitPlan, frame_for_partition
from buildml.unsupervised.cluster import assign_clusters
from buildml.unsupervised.features import matrix_from_frame
from buildml.unsupervised.results import ClusterEvalResult, ClusterPlan

PartitionOrAll = PartitionName | Literal["all"]


def evaluate_clustering(
    dataset: Dataset,
    plan: ClusterPlan,
    split_plan: SplitPlan | None,
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
) -> ClusterEvalResult:
    """Score a train-fitted cluster plan on a partition without refitting.

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
external_label_column:
    external label column (str | None).
sample_size:
    sample size (int | None).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
compute_stability:
    compute stability (bool).
stability_runs:
    Number of train subsamples, at least two when stability is requested.
stability_sample_fraction:
    Finite fraction in (0, 1]; sampling is without replacement. Labels are
    compared on shared row positions against the first subsample.
compute_elbow:
    compute elbow (bool).
elbow_k_min:
    elbow k min (int).
elbow_k_max:
    elbow k max (int).

Returns
-------
ClusterEvalResult
    Serializable result summary (ClusterEvalResult) for history recording.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if compute_stability:
        if isinstance(stability_runs, (bool, np.bool_)) or not isinstance(stability_runs, Integral) or stability_runs < 2:
            raise ValidationError("stability_runs must be an integer >= 2.")
        if (
            isinstance(stability_sample_fraction, (bool, np.bool_))
            or not isinstance(stability_sample_fraction, Real)
            or not np.isfinite(stability_sample_fraction)
            or not 0 < stability_sample_fraction <= 1
        ):
            raise ValidationError("stability_sample_fraction must be finite and in (0, 1].")
    _, assign = assign_clusters(
        dataset, plan, split_plan, partition=partition, attach=False
    )
    labels = np.asarray(assign.labels, dtype=int)
    if partition == "all":
        frame = dataset._ensure_pandas()
        part_name = "all"
    else:
        if split_plan is None:
            raise ValidationError(
                f"partition='{partition}' requires a SplitPlan. Call session.split(...)."
            )
        frame = frame_for_partition(dataset, split_plan, partition)
        part_name = str(partition)

    x = matrix_from_frame(frame, list(plan.columns))
    if x.shape[0] != len(labels):
        raise ValidationError("Internal error: feature rows and labels length mismatch")

    observed = sorted({int(v) for v in labels if int(v) >= 0})
    n_clusters = len(observed)
    metrics: dict[str, float] = {}
    warnings: list[str] = []
    disclosures = [
        "Internal cluster metrics describe geometry on this partition under a "
        "train-fitted assigner; they are not predictive utility.",
        "Cluster validity is not ground truth. Prefer domain review before operational claims.",
    ]
    recommendations: list[str] = []
    diagnostics: dict[str, Any] = {}

    usable_mask = labels >= 0
    n_usable = int(usable_mask.sum())
    if n_clusters < 2 or n_usable < 3:
        warnings.append(
            "Need at least 2 non-noise clusters and 3 non-noise rows for internal metrics."
        )
    else:
        x_eval = x[usable_mask]
        y_eval = labels[usable_mask]
        if sample_size is not None and x_eval.shape[0] > int(sample_size):
            rng = np.random.default_rng(random_state)
            take = rng.choice(x_eval.shape[0], size=int(sample_size), replace=False)
            x_s = x_eval[take]
            y_s = y_eval[take]
            disclosures.append(
                f"Silhouette sampled to {len(take)} rows (sample_size={sample_size})."
            )
        else:
            x_s, y_s = x_eval, y_eval
        try:
            if len(set(int(v) for v in y_s)) >= 2:
                metrics["silhouette"] = float(silhouette_score(x_s, y_s))
        except Exception as exc:  # pragma: no cover
            warnings.append(f"silhouette unavailable: {exc}")
        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(x_eval, y_eval))
        except Exception as exc:  # pragma: no cover
            warnings.append(f"calinski_harabasz unavailable: {exc}")
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(x_eval, y_eval))
        except Exception as exc:  # pragma: no cover
            warnings.append(f"davies_bouldin unavailable: {exc}")

    if assign.n_noise:
        metrics["noise_rate"] = float(assign.n_noise) / float(max(assign.n_rows, 1))

    # Inertia / elbow from fit diagnostics or optional recompute on train
    if plan.method == "kmeans" and plan.config.get("elbow_inertia"):
        diagnostics["elbow_inertia"] = plan.config["elbow_inertia"]
        if "inertia" not in metrics and partition == "train":
            fit_inertia = plan.config.get("inertia")
            if fit_inertia is None and hasattr(plan.estimator_, "inertia_"):
                metrics["inertia"] = float(plan.estimator_.inertia_)

    if compute_elbow and plan.method in {"kmeans", "agglomerative"}:
        train_frame = (
            frame
            if partition == "train"
            else (
                frame_for_partition(dataset, split_plan, "train")
                if split_plan is not None
                else None
            )
        )
        if train_frame is not None:
            x_train = matrix_from_frame(train_frame, list(plan.columns))
            elbow = _elbow_curve(
                x_train,
                k_min=elbow_k_min,
                k_max=elbow_k_max,
                random_state=random_state,
            )
            diagnostics["elbow_inertia"] = elbow
            disclosures.append(
                "Elbow inertia curve computed on train partition (k-means refits for diagnostics only)."
            )

    if compute_stability and split_plan is not None:
        train_frame = frame_for_partition(dataset, split_plan, "train")
        x_train = matrix_from_frame(train_frame, list(plan.columns))
        k = plan.n_clusters or 2
        if k >= 2 and x_train.shape[0] >= k + 2:
            stab = _bootstrap_stability(
                x_train,
                n_clusters=int(k),
                runs=int(stability_runs),
                sample_fraction=float(stability_sample_fraction),
                random_state=random_state,
            )
            diagnostics["stability_comparisons"] = int(stab["comparisons"])
            if stab["comparisons"]:
                metrics["stability_ari_mean"] = float(stab["mean"])
                metrics["stability_ari_std"] = float(stab["std"])
            else:
                warnings.append("Stability unavailable: subsamples share fewer than 2 training rows.")
            diagnostics["stability_runs"] = int(stability_runs)
            disclosures.append(
                "Subsample stability uses repeated k-means on train subsamples; "
                "ARI compares labels on shared training rows against the first subsample: not ground truth."
            )
        else:
            warnings.append("Stability skipped: insufficient train rows or k for bootstrap.")

    if plan.method == "gmm" and plan.config.get("gmm_bic"):
        diagnostics["gmm_bic"] = plan.config["gmm_bic"]
        if plan.config.get("gmm_selected_k") is not None:
            diagnostics["gmm_selected_k"] = plan.config["gmm_selected_k"]

    transductive = plan.method in {"spectral", "optics"}
    if transductive:
        disclosures.append(
            f"{plan.method} fit is transductive on train; holdout metrics use disclosed assign approximations."
        )

    external: dict[str, float] = {}
    if external_label_column is not None:
        if external_label_column not in frame.columns:
            raise ValidationError(
                f"external_label_column '{external_label_column}' not found on partition"
            )
        ref = frame[external_label_column]
        if ref.isna().any():
            raise ValidationError(
                "external_label_column contains nulls; drop or impute before external metrics"
            )
        # Factorization needs equality/hashability, not ordering between scalars.
        ref_ids, _ = pd.factorize(ref, sort=False)
        if n_usable < len(labels):
            y_pred = labels[usable_mask]
            y_true = ref_ids[usable_mask]
            disclosures.append(
                "External ARI/NMI computed on non-noise assigned rows only."
            )
        else:
            y_pred = labels
            y_true = ref_ids
        if n_usable:
            external["adjusted_rand_index"] = float(adjusted_rand_score(y_true, y_pred))
            external["normalized_mutual_info"] = float(
                normalized_mutual_info_score(y_true, y_pred)
            )
        else:
            warnings.append("External agreement unavailable: no non-noise rows.")
        disclosures.append(
            "External labels were supplied by the caller. Agreement metrics do not "
            "make the fit supervised and do not imply causal structure."
        )

    if plan.used_reduce_components:
        recommendations.append(
            "Clusters were fit in reduced component space; interpret loadings/embeddings "
            "before naming clusters as original features."
        )
    if "silhouette" in metrics and metrics["silhouette"] < 0.15:
        recommendations.append(
            "Low silhouette suggests weak separation; revisit scaling, features, "
            "or n_clusters: or accept overlapping structure."
        )
    if partition == "train":
        recommendations.append(
            "Train-partition geometry is optimistic for model selection; prefer "
            "validation/test assign+evaluate for holdout claims."
        )
    else:
        recommendations.append(
            "Report partition name beside every metric; train-fit / holdout-assign "
            "is the leakage-safe path."
        )

    return ClusterEvalResult(
        partition=part_name,
        method=plan.method,
        n_rows=int(assign.n_rows),
        n_clusters_observed=n_clusters,
        metrics=metrics,
        external_metrics=external,
        disclosures=tuple(dict.fromkeys(disclosures)),
        warnings=tuple(warnings),
        recommendations=tuple(recommendations),
        diagnostics=diagnostics,
    )


def _elbow_curve(
    x: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    random_state: int | None,
) -> dict[str, float]:
    n_train = x.shape[0]
    hi = min(int(k_max), n_train - 1)
    lo = max(2, int(k_min))
    out: dict[str, float] = {}
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(x)
        out[str(k)] = float(km.inertia_)
    return out


def _bootstrap_stability(
    x: np.ndarray,
    *,
    n_clusters: int,
    runs: int,
    sample_fraction: float,
    random_state: int | None,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    n = x.shape[0]
    size = max(n_clusters + 1, int(n * sample_fraction))
    aris: list[float] = []
    ref_labels: np.ndarray | None = None
    ref_indices: np.ndarray | None = None
    for _ in range(runs):
        idx = rng.choice(n, size=size, replace=False)
        sub = x[idx]
        km = KMeans(n_clusters=n_clusters, random_state=int(rng.integers(0, 2**31)), n_init=3)
        lab = km.fit_predict(sub)
        if ref_labels is None:
            ref_labels = lab
            ref_indices = idx
            continue
        assert ref_indices is not None
        _, ref_positions, positions = np.intersect1d(ref_indices, idx, return_indices=True)
        if len(positions) >= 2:
            aris.append(float(adjusted_rand_score(ref_labels[ref_positions], lab[positions])))
    if not aris:
        return {"mean": float("nan"), "std": float("nan"), "comparisons": 0}
    return {"mean": float(np.mean(aris)), "std": float(np.std(aris)), "comparisons": len(aris)}
