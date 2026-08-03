"""Thin Session facades over buildml.unsupervised."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
    """Fit a clusterer on the train partition only.

    Delegates to :func:`buildml.unsupervised.cluster.fit_clusterer`, stores the
    :class:`~buildml.unsupervised.results.ClusterPlan` on Session, and records
    the fit. Follow with :func:`assign_clusters_op` or :func:`evaluate_clusters`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    method:
        Clustering method key (``kmeans``, ``gmm``, ``hdbscan``, etc.).
    n_clusters:
        Target cluster count for parametric methods; ignored for density methods.
    columns:
        Optional explicit feature columns; ``None`` auto-selects numerics.
    random_state:
        Seed for stochastic initialization and sampling.
    n_init:
        Number of k-means restarts (``auto`` uses sklearn default).
    max_iter:
        Maximum iterations for iterative clusterers.
    linkage:
        Linkage criterion for hierarchical clustering.
    eps:
        Neighborhood radius for DBSCAN.
    min_samples:
        Minimum samples per core point for DBSCAN/OPTICS.
    gmm_covariance_type:
        Covariance structure for Gaussian mixture models.
    gmm_max_components:
        Upper bound on components when ``auto_k`` selects GMM k.
    gmm_select_by:
        Model-selection score for GMM component count (``bic`` or ``aic``).
    hdbscan_min_cluster_size:
        Minimum cluster size for HDBSCAN.
    hdbscan_min_samples:
        Core distance samples for HDBSCAN; defaults to min cluster size.
    spectral_affinity:
        Affinity matrix type for spectral clustering.
    spectral_n_neighbors:
        Neighbors for spectral nearest-neighbors affinity.
    optics_min_samples:
        Minimum samples for OPTICS core distances.
    optics_xi:
        Steepness threshold for OPTICS cluster extraction.
    optics_min_cluster_size:
        Minimum cluster size for OPTICS extraction.
    bandwidth:
        Kernel bandwidth for mean-shift; ``None`` estimates from data.
    latent_dim:
        Embedding dimension for deep clustering backend.
    pretrain_epochs:
        Pretraining epochs for deep clustering autoencoder.
    finetune_epochs:
        Fine-tuning epochs for deep clustering head.
    batch_size:
        Minibatch size for deep clustering backend.
    learning_rate:
        Optimizer learning rate for deep clustering backend.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists on Session.
    label_column:
        Output column name for cluster assignments when attaching.
    auto_k:
        When True, search ``auto_k_min``..``auto_k_max`` for k-means/GMM.
    auto_k_min:
        Lower bound for automatic k search.
    auto_k_max:
        Upper bound for automatic k search.

    Returns
    -------
    ClusterFitResult
        Serializable fit summary including cluster count and method disclosures.
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
    """Assign cluster labels with the train-fitted plan without refitting.

    Delegates to :func:`buildml.unsupervised.cluster.assign_clusters`. When
    ``attach=True``, cluster labels are merged into Session dataset.

    Parameters
    ----------
    session:
        Active Session with a cluster plan from :func:`fit_clusters`.
    partition:
        Partition to assign (``train``, ``validation``, ``test``, or ``all``).
    attach:
        When True, attach cluster label column to the Session dataset frame.

    Returns
    -------
    ClusterAssignResult
        Cluster assignments and optional attached column metadata.

    Raises
    ------
    ValidationError
        When no cluster plan exists on the Session.
    """
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
    """Evaluate train-fitted clusters on a holdout partition.

    Delegates to :func:`buildml.unsupervised.evaluate.evaluate_clustering`.
    Computes internal metrics and optional external alignment when labels exist.

    Parameters
    ----------
    session:
        Active Session with a cluster plan from :func:`fit_clusters`.
    partition:
        Holdout partition to score. Validation falls back to test when absent.
    external_label_column:
        Optional column for external cluster-quality metrics (e.g. ARI).
    sample_size:
        Optional subsample size for expensive metrics; ``None`` uses all rows.
    random_state:
        Seed for subsampling and stability bootstraps.
    compute_stability:
        When True, run bootstrap stability diagnostics.
    stability_runs:
        Number of bootstrap runs for stability analysis.
    stability_sample_fraction:
        Fraction of rows sampled per stability bootstrap.
    compute_elbow:
        When True, compute elbow curve for k-means k selection diagnostics.
    elbow_k_min:
        Minimum k for elbow diagnostics.
    elbow_k_max:
        Maximum k for elbow diagnostics.

    Returns
    -------
    ClusterEvalResult
        Internal and optional external clustering metrics.

    Raises
    ------
    ValidationError
        When no cluster plan exists on the Session.
    """
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
    """Persist the active cluster plan as ``buildml.unsupervised_bundle.v2``.

    Delegates to :func:`buildml.unsupervised.checkpoint.save_unsupervised_bundle`.
    Reload with :func:`load_unsupervised_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a cluster plan from :func:`fit_clusters`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no cluster plan exists on the Session.
    """
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


def load_unsupervised_bundle_op(session, path: str | Path, *, trusted: bool = False) -> Any:
    """Load an unsupervised clustering bundle into this Session.

    Delegates to :func:`buildml.unsupervised.checkpoint.load_unsupervised_bundle`
    and clears prior fit/assign/eval results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded cluster plan.
    path:
        Path to a ``buildml.unsupervised_bundle.v2`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with cluster plan attached for chaining.
    """
    plan = load_unsupervised_bundle(path, trusted=trusted)
    session._cluster_plan = plan
    session._cluster_fit_result = None
    session._cluster_assign_result = None
    session._cluster_eval_result = None
    session._record(
        "load_unsupervised_bundle",
        {"path": str(path), "method": plan.method, "n_clusters": plan.n_clusters},
        result_summary=plan.to_dict(),
    )
    return cast("Session", session)