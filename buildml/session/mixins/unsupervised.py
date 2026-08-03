"""Session mixin: unsupervised domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import unsupervised_ops
from buildml.session.mixins._shared import *  # noqa: F403


class UnsupervisedSessionMixin:
    """Public Session methods for the unsupervised domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _cluster_assign_result: Any
        _cluster_eval_result: Any
        _cluster_fit_result: Any
        _cluster_plan: Any

    def fit_clusters(
        self,
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
    ) -> ClusterFitResult:
        """Fit a clusterer on the train partition only.

        Session facade over :func:`buildml.session.unsupervised_ops.fit_clusters`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        ClusterFitResult
            Serializable fit summary including cluster count and method disclosures.

        See Also
        --------
        :func:`buildml.session.unsupervised_ops.fit_clusters`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ClusterFitResult", unsupervised_ops.fit_clusters(
            self,
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
            label_column=label_column,
            auto_k=auto_k,
            auto_k_min=auto_k_min,
            auto_k_max=auto_k_max,
        ))

    def assign_clusters(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
    ) -> ClusterAssignResult:
        """Assign cluster labels with the train-fitted plan without refitting.

        Session facade over :func:`buildml.session.unsupervised_ops.assign_clusters_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        ClusterAssignResult
            Cluster assignments and optional attached column metadata.

        See Also
        --------
        :func:`buildml.session.unsupervised_ops.assign_clusters_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ClusterAssignResult", unsupervised_ops.assign_clusters_op(
            self, partition=partition, attach=attach
        ))

    def evaluate_clusters(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
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
        """Evaluate train-fitted clusters on a holdout partition.

        Session facade over :func:`buildml.session.unsupervised_ops.evaluate_clusters`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        ClusterEvalResult
            Internal and optional external clustering metrics.

        See Also
        --------
        :func:`buildml.session.unsupervised_ops.evaluate_clusters`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ClusterEvalResult", unsupervised_ops.evaluate_clusters(
            self,
            partition=partition,
            external_label_column=external_label_column,
            sample_size=sample_size,
            random_state=random_state,
            compute_stability=compute_stability,
            stability_runs=stability_runs,
            stability_sample_fraction=stability_sample_fraction,
            compute_elbow=compute_elbow,
            elbow_k_min=elbow_k_min,
            elbow_k_max=elbow_k_max,
        ))

    @property
    def cluster_plan(self) -> ClusterPlan | None:
        """Return the last unsupervised cluster plan, if any.

        Stored on this Session after :meth:`fit_clusters` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        ClusterPlan or None
            ``None`` before the first :meth:`fit_clusters` call on this session.
        """
        return cast("ClusterPlan | None", self._cluster_plan)

    @property
    def cluster_fit_result(self) -> ClusterFitResult | None:
        """Return the last cluster fit result, if any.

        Stored on this Session after :meth:`fit_clusters` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        ClusterFitResult or None
            ``None`` before the first :meth:`fit_clusters` call on this session.
        """
        return cast("ClusterFitResult | None", self._cluster_fit_result)

    @property
    def cluster_assign_result(self) -> ClusterAssignResult | None:
        """Return the last cluster assignment result, if any.

        Session-held result for ``cluster_assign_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ClusterAssignResult | None", self._cluster_assign_result)

    @property
    def cluster_eval_result(self) -> ClusterEvalResult | None:
        """Return the last cluster evaluation result, if any.

        Session-held result for ``cluster_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ClusterEvalResult | None", self._cluster_eval_result)

    def save_unsupervised_bundle(self, path: str | Path) -> Path:
        """Persist the active cluster plan as ``buildml.unsupervised_bundle.v2``.

        Session facade over :func:`buildml.session.unsupervised_ops.save_unsupervised_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.unsupervised_ops.save_unsupervised_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", unsupervised_ops.save_unsupervised_bundle_op(self, path=path))

    def load_unsupervised_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an unsupervised clustering bundle into this Session.

        Session facade over :func:`buildml.session.unsupervised_ops.load_unsupervised_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with cluster plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.unsupervised_ops.load_unsupervised_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", unsupervised_ops.load_unsupervised_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def unsupervised_capability_matrix() -> dict[str, Any]:
        """
        Report which clustering and dimensionality-reduction backends are available here.

        Call before :meth:`fit_clusters` or :meth:`reduce_dimensions` to choose among
        sklearn, HDBSCAN, torch, or industry extras on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Clustering backends, methods, and install hints from
            :func:`buildml.unsupervised.catalog.unsupervised_capability_matrix`.
        """
        from buildml.unsupervised.catalog import unsupervised_capability_matrix

        return cast("dict[str, Any]", unsupervised_capability_matrix())
