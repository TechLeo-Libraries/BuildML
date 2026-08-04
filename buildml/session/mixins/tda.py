"""Session mixin: tda domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import tda_ops
from buildml.session.mixins._shared import *  # noqa: F403


class TdaSessionMixin:
    """Public Session methods for the tda domain.

    Preferred namespaced API: ``session.tda.*`` (domain flat actions emit DeprecationWarning until BuildML 3.0).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _tda_eval_result: Any
        _tda_fit_result: Any
        _tda_plan: Any
        _tda_predict_result: Any
        _tda_transform_result: Any

    def fit_tda(
        self,
        *,
        backend: TdaBackend | None = None,
        vectorization: Vectorization = "persistence_image",
        homology_dims: Sequence[int] = (0, 1),
        knn: int = 16,
        maxdim: int = 1,
        thresh: float | None = None,
        n_bins: int = 20,
        n_layers: int = 3,
        pixel_size: float | None = None,
        standardize: bool = True,
        head: TdaHead = "logistic_regression",
        task: TdaTask | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        max_points_guard: int = 4000,
        subsample_strategy: SubsampleStrategy = "error",
        mapper: bool = False,
    ) -> TdaFitResult:
        """Fit TDA features and optional supervised head on Session train only.

        Session facade over :func:`buildml.session.tda_ops.fit_tda_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TdaFitResult
            Serializable fit summary including homology and vectorizer state.

        See Also
        --------
        :func:`buildml.session.tda_ops.fit_tda_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TdaFitResult", tda_ops.fit_tda_op(
            self,
            backend=backend,
            vectorization=vectorization,
            homology_dims=homology_dims,
            knn=knn,
            maxdim=maxdim,
            thresh=thresh,
            n_bins=n_bins,
            n_layers=n_layers,
            pixel_size=pixel_size,
            standardize=standardize,
            head=head,
            task=task,
            columns=columns,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            max_points_guard=max_points_guard,
            subsample_strategy=subsample_strategy,
            mapper=mapper,
        ))

    @staticmethod
    def tda_capability_matrix() -> dict[str, Any]:
        """Return the TDA backend and vectorization capability matrix.

        Session facade over :func:`buildml.session.tda_ops.tda_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported vectorizations.

        See Also
        --------
        :func:`buildml.session.tda_ops.tda_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", tda_ops.tda_capability_matrix_op())

    def transform_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        backend: TdaBackend | None = None,
    ) -> TdaTransformResult:
        """Transform a partition with the frozen train-fitted TDA pipeline.

        Session facade over :func:`buildml.session.tda_ops.transform_tda_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TdaTransformResult
            Vectorized persistence features for the requested partition.

        See Also
        --------
        :func:`buildml.session.tda_ops.transform_tda_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TdaTransformResult", tda_ops.transform_tda_op(self, partition=partition, backend=backend))

    def predict_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> TdaPredictResult:
        """Predict with the optional TDA supervised head on a partition.

        Session facade over :func:`buildml.session.tda_ops.predict_tda_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TdaPredictResult
            Predictions and optional probabilities from the TDA head.

        See Also
        --------
        :func:`buildml.session.tda_ops.predict_tda_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TdaPredictResult", tda_ops.predict_tda_op(self, partition=partition))

    def evaluate_tda(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        backend: TdaBackend | None = None,
        compare_diagram_distances: bool = False,
        diagram_distance_metric: DiagramDistanceMetric = "wasserstein",
        diagram_distance_dim: int = 1,
    ) -> TdaEvalResult:
        """Evaluate the TDA head on a holdout partition.

        Session facade over :func:`buildml.session.tda_ops.evaluate_tda_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        TdaEvalResult
            Holdout metrics for the supervised TDA head and optional distances.

        See Also
        --------
        :func:`buildml.session.tda_ops.evaluate_tda_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("TdaEvalResult", tda_ops.evaluate_tda_op(
            self,
            partition=partition,
            backend=backend,
            compare_diagram_distances=compare_diagram_distances,
            diagram_distance_metric=diagram_distance_metric,
            diagram_distance_dim=diagram_distance_dim,
        ))

    @property
    def tda_plan(self) -> TdaPlan | None:
        """Return the TDA plan built by the most recent fit_tda call.

        Session-held result for ``tda_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("TdaPlan | None", self._tda_plan)

    @property
    def tda_fit_result(self) -> TdaFitResult | None:
        """
        Return the report from the most recent TDA fit.

        Stored on Session after :meth:`fit_tda` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.tda.results.TdaFitResult` or None
            ``None`` until :meth:`fit_tda` has run.
        """
        return cast("TdaFitResult | None", self._tda_fit_result)

    @property
    def tda_eval_result(self) -> TdaEvalResult | None:
        """Return the metrics from the most recent TDA evaluation.

        Session-held result for ``tda_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("TdaEvalResult | None", self._tda_eval_result)

    @property
    def tda_transform_result(self) -> TdaTransformResult | None:
        """Return the topological features from the most recent transform_tda call.

        Session-held result for ``tda_transform_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("TdaTransformResult | None", self._tda_transform_result)

    @property
    def tda_predict_result(self) -> TdaPredictResult | None:
        """Return the predictions from the most recent predict_tda call.

        Session-held result for ``tda_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("TdaPredictResult | None", self._tda_predict_result)

    def save_tda_bundle(self, path: str | Path) -> Path:
        """Persist the active TdaPlan as ``buildml.tda_bundle.v2``.

        Session facade over :func:`buildml.session.tda_ops.save_tda_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.tda_ops.save_tda_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", tda_ops.save_tda_bundle_op(self, path=path))

    def load_tda_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a TDA bundle into this Session.

        Session facade over :func:`buildml.session.tda_ops.load_tda_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with TdaPlan attached for chaining.

        See Also
        --------
        :func:`buildml.session.tda_ops.load_tda_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", tda_ops.load_tda_bundle_op(self, path=path, trusted=trusted))
