"""Session mixin: cbr domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import cbr_ops
from buildml.session.mixins._shared import *  # noqa: F403


class CbrSessionMixin:
    """Public Session methods for the cbr domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _cbr_eval_result: Any
        _cbr_fit_result: Any
        _cbr_plan: Any
        _cbr_predict_result: Any
        _cbr_retain_result: Any
        _cbr_retrieve_result: Any

    @staticmethod
    def cbr_capability_matrix() -> dict[str, Any]:
        """
        Report which case-based retrieval backends are available on this machine.

        Call before :meth:`fit_cbr` when choosing among sklearn kNN, ANN industry
        extras, text embeddings, or torch metric encoders. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Retrieval backends, metrics, and install hints from
            :func:`buildml.cbr.catalog.cbr_capability_matrix`.
        """
        from buildml.cbr.catalog import cbr_capability_matrix

        return cast("dict[str, Any]", cbr_capability_matrix())

    def fit_cbr(
        self,
        *,
        backend: str | None = None,
        task: CbrTask | None = None,
        metric: CbrMetric = "euclidean",
        reuse: CbrReuseMode = "distance_weighted",
        adapt: CbrAdaptMode = "none",
        k: int = 5,
        columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        text_columns: list[str] | None = None,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        standardize: bool = True,
        distance_eps: float = 1e-8,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        torch_epochs: int = 40,
        device: str = "cpu",
    ) -> CbrFitResult:
        """Build a case base from Session train.

        Session facade over :func:`buildml.session.cbr_ops.fit_cbr_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CbrFitResult
            Serializable fit summary including case-base size and disclosures.

        See Also
        --------
        :func:`buildml.session.cbr_ops.fit_cbr_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CbrFitResult", cbr_ops.fit_cbr_op(
            self,
            backend=backend,
            task=task,
            metric=metric,
            reuse=reuse,
            adapt=adapt,
            k=k,
            columns=columns,
            categorical_columns=categorical_columns,
            text_columns=text_columns,
            text_model_name=text_model_name,
            standardize=standardize,
            distance_eps=distance_eps,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            torch_epochs=torch_epochs,
            device=device,
        ))

    def retrieve_cases(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int | None = None,
        backend: str | None = None,
    ) -> CbrRetrieveResult:
        """Retrieve k nearest cases for a partition (no reuse).

        Session facade over :func:`buildml.session.cbr_ops.retrieve_cases_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CbrRetrieveResult
            Retrieved cases and distance traces for each query row.

        See Also
        --------
        :func:`buildml.session.cbr_ops.retrieve_cases_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CbrRetrieveResult", cbr_ops.retrieve_cases_op(
            self, partition=partition, k=k, backend=backend
        ))

    def predict_cbr(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        k: int | None = None,
        return_traces: bool = True,
        backend: str | None = None,
    ) -> CbrPredictResult:
        """Predict via retrieve + reuse (no case-base update).

        Session facade over :func:`buildml.session.cbr_ops.predict_cbr_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CbrPredictResult
            Predictions and optional retrieval traces for the partition.

        See Also
        --------
        :func:`buildml.session.cbr_ops.predict_cbr_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CbrPredictResult", cbr_ops.predict_cbr_op(
            self,
            partition=partition,
            k=k,
            return_traces=return_traces,
            backend=backend,
        ))

    def evaluate_cbr(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        k: int | None = None,
    ) -> CbrEvalResult:
        """Evaluate CBR on a holdout partition.

        Session facade over :func:`buildml.session.cbr_ops.evaluate_cbr_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CbrEvalResult
            Holdout metrics and retrieval disclosures.

        See Also
        --------
        :func:`buildml.session.cbr_ops.evaluate_cbr_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CbrEvalResult", cbr_ops.evaluate_cbr_op(self, partition=partition, k=k))

    def retain_cbr(
        self,
        *,
        labeled_frame: Any | None = None,
        row_indices: Sequence[Any] | None = None,
        solution_column: str | None = None,
        source_disclosure: str,
        allow_overlap_with_train: bool = True,
    ) -> CbrRetainResult:
        """Retain new labeled cases (refuses Session validation/test indices).

        Session facade over :func:`buildml.session.cbr_ops.retain_cbr_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CbrRetainResult
            Retain summary including updated case-base size.

        See Also
        --------
        :func:`buildml.session.cbr_ops.retain_cbr_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CbrRetainResult", cbr_ops.retain_cbr_op(
            self,
            labeled_frame=labeled_frame,
            row_indices=row_indices,
            solution_column=solution_column,
            source_disclosure=source_disclosure,
            allow_overlap_with_train=allow_overlap_with_train,
        ))

    @property
    def cbr_plan(self) -> CbrPlan | None:
        """Return the case memory built by the most recent CBR fit.

        Session-held result for ``cbr_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CbrPlan | None", self._cbr_plan)

    @property
    def cbr_fit_result(self) -> CbrFitResult | None:
        """
        Return the report from the most recent CBR fit.

        Stored on Session after :meth:`fit_cbr` so downstream calls can replay the same plan without refitting.

        Returns
        -------
        :class:`~buildml.cbr.results.CbrFitResult` or None
            ``None`` until :meth:`fit_cbr` has run.
        """
        return cast("CbrFitResult | None", self._cbr_fit_result)

    @property
    def cbr_eval_result(self) -> CbrEvalResult | None:
        """Return the metrics from the most recent CBR evaluation.

        Session-held result for ``cbr_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CbrEvalResult | None", self._cbr_eval_result)

    @property
    def cbr_predict_result(self) -> CbrPredictResult | None:
        """Return the predictions from the most recent CBR scoring call.

        Session-held result for ``cbr_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CbrPredictResult | None", self._cbr_predict_result)

    @property
    def cbr_retrieve_result(self) -> CbrRetrieveResult | None:
        """Return the nearest cases from the most recent retrieval call.

        Session-held result for ``cbr_retrieve_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CbrRetrieveResult | None", self._cbr_retrieve_result)

    @property
    def cbr_retain_result(self) -> CbrRetainResult | None:
        """Return the report from the most recent case retention call.

        Session-held result for ``cbr_retain_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CbrRetainResult | None", self._cbr_retain_result)

    def save_cbr_bundle(self, path: str | Path) -> Path:
        """Persist the active CbrPlan.

        Session facade over :func:`buildml.session.cbr_ops.save_cbr_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.cbr_ops.save_cbr_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", cbr_ops.save_cbr_bundle_op(self, path=path))

    def load_cbr_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a CBR bundle into this Session.

        Session facade over :func:`buildml.session.cbr_ops.load_cbr_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with CBR plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.cbr_ops.load_cbr_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", cbr_ops.load_cbr_bundle_op(self, path=path, trusted=trusted))
