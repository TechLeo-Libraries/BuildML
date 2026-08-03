"""Session mixin: synthetic domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import synthetic_ops
from buildml.session.mixins._shared import *  # noqa: F403


class SyntheticSessionMixin:
    """Public Session methods for the synthetic domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _synthesizer_plan: Any
        _synthetic_eval_result: Any
        _synthetic_fit_result: Any
        _synthetic_sample_result: Any

    def fit_synthesizer(
        self,
        *,
        backend: SyntheticBackend | None = None,
        method: SynthesizerMethod = "gaussian_copula",
        columns: Sequence[str] | None = None,
        random_state: int = 42,
        smooth_sigma: float = 0.0,
        correlation_ridge: float = 1e-3,
        target_column: str | None = None,
        k_neighbors: int = 5,
        sampling_strategy: str | float | dict[str, float] = "auto",
        epochs: int = 300,
        batch_size: int = 500,
    ) -> SynthesizerFitResult:
        """Fit a tabular synthesizer on Session train rows only.

        Session facade over :func:`buildml.session.synthetic_ops.fit_synthesizer_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SyntheticFitResult
            Serializable fit summary including schema and method disclosures.

        See Also
        --------
        :func:`buildml.session.synthetic_ops.fit_synthesizer_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SynthesizerFitResult", synthetic_ops.fit_synthesizer_op(
            self,
            backend=backend,
            method=method,
            columns=columns,
            random_state=random_state,
            smooth_sigma=smooth_sigma,
            correlation_ridge=correlation_ridge,
            target_column=target_column,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            epochs=epochs,
            batch_size=batch_size,
        ))

    def sample_synthetic(
        self,
        *,
        n: int | None = None,
        random_state: int | None = None,
        condition: dict[str, Any] | None = None,
        merge_mode: MergeMode = "none",
        provenance_column: str = "_synthetic",
        validate: bool = False,
    ) -> SyntheticSampleResult:
        """Sample synthetic rows from the frozen synthesizer plan.

        Session facade over :func:`buildml.session.synthetic_ops.sample_synthetic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SyntheticSampleResult
            Sampled frame and merge metadata. May update Session dataset/split.

        See Also
        --------
        :func:`buildml.session.synthetic_ops.sample_synthetic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SyntheticSampleResult", synthetic_ops.sample_synthetic_op(
            self,
            n=n,
            random_state=random_state,
            condition=condition,
            merge_mode=merge_mode,
            provenance_column=provenance_column,
            validate=validate,
        ))

    def evaluate_synthetic(
        self,
        *,
        mode: EvalMode = "fidelity",
        eval_backend: EvalBackend = "auto",
        partition: PartitionName = "test",
        n_synthetic: int | None = None,
        random_state: int = 0,
        estimator: Literal["auto", "logistic", "ridge"] = "auto",
    ) -> SyntheticEvalResult:
        """Evaluate the frozen synthesizer for fidelity or TSTR utility.

        Session facade over :func:`buildml.session.synthetic_ops.evaluate_synthetic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SyntheticEvalResult
            Fidelity or TSTR metrics and evaluation disclosures.

        See Also
        --------
        :func:`buildml.session.synthetic_ops.evaluate_synthetic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SyntheticEvalResult", synthetic_ops.evaluate_synthetic_op(
            self,
            mode=mode,
            eval_backend=eval_backend,
            partition=partition,
            n_synthetic=n_synthetic,
            random_state=random_state,
            estimator=estimator,
        ))

    @staticmethod
    def synthetic_capability_matrix() -> dict[str, Any]:
        """Return the synthetic-data backend/method capability matrix.

        Session facade over :func:`buildml.session.synthetic_ops.synthetic_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported synthesizer methods.

        See Also
        --------
        :func:`buildml.session.synthetic_ops.synthetic_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", synthetic_ops.synthetic_capability_matrix_op())

    @property
    def synthesizer_plan(self) -> SynthesizerPlan | None:
        """Return the synthesizer plan built by the most recent fit_synthesizer call.

        Session-held result for ``synthesizer_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SynthesizerPlan | None", self._synthesizer_plan)

    @property
    def synthetic_fit_result(self) -> SynthesizerFitResult | None:
        """Return the report from the most recent synthesizer fit.

        Session-held result for ``synthetic_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SynthesizerFitResult | None", self._synthetic_fit_result)

    @property
    def synthetic_eval_result(self) -> SyntheticEvalResult | None:
        """Return the metrics from the most recent synthetic evaluation.

        Session-held result for ``synthetic_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SyntheticEvalResult | None", self._synthetic_eval_result)

    @property
    def synthetic_sample_result(self) -> SyntheticSampleResult | None:
        """Return the sample from the most recent sample_synthetic call.

        Session-held result for ``synthetic_sample_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SyntheticSampleResult | None", self._synthetic_sample_result)

    def save_synthetic_bundle(self, path: str | Path) -> Path:
        """Persist the active synthesizer plan as ``buildml.synthetic_bundle.v1``.

        Session facade over :func:`buildml.session.synthetic_ops.save_synthetic_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.synthetic_ops.save_synthetic_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", synthetic_ops.save_synthetic_bundle_op(self, path=path))

    def load_synthetic_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a synthetic-data bundle into this Session.

        Session facade over :func:`buildml.session.synthetic_ops.load_synthetic_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with synthesizer plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.synthetic_ops.load_synthetic_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", synthetic_ops.load_synthetic_bundle_op(self, path=path, trusted=trusted))
