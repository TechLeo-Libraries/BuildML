"""Session mixin: multitask domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import multitask_ops
from buildml.session.mixins._shared import *  # noqa: F403


class MultitaskSessionMixin:
    """Public Session methods for the multitask domain.

    Preferred namespaced API: ``session.multitask.*`` (domain flat actions emit DeprecationWarning until BuildML 3.0).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _multitask_eval_result: Any
        _multitask_fit_result: Any
        _multitask_plan: Any
        _multitask_predict_result: Any

    def fit_multitask(
        self,
        *,
        backend: MultiTaskBackend | None = None,
        method: MultiTaskMethod = "multi_output",
        task: MultiTaskTask = "auto",
        targets: list[str] | tuple[str, ...] | None = None,
        columns: list[str] | None = None,
        base_estimator: MultiTaskBaseEstimator | str = "logistic_regression",
        random_state: int | None = 0,
        order: list[str] | tuple[str, ...] | None = None,
        prefer_reduce_components: bool = True,
        prediction_prefix: str = "multitask_pred",
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> MultiTaskFitResult:
        """Fit a multi-target estimator on the train partition only.

        Session facade over :func:`buildml.session.multitask_ops.fit_multitask_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        MultiTaskFitResult
            Serializable fit summary per target and backend disclosures.

        See Also
        --------
        :func:`buildml.session.multitask_ops.fit_multitask_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("MultiTaskFitResult", multitask_ops.fit_multitask_op(
            self,
            backend=backend,
            method=method,
            task=task,
            targets=targets,
            columns=columns,
            base_estimator=base_estimator,
            random_state=random_state,
            order=order,
            prefer_reduce_components=prefer_reduce_components,
            prediction_prefix=prediction_prefix,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        ))

    def predict_multitask(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        attach: bool = False,
        prediction_prefix: str | None = None,
    ) -> MultiTaskPredictResult:
        """Predict all targets with the frozen multi-task plan without refitting.

        Session facade over :func:`buildml.session.multitask_ops.predict_multitask_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        MultiTaskPredictResult
            Per-target predictions and optional attached column metadata.

        See Also
        --------
        :func:`buildml.session.multitask_ops.predict_multitask_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("MultiTaskPredictResult", multitask_ops.predict_multitask_op(
            self,
            partition=partition,
            attach=attach,
            prediction_prefix=prediction_prefix,
        ))

    def evaluate_multitask(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> MultiTaskEvalResult:
        """Evaluate the multi-task plan on a holdout partition without refitting.

        Session facade over :func:`buildml.session.multitask_ops.evaluate_multitask_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        MultiTaskEvalResult
            Per-target and aggregated holdout metrics.

        See Also
        --------
        :func:`buildml.session.multitask_ops.evaluate_multitask_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("MultiTaskEvalResult", multitask_ops.evaluate_multitask_op(self, partition=partition))

    @property
    def multitask_plan(self) -> MultiTaskPlan | None:
        """Return the last multi-task plan, if any.

        Stored on this Session after :meth:`fit_multitask` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        MultiTaskPlan or None
            ``None`` before the first :meth:`fit_multitask` call on this session.
        """
        return cast("MultiTaskPlan | None", self._multitask_plan)

    @property
    def multitask_fit_result(self) -> MultiTaskFitResult | None:
        """Return the last multi-task fit result, if any.

        Stored on this Session after :meth:`fit_multitask` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        MultiTaskFitResult or None
            ``None`` before the first :meth:`fit_multitask` call on this session.
        """
        return cast("MultiTaskFitResult | None", self._multitask_fit_result)

    @property
    def multitask_predict_result(self) -> MultiTaskPredictResult | None:
        """Return the last multi-task prediction result, if any.

        Session-held result for ``multitask_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("MultiTaskPredictResult | None", self._multitask_predict_result)

    @property
    def multitask_eval_result(self) -> MultiTaskEvalResult | None:
        """Return the last multi-task evaluation result, if any.

        Session-held result for ``multitask_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("MultiTaskEvalResult | None", self._multitask_eval_result)

    def save_multitask_bundle(self, path: str | Path) -> Path:
        """Persist the active multi-task plan as ``buildml.multitask_bundle.v1``.

        Session facade over :func:`buildml.session.multitask_ops.save_multitask_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.multitask_ops.save_multitask_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", multitask_ops.save_multitask_bundle_op(self, path=path))

    def load_multitask_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a multi-task bundle into this Session.

        Session facade over :func:`buildml.session.multitask_ops.load_multitask_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with multi-task plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.multitask_ops.load_multitask_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", multitask_ops.load_multitask_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def multitask_capability_matrix() -> dict[str, Any]:
        """
        Report which multi-task learning backends are available on this machine.

        Call before :meth:`fit_multitask` to confirm chained sklearn, torch
        shared-trunk, or industry extras on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Multi-task backends, heads, and install hints from
            :func:`buildml.multitask.catalog.multitask_capability_matrix`.
        """
        from buildml.multitask.catalog import multitask_capability_matrix

        return cast("dict[str, Any]", multitask_capability_matrix())
