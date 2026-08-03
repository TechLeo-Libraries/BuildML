"""Session mixin: online domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import online_ops
from buildml.session.mixins._shared import *  # noqa: F403


class OnlineSessionMixin:
    """Public Session methods for the online domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _online_eval_result: Any
        _online_fit_result: Any
        _online_plan: Any
        _online_predict_result: Any
        _online_update_result: Any

    def fit_online(
        self,
        *,
        backend: OnlineBackend | None = None,
        estimator: OnlineEstimator | str = "sgd_classifier",
        task: OnlineTask | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        chunk_size: int = 50,
        n_init: int | None = None,
        indices: list[Any] | tuple[Any, ...] | None = None,
        classes: list[Any] | tuple[Any, ...] | None = None,
        prefer_reduce_components: bool = True,
        allow_refit_fallback: bool = False,
        drift_disclose: bool = True,
        drift_detector: OnlineDriftDetector | None = None,
        buffer_size: int = 512,
        epochs_per_update: int = 5,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        ewc_lambda: float = 100.0,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> OnlineFitResult:
        """Warm-start an incremental estimator on the first train chunk.

        Session facade over :func:`buildml.session.online_ops.fit_online_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        OnlineFitResult
            Serializable fit summary including warnings and init-chunk stats.

        See Also
        --------
        :func:`buildml.session.online_ops.fit_online_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("OnlineFitResult", online_ops.fit_online_op(
            self,
            backend=backend,
            estimator=estimator,
            task=task,
            columns=columns,
            random_state=random_state,
            chunk_size=chunk_size,
            n_init=n_init,
            indices=indices,
            classes=classes,
            prefer_reduce_components=prefer_reduce_components,
            allow_refit_fallback=allow_refit_fallback,
            drift_disclose=drift_disclose,
            drift_detector=drift_detector,
            buffer_size=buffer_size,
            epochs_per_update=epochs_per_update,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ewc_lambda=ewc_lambda,
            hidden_dim=hidden_dim,
            device=device,
        ))

    def partial_fit_online(
        self,
        *,
        n_rows: int | None = None,
        indices: list[Any] | tuple[Any, ...] | None = None,
        frame: pd.DataFrame | None = None,
    ) -> OnlineUpdateResult:
        """Apply one incremental partial_fit update on the next train chunk or frame.

        Session facade over :func:`buildml.session.online_ops.partial_fit_online_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        OnlineUpdateResult
            Serializable update summary including drift notes and refit mode.

        See Also
        --------
        :func:`buildml.session.online_ops.partial_fit_online_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("OnlineUpdateResult", online_ops.partial_fit_online_op(
            self,
            n_rows=n_rows,
            indices=indices,
            frame=frame,
        ))

    def evaluate_online(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        drift_check: bool = True,
    ) -> OnlineEvalResult:
        """Evaluate the online learner on a holdout partition without updating it.

        Session facade over :func:`buildml.session.online_ops.evaluate_online_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        OnlineEvalResult
            Holdout metrics and optional drift flags. Does not mutate the estimator.

        See Also
        --------
        :func:`buildml.session.online_ops.evaluate_online_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("OnlineEvalResult", online_ops.evaluate_online_op(
            self, partition=partition, drift_check=drift_check
        ))

    def predict_online(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> OnlinePredictResult:
        """Predict with the incremental online estimator without updating it.

        Session facade over :func:`buildml.session.online_ops.predict_online_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        OnlinePredictResult
            Predictions and optional probabilities for the requested partition.

        See Also
        --------
        :func:`buildml.session.online_ops.predict_online_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("OnlinePredictResult", online_ops.predict_online_op(self, partition=partition))

    @property
    def online_plan(self) -> OnlinePlan | None:
        """Return the last online-learning plan, if any.

        Stored on this Session after :meth:`fit_online` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        OnlinePlan or None
            ``None`` before the first :meth:`fit_online` call on this session.
        """
        return cast("OnlinePlan | None", self._online_plan)

    @property
    def online_fit_result(self) -> OnlineFitResult | None:
        """Return the last online fit result, if any.

        Stored on this Session after :meth:`fit_online` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        OnlineFitResult or None
            ``None`` before the first :meth:`fit_online` call on this session.
        """
        return cast("OnlineFitResult | None", self._online_fit_result)

    @property
    def online_update_result(self) -> OnlineUpdateResult | None:
        """Return the last online update result, if any.

        Session-held result for ``online_update_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("OnlineUpdateResult | None", self._online_update_result)

    @property
    def online_eval_result(self) -> OnlineEvalResult | None:
        """Return the last online evaluation result, if any.

        Stored on this Session after :meth:`evaluate_online` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        OnlineEvalResult or None
            ``None`` before the first :meth:`evaluate_online` call on this session.
        """
        return cast("OnlineEvalResult | None", self._online_eval_result)

    @property
    def online_predict_result(self) -> OnlinePredictResult | None:
        """Return the last online prediction result, if any.

        Session-held result for ``online_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("OnlinePredictResult | None", self._online_predict_result)

    def save_online_bundle(self, path: str | Path) -> Path:
        """Persist the active online plan as ``buildml.online_bundle.v1``.

        Session facade over :func:`buildml.session.online_ops.save_online_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.online_ops.save_online_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", online_ops.save_online_bundle_op(self, path=path))

    def load_online_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an online-learning bundle into this Session.

        Session facade over :func:`buildml.session.online_ops.load_online_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with online plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.online_ops.load_online_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", online_ops.load_online_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def online_capability_matrix() -> dict[str, Any]:
        """
        Report which online and incremental learning backends are available here.

        Call before :meth:`fit_online` to see whether River, sklearn partial_fit,
        or torch streaming paths imported successfully. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Online backends, update modes, and install hints from
            :func:`buildml.online.catalog.online_capability_matrix`.
        """
        from buildml.online.catalog import online_capability_matrix

        return cast("dict[str, Any]", online_capability_matrix())
