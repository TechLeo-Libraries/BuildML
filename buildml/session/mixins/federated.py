"""Session mixin: federated domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import federated_ops
from buildml.session.mixins._shared import *  # noqa: F403


class FederatedSessionMixin:
    """Public Session methods for the federated domain.

    Preferred namespaced API: ``session.federated.*`` (domain flat actions emit DeprecationWarning until BuildML 3.0).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _federated_eval_result: Any
        _federated_fit_result: Any
        _federated_plan: Any
        _federated_predict_result: Any

    def fit_federated(
        self,
        *,
        backend: FederatedBackend | None = None,
        method: FederatedMethod = "fedavg",
        estimator: FederatedEstimator = "sgd_classifier",
        task: FederatedTask | None = None,
        client_column: str | None = None,
        columns: list[str] | None = None,
        n_rounds: int = 5,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        mu: float = 0.0,
        random_state: int | None = 0,
        prefer_reduce_components: bool = True,
        min_client_rows: int = 2,
    ) -> FederatedFitResult:
        """Simulate federated averaging on this Session train clients.

        Session facade over :func:`buildml.session.federated_ops.fit_federated_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        FederatedFitResult
            Serializable fit summary including rounds, clients, and disclosures.

        See Also
        --------
        :func:`buildml.session.federated_ops.fit_federated_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("FederatedFitResult", federated_ops.fit_federated_op(
            self,
            backend=backend,
            method=method,
            estimator=estimator,
            task=task,
            client_column=client_column,
            columns=columns,
            n_rounds=n_rounds,
            local_epochs=local_epochs,
            client_fraction=client_fraction,
            mu=mu,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            min_client_rows=min_client_rows,
        ))

    def evaluate_federated(
        self,
        *,
        backend: FederatedBackend | None = None,
        partition: PartitionName | Literal["all"] = "validation",
        per_client: bool = True,
    ) -> FederatedEvalResult:
        """Evaluate the global federated model on a holdout partition.

        Session facade over :func:`buildml.session.federated_ops.evaluate_federated_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        FederatedEvalResult
            Global and optional per-client holdout metrics.

        See Also
        --------
        :func:`buildml.session.federated_ops.evaluate_federated_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("FederatedEvalResult", federated_ops.evaluate_federated_op(
            self,
            backend=backend,
            partition=partition,
            per_client=per_client,
        ))

    def predict_federated(
        self,
        *,
        backend: FederatedBackend | None = None,
        partition: PartitionName | Literal["all"] = "test",
    ) -> FederatedPredictResult:
        """Predict with the global federated model without local updates.

        Session facade over :func:`buildml.session.federated_ops.predict_federated_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        FederatedPredictResult
            Predictions from the aggregated global model.

        See Also
        --------
        :func:`buildml.session.federated_ops.predict_federated_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("FederatedPredictResult", federated_ops.predict_federated_op(
            self,
            backend=backend,
            partition=partition,
        ))

    @property
    def federated_plan(self) -> FederatedPlan | None:
        """Return the last federated plan, if any.

        Stored on this Session after :meth:`fit_federated` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        FederatedPlan or None
            ``None`` before the first :meth:`fit_federated` call on this session.
        """
        return cast("FederatedPlan | None", self._federated_plan)

    @property
    def federated_fit_result(self) -> FederatedFitResult | None:
        """Return the last federated fit result, if any.

        Stored on this Session after :meth:`fit_federated` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        FederatedFitResult or None
            ``None`` before the first :meth:`fit_federated` call on this session.
        """
        return cast("FederatedFitResult | None", self._federated_fit_result)

    @property
    def federated_eval_result(self) -> FederatedEvalResult | None:
        """Return the last federated evaluation result, if any.

        Session-held result for ``federated_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("FederatedEvalResult | None", self._federated_eval_result)

    @property
    def federated_predict_result(self) -> FederatedPredictResult | None:
        """Return the last federated prediction result, if any.

        Session-held result for ``federated_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("FederatedPredictResult | None", self._federated_predict_result)

    def save_federated_bundle(self, path: str | Path) -> Path:
        """Persist the active federated plan as ``buildml.federated_bundle.v1``.

        Session facade over :func:`buildml.session.federated_ops.save_federated_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.federated_ops.save_federated_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", federated_ops.save_federated_bundle_op(self, path=path))

    def export_round_history(
        self,
        path: str | Path,
        *,
        include_disclosures: bool = False,
    ) -> Path:
        """Export federated round metrics to JSON for audit and teaching overlays.

        Session facade over :func:`buildml.session.federated_ops.export_round_history_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved output file path.

        See Also
        --------
        :func:`buildml.session.federated_ops.export_round_history_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", federated_ops.export_round_history_op(
            self,
            path,
            include_disclosures=include_disclosures,
        ))

    def load_federated_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a federated-learning bundle into this Session.

        Session facade over :func:`buildml.session.federated_ops.load_federated_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with federated plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.federated_ops.load_federated_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", federated_ops.load_federated_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def federated_capability_matrix() -> dict[str, Any]:
        """
        Report which federated learning backends are available on this machine.

        Call before federated fit or aggregation helpers to confirm Flower, sklearn
        FedAvg, or native simulation paths. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Federated backends and install hints from
            :func:`buildml.federated.catalog.federated_capability_matrix`.
        """
        from buildml.federated.catalog import federated_capability_matrix

        return cast("dict[str, Any]", federated_capability_matrix())
