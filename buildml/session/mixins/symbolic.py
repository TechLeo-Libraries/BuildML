"""Session mixin: symbolic domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import symbolic_ops
from buildml.session.mixins._shared import *  # noqa: F403


class SymbolicSessionMixin:
    """Public Session methods for the symbolic domain.

    Preferred namespaced API: ``session.symbolic.*`` (domain flat actions emit DeprecationWarning until BuildML 3.0).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _neuro_symbolic_fit_result: Any
        _neuro_symbolic_plan: Any
        _neuro_symbolic_predict_result: Any
        _symbolic_eval_result: Any
        _symbolic_fit_result: Any
        _symbolic_plan: Any
        _symbolic_predict_result: Any

    def fit_symbolic(
        self,
        *,
        backend: SymbolicBackend | None = None,
        source: SymbolicSource = "decision_tree",
        method: IndustrySymbolicMethod | None = None,
        task: SymbolicTask | None = None,
        rules: Sequence[Mapping[str, Any] | Rule] | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        max_rules: int = 32,
        default_consequent: Any = None,
        prefer_reduce_components: bool = True,
        verify_constraints: bool = False,
    ) -> SymbolicFitResult:
        """Compile or induce a symbolic rule base on Session train.

        Session facade over :func:`buildml.session.symbolic_ops.fit_symbolic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SymbolicFitResult
            Serializable fit summary including rule count and disclosures.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.fit_symbolic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SymbolicFitResult", symbolic_ops.fit_symbolic_op(
            self,
            backend=backend,
            source=source,
            method=method,
            task=task,
            rules=rules,
            columns=columns,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_rules=max_rules,
            default_consequent=default_consequent,
            prefer_reduce_components=prefer_reduce_components,
            verify_constraints=verify_constraints,
        ))

    def evaluate_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> SymbolicEvalResult:
        """Evaluate the symbolic plan on a holdout partition.

        Session facade over :func:`buildml.session.symbolic_ops.evaluate_symbolic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SymbolicEvalResult
            Holdout metrics and rule-coverage disclosures.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.evaluate_symbolic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SymbolicEvalResult", symbolic_ops.evaluate_symbolic_op(self, partition=partition))

    def predict_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_traces: bool = True,
    ) -> SymbolicPredictResult:
        """Predict with the symbolic rule base (no update).

        Session facade over :func:`buildml.session.symbolic_ops.predict_symbolic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SymbolicPredictResult
            Predictions and optional rule traces for the partition.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.predict_symbolic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SymbolicPredictResult", symbolic_ops.predict_symbolic_op(
            self,
            partition=partition,
            return_traces=return_traces,
        ))

    def fit_neuro_symbolic(
        self,
        *,
        backend: NeuroSymbolicBackend | None = None,
        mode: NeuroSymbolicMode = "constraint_overlay",
        base_estimator: BaseEstimatorName = "logistic_regression",
        torch_method: str | None = None,
        task: SymbolicTask | None = None,
        rules: Sequence[Mapping[str, Any] | Rule] | None = None,
        rule_source: SymbolicSource = "decision_tree",
        columns: list[str] | None = None,
        random_state: int | None = 0,
        soft_strength: float = 0.5,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        max_rules: int = 24,
        prefer_reduce_components: bool = True,
        torch_epochs: int = 60,
        device: str = "cpu",
    ) -> NeuroSymbolicFitResult:
        """Fit a sklearn + symbolic hybrid on Session train.

        Session facade over :func:`buildml.session.symbolic_ops.fit_neuro_symbolic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        NeuroSymbolicFitResult
            Serializable fit summary including hybrid disclosures.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.fit_neuro_symbolic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NeuroSymbolicFitResult", symbolic_ops.fit_neuro_symbolic_op(
            self,
            backend=backend,
            mode=mode,
            base_estimator=base_estimator,
            torch_method=torch_method,
            task=task,
            rules=rules,
            rule_source=rule_source,
            columns=columns,
            random_state=random_state,
            soft_strength=soft_strength,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_rules=max_rules,
            prefer_reduce_components=prefer_reduce_components,
            torch_epochs=torch_epochs,
            device=device,
        ))

    def evaluate_neuro_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> SymbolicEvalResult:
        """Evaluate the neuro-symbolic plan on a holdout partition.

        Session facade over :func:`buildml.session.symbolic_ops.evaluate_neuro_symbolic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        SymbolicEvalResult
            Holdout metrics and hybrid overlay disclosures.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.evaluate_neuro_symbolic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SymbolicEvalResult", symbolic_ops.evaluate_neuro_symbolic_op(self, partition=partition))

    def predict_neuro_symbolic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_traces: bool = True,
    ) -> SymbolicPredictResult:
        """Predict with the neuro-symbolic hybrid (no update).

        Session facade over :func:`buildml.session.symbolic_ops.predict_neuro_symbolic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        NeuroSymbolicPredictResult
            Predictions and optional traces for the partition.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.predict_neuro_symbolic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("SymbolicPredictResult", symbolic_ops.predict_neuro_symbolic_op(
            self,
            partition=partition,
            return_traces=return_traces,
        ))

    @property
    def symbolic_plan(self) -> SymbolicPlan | None:
        """Return the symbolic rule plan built by the most recent symbolic fit.

        Session-held result for ``symbolic_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SymbolicPlan | None", self._symbolic_plan)

    @property
    def neuro_symbolic_plan(self) -> NeuroSymbolicPlan | None:
        """Return the neuro-symbolic plan built by the most recent hybrid fit.

        Session-held result for ``neuro_symbolic_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NeuroSymbolicPlan | None", self._neuro_symbolic_plan)

    @property
    def symbolic_fit_result(self) -> SymbolicFitResult | None:
        """Return the report from the most recent pure-symbolic fit.

        Session-held result for ``symbolic_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SymbolicFitResult | None", self._symbolic_fit_result)

    @property
    def neuro_symbolic_fit_result(self) -> NeuroSymbolicFitResult | None:
        """Return the report from the most recent neuro-symbolic fit.

        Session-held result for ``neuro_symbolic_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NeuroSymbolicFitResult | None", self._neuro_symbolic_fit_result)

    @property
    def symbolic_eval_result(self) -> SymbolicEvalResult | None:
        """Return the metrics from the most recent symbolic or neuro-symbolic evaluation.

        Session-held result for ``symbolic_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SymbolicEvalResult | None", self._symbolic_eval_result)

    @property
    def symbolic_predict_result(self) -> SymbolicPredictResult | None:
        """Return the predictions from the most recent pure-symbolic scoring call.

        Session-held result for ``symbolic_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SymbolicPredictResult | None", self._symbolic_predict_result)

    @property
    def neuro_symbolic_predict_result(self) -> SymbolicPredictResult | None:
        """Return the predictions from the most recent neuro-symbolic scoring call.

        Session-held result for ``neuro_symbolic_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SymbolicPredictResult | None", self._neuro_symbolic_predict_result)

    def save_symbolic_bundle(self, path: str | Path) -> Path:
        """Persist the active symbolic or neuro-symbolic plan.

        Session facade over :func:`buildml.session.symbolic_ops.save_symbolic_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.save_symbolic_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", symbolic_ops.save_symbolic_bundle_op(self, path=path))

    def load_symbolic_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a symbolic bundle into this Session.

        Session facade over :func:`buildml.session.symbolic_ops.load_symbolic_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with symbolic plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.load_symbolic_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", symbolic_ops.load_symbolic_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def symbolic_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for symbolic / neuro-symbolic backends.

        Session facade over :func:`buildml.session.symbolic_ops.symbolic_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Nested map of backend identifiers to supported sources and methods.

        See Also
        --------
        :func:`buildml.session.symbolic_ops.symbolic_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", symbolic_ops.symbolic_capability_matrix_op())
