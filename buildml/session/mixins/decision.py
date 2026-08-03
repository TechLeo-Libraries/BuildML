"""Session mixin: decision domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import decision_ops
from buildml.session.mixins._shared import *  # noqa: F403


class DecisionSessionMixin:
    """Public Session methods for the decision domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _decision_apply_result: Any
        _decision_eval_result: Any
        _decision_fit_result: Any
        _decision_plan: Any

    def fit_decision_policy(
        self,
        *,
        method: DecisionMethod = "threshold",
        backend: str | None = None,
        partition: TuningPartition = "validation",
        allow_test_tuning: bool = False,
        fp_cost: float | None = None,
        fn_cost: float | None = None,
        tp_benefit: float = 0.0,
        tn_benefit: float = 0.0,
        cost_matrix: Sequence[Sequence[float]] | None = None,
        class_labels: list[str] | None = None,
        capacity: int | None = None,
        budget: float | None = None,
        score_source: ScoreSource = "model_proba",
        score_column: str | None = None,
        cost_column: str | None = None,
        value_column: str | None = None,
        id_column: str | None = None,
        knapsack_solver: KnapsackSolver = "dp",
        objective: AllocationObjective = "maximize_score",
        min_score: float | None = None,
        lp_max_fraction: float = 1.0,
    ) -> DecisionFitResult:
        """Fit a decision policy on train or validation without refitting the model.

        Session facade over :func:`buildml.session.decision_ops.fit_decision_policy_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        DecisionFitResult
            Serializable fit summary including tuned threshold or allocation.

        See Also
        --------
        :func:`buildml.session.decision_ops.fit_decision_policy_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DecisionFitResult", decision_ops.fit_decision_policy_op(
            self,
            method=method,
            backend=backend,
            partition=partition,
            allow_test_tuning=allow_test_tuning,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            cost_matrix=cost_matrix,
            class_labels=class_labels,
            capacity=capacity,
            budget=budget,
            score_source=score_source,
            score_column=score_column,
            cost_column=cost_column,
            value_column=value_column,
            id_column=id_column,
            knapsack_solver=knapsack_solver,
            objective=objective,
            min_score=min_score,
            lp_max_fraction=lp_max_fraction,
        ))

    def apply_decisions(
        self,
        *,
        partition: PartitionName | Literal["all"] | None = "test",
        candidates: pd.DataFrame | None = None,
    ) -> ApplyDecisionsResult:
        """Apply the frozen DecisionPlan to a partition or candidate frame.

        Session facade over :func:`buildml.session.decision_ops.apply_decisions_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        DecisionApplyResult
            Selected rows, scores, and allocation metadata for the partition.

        See Also
        --------
        :func:`buildml.session.decision_ops.apply_decisions_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ApplyDecisionsResult", decision_ops.apply_decisions_op(
            self, partition=partition, candidates=candidates
        ))

    def evaluate_decisions(
        self,
        *,
        partition: PartitionName = "test",
    ) -> DecisionEvalResult:
        """Evaluate the frozen DecisionPlan on a holdout partition.

        Session facade over :func:`buildml.session.decision_ops.evaluate_decisions_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        DecisionEvalResult
            Cost, benefit, and confusion-style metrics for the frozen plan.

        See Also
        --------
        :func:`buildml.session.decision_ops.evaluate_decisions_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DecisionEvalResult", decision_ops.evaluate_decisions_op(self, partition=partition))

    @property
    def decision_plan(self) -> DecisionPlan | None:
        """Return the decision policy built by the most recent fit_decision_policy call.

        Session-held result for ``decision_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("DecisionPlan | None", self._decision_plan)

    @property
    def decision_fit_result(self) -> DecisionFitResult | None:
        """Return the report from the most recent decision-policy fit.

        Session-held result for ``decision_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("DecisionFitResult | None", self._decision_fit_result)

    @property
    def decision_eval_result(self) -> DecisionEvalResult | None:
        """Return the metrics from the most recent decision-policy evaluation.

        Session-held result for ``decision_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("DecisionEvalResult | None", self._decision_eval_result)

    @property
    def decision_apply_result(self) -> ApplyDecisionsResult | None:
        """Return the decisions from the most recent apply_decisions call.

        Session-held result for ``decision_apply_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ApplyDecisionsResult | None", self._decision_apply_result)

    def save_decision_bundle(self, path: str | Path) -> Path:
        """Persist the active DecisionPlan as ``buildml.decision_bundle.v1``.

        Session facade over :func:`buildml.session.decision_ops.save_decision_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.decision_ops.save_decision_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", decision_ops.save_decision_bundle_op(self, path=path))

    def load_decision_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a decision bundle into this Session.

        Session facade over :func:`buildml.session.decision_ops.load_decision_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with DecisionPlan attached for chaining.

        See Also
        --------
        :func:`buildml.session.decision_ops.load_decision_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", decision_ops.load_decision_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def decision_capability_matrix() -> dict[str, Any]:
        """Return the decision/optimization capability matrix for this install.

        Session facade over :func:`buildml.session.decision_ops.decision_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Nested map of method identifiers to supported backends and options.

        See Also
        --------
        :func:`buildml.session.decision_ops.decision_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", decision_ops.decision_capability_matrix_op())

    @staticmethod
    def optimize_capability_matrix() -> dict[str, Any]:
        """Return the decision/optimization capability matrix for this install.

        Session facade over :func:`buildml.session.decision_ops.decision_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Nested map of method identifiers to supported backends and options.

        See Also
        --------
        :func:`buildml.session.decision_ops.decision_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", decision_ops.decision_capability_matrix_op())
