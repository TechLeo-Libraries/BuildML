"""Session mixin: causal domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import causal_ops
from buildml.session.mixins._shared import *  # noqa: F403


class CausalSessionMixin:
    """Public Session methods for the causal domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _causal_assumptions: Any
        _causal_estimate_result: Any
        _causal_eval_result: Any
        _causal_fit_result: Any
        _causal_plan: Any
        _causal_refute_result: Any

    def declare_causal_assumptions(
        self,
        *,
        treatment: str,
        outcome: str,
        confounders: Sequence[str] | None,
        estimand: str = "ATE",
        identification: str = "backdoor",
        instruments: Sequence[str] | None = None,
        acknowledge_unconfoundedness: bool = False,
        acknowledge_positivity: bool = False,
        allow_empty_confounders: bool = False,
    ) -> CausalAssumptions:
        """Declare identification assumptions required before causal estimation.

        Session facade over :func:`buildml.session.causal_ops.declare_causal_assumptions_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CausalAssumptions
            Validated assumptions object stored on this Session.

        See Also
        --------
        :func:`buildml.session.causal_ops.declare_causal_assumptions_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CausalAssumptions", causal_ops.declare_causal_assumptions_op(
            self,
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            estimand=estimand,
            identification=identification,
            instruments=instruments,
            acknowledge_unconfoundedness=acknowledge_unconfoundedness,
            acknowledge_positivity=acknowledge_positivity,
            allow_empty_confounders=allow_empty_confounders,
        ))

    def fit_causal(
        self,
        *,
        backend: CausalBackend | None = None,
        method: CausalMethod = "aipw",
        assumptions: CausalAssumptions | dict[str, Any] | None = None,
        bootstrap_samples: int = 200,
        random_state: int | None = 0,
        clip_propensity: tuple[float, float] = (0.01, 0.99),
        outcome_model: str = "ridge",
        propensity_model: str = "logistic_regression",
    ) -> CausalFitResult:
        """Fit causal models on this Session train and estimate ATE.

        Session facade over :func:`buildml.session.causal_ops.fit_causal_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CausalFitResult
            Serializable fit summary including ATE point estimate and warnings.

        See Also
        --------
        :func:`buildml.session.causal_ops.fit_causal_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CausalFitResult", causal_ops.fit_causal_op(
            self,
            backend=backend,
            method=method,
            assumptions=assumptions,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
            clip_propensity=clip_propensity,
            outcome_model=outcome_model,
            propensity_model=propensity_model,
        ))

    def estimate_causal(
        self,
        *,
        partition: PartitionName | Literal["all"] = "train",
        bootstrap_samples: int | None = None,
        random_state: int | None = None,
    ) -> CausalEstimateResult:
        """Estimate ATE on a partition using the fitted CausalPlan.

        Session facade over :func:`buildml.session.causal_ops.estimate_causal_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CausalEstimateResult
            Partition ATE estimate with optional bootstrap interval.

        See Also
        --------
        :func:`buildml.session.causal_ops.estimate_causal_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CausalEstimateResult", causal_ops.estimate_causal_op(
            self,
            partition=partition,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        ))

    def evaluate_causal(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        bootstrap_samples: int | None = None,
    ) -> CausalEvalResult:
        """Evaluate nuisance predictive quality and ATE on a holdout partition.

        Session facade over :func:`buildml.session.causal_ops.evaluate_causal_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CausalEvalResult
            Nuisance metrics and partition ATE evaluation summary.

        See Also
        --------
        :func:`buildml.session.causal_ops.evaluate_causal_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CausalEvalResult", causal_ops.evaluate_causal_op(
            self,
            partition=partition,
            bootstrap_samples=bootstrap_samples,
        ))

    def refute_causal(
        self,
        *,
        kind: CausalRefuteKind = "placebo_treatment",
        random_state: int | None = 0,
    ) -> CausalRefuteResult:
        """Simple placebo / random-confounder sensitivity disclosure.

        Session facade over :func:`buildml.session.causal_ops.refute_causal_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        CausalRefuteResult
            Refutation outcome and sensitivity disclosures.

        See Also
        --------
        :func:`buildml.session.causal_ops.refute_causal_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CausalRefuteResult", causal_ops.refute_causal_op(
            self,
            kind=kind,
            random_state=random_state,
        ))

    @property
    def causal_assumptions(self) -> CausalAssumptions | None:
        """Return the last declared causal assumptions, if any.

        Session-held result for ``causal_assumptions``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CausalAssumptions | None", self._causal_assumptions)

    @property
    def causal_plan(self) -> CausalPlan | None:
        """Return the last causal plan, if any.

        Stored on this Session after :meth:`fit_causal` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        CausalPlan or None
            ``None`` before the first :meth:`fit_causal` call on this session.
        """
        return cast("CausalPlan | None", self._causal_plan)

    @property
    def causal_fit_result(self) -> CausalFitResult | None:
        """Return the last causal fit result, if any.

        Stored on this Session after :meth:`fit_causal` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        CausalFitResult or None
            ``None`` before the first :meth:`fit_causal` call on this session.
        """
        return cast("CausalFitResult | None", self._causal_fit_result)

    @property
    def causal_estimate_result(self) -> CausalEstimateResult | None:
        """Return the last causal estimate result, if any.

        Session-held result for ``causal_estimate_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CausalEstimateResult | None", self._causal_estimate_result)

    @property
    def causal_eval_result(self) -> CausalEvalResult | None:
        """Return the metrics from the most recent causal evaluation.

        Session-held result for ``causal_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CausalEvalResult | None", self._causal_eval_result)

    @property
    def causal_refute_result(self) -> CausalRefuteResult | None:
        """Return the refutation from the most recent refute_causal call.

        Session-held result for ``causal_refute_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CausalRefuteResult | None", self._causal_refute_result)

    def save_causal_bundle(self, path: str | Path) -> Path:
        """Persist the active CausalPlan as ``buildml.causal_bundle.v1``.

        Session facade over :func:`buildml.session.causal_ops.save_causal_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.causal_ops.save_causal_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", causal_ops.save_causal_bundle_op(self, path=path))

    def load_causal_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a causal bundle into this Session.

        Session facade over :func:`buildml.session.causal_ops.load_causal_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session with causal plan attached for chaining.

        See Also
        --------
        :func:`buildml.session.causal_ops.load_causal_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", causal_ops.load_causal_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def causal_capability_matrix() -> dict[str, Any]:
        """
        Report which causal inference backends and estimators are available here.

        Call before :meth:`estimate_causal_effect` or related causal fit methods to
        confirm DoWhy, EconML, or native paths on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Causal backends, identification methods, and install hints from
            :func:`buildml.causal.catalog.causal_capability_matrix`.
        """
        from buildml.causal.catalog import causal_capability_matrix

        return cast("dict[str, Any]", causal_capability_matrix())
