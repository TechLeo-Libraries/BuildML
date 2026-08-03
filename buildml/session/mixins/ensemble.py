"""Session mixin: ensemble domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import ensemble_ops
from buildml.session.mixins._shared import *  # noqa: F403


class EnsembleSessionMixin:
    """Public Session methods for the ensemble domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _ensemble_fit_result: Any
        _ensemble_plan: Any

    @staticmethod
    def ensemble_capability_matrix() -> dict[str, Any]:
        """Report which ensemble strategies are available with core sklearn.

        Call before :meth:`fit_voting`, :meth:`fit_stacking`, or
        :meth:`fit_blending` to confirm strategies and non-goals on this
        install. Read-only introspection: no dataset required.

        Returns
        -------
        dict[str, Any]
            Nested backends / strategies from
            :func:`buildml.ensemble.catalog.ensemble_capability_matrix`.

        See Also
        --------
        :func:`buildml.ensemble.catalog.ensemble_capability_matrix`
            Canonical matrix fields and disclosures.
        """
        from buildml.ensemble.catalog import ensemble_capability_matrix

        return cast("dict[str, Any]", ensemble_capability_matrix())

    def fit_voting(
        self,
        estimators: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        voting: VotingMethod = "hard",
        weights: Sequence[float] | None = None,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> EnsembleFitResult:
        """Fit a voting ensemble on the train partition only.

        Session facade over :func:`buildml.session.ensemble_ops.fit_voting`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        EnsembleFitResult
            Serializable fit summary including base estimator names.

        See Also
        --------
        :func:`buildml.session.ensemble_ops.fit_voting`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EnsembleFitResult", ensemble_ops.fit_voting(
            self, estimators, voting=voting, weights=weights, task=task
        ))

    def fit_stacking(
        self,
        estimators: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        final_estimator: Any | None = None,
        cv: int = 5,
        passthrough: bool = False,
        stack_method: str = "auto",
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> EnsembleFitResult:
        """Fit a stacking ensemble on the train partition only.

        Session facade over :func:`buildml.session.ensemble_ops.fit_stacking`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        EnsembleFitResult
            Serializable fit summary including CV fold count and base names.

        See Also
        --------
        :func:`buildml.session.ensemble_ops.fit_stacking`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EnsembleFitResult", ensemble_ops.fit_stacking(
            self,
            estimators,
            final_estimator=final_estimator,
            cv=cv,
            passthrough=passthrough,
            stack_method=stack_method,
            task=task,
        ))

    def fit_blending(
        self,
        estimators: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        final_estimator: Any | None = None,
        holdout_fraction: float = 0.2,
        blend_method: BlendMethod = "predict_proba",
        random_state: int | None = 0,
        refit_bases_on_full_train: bool = True,
        passthrough: bool = False,
        task: Literal["classification", "regression", "auto"] = "auto",
    ) -> EnsembleFitResult:
        """Fit a holdout-blend ensemble on the train partition only.

        Session facade over :func:`buildml.session.ensemble_ops.fit_blending`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        EnsembleFitResult
            Serializable fit summary including holdout fraction disclosures.

        See Also
        --------
        :func:`buildml.session.ensemble_ops.fit_blending`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EnsembleFitResult", ensemble_ops.fit_blending(
            self,
            estimators,
            final_estimator=final_estimator,
            holdout_fraction=holdout_fraction,
            blend_method=blend_method,
            random_state=random_state,
            refit_bases_on_full_train=refit_bases_on_full_train,
            passthrough=passthrough,
            task=task,
        ))

    def evaluate_ensemble(
        self,
        *,
        partition: Literal["train", "validation", "test"] = "test",
    ) -> EvaluateResult:
        """Evaluate the last native ensemble with classical supervised metrics.

        Session facade over :func:`buildml.session.ensemble_ops.evaluate_ensemble`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        EvaluateResult
            Classical metrics plus ensemble strategy disclosures.

        See Also
        --------
        :func:`buildml.session.ensemble_ops.evaluate_ensemble`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EvaluateResult", ensemble_ops.evaluate_ensemble(self, partition=partition))

    @property
    def ensemble_plan(self) -> EnsemblePlan | None:
        """Return the last native ensemble plan, if any.

        Stored on this Session after :meth:`fit_voting` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        EnsemblePlan or None
            ``None`` before the first :meth:`fit_voting` call on this session.
        """
        return cast("EnsemblePlan | None", self._ensemble_plan)

    @property
    def ensemble_fit_result(self) -> EnsembleFitResult | None:
        """Return the last ensemble fit result, if any.

        Stored on this Session after :meth:`fit_voting` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        EnsembleFitResult or None
            ``None`` before the first :meth:`fit_voting` call on this session.
        """
        return cast("EnsembleFitResult | None", self._ensemble_fit_result)

    def save_ensemble_bundle(self, path: str | Path) -> Path:
        """Persist the active EnsemblePlan as ``buildml.ensemble_bundle.v1``.

        Session facade over :func:`buildml.session.ensemble_ops.save_ensemble_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.ensemble_ops.save_ensemble_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", ensemble_ops.save_ensemble_bundle_op(self, path=path))

    def load_ensemble_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load an ensemble bundle into this Session.

        Session facade over :func:`buildml.session.ensemble_ops.load_ensemble_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with EnsemblePlan and ``fit_result`` attached.

        See Also
        --------
        :func:`buildml.session.ensemble_ops.load_ensemble_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", ensemble_ops.load_ensemble_bundle_op(self, path=path, trusted=trusted))
