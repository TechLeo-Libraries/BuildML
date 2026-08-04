"""Session mixin: probabilistic domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import probabilistic_ops
from buildml.session.mixins._shared import *  # noqa: F403


class ProbabilisticSessionMixin:
    """Public Session methods for the probabilistic domain.

    Preferred namespaced API: ``session.probabilistic.*`` (domain flat actions emit DeprecationWarning until BuildML 3.0).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _probabilistic_eval_result: Any
        _probabilistic_fit_result: Any
        _probabilistic_interval_result: Any
        _probabilistic_plan: Any
        _probabilistic_predict_result: Any

    def fit_probabilistic(
        self,
        *,
        backend: str | None = None,
        estimator: ProbabilisticEstimator = "bayesian_ridge",
        task: ProbabilisticTask | None = None,
        columns: list[str] | None = None,
        random_state: int | None = 0,
        alpha: float = 0.1,
        conformal: bool = True,
        conformal_calibration_fraction: float = 0.2,
        interval_method: IntervalMethod | None = None,
        prefer_reduce_components: bool = True,
        n_restarts_optimizer: int = 0,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
    ) -> ProbabilisticFitResult:
        """Fit a Bayesian or probabilistic estimator on this Session train only.

        Session facade over :func:`buildml.session.probabilistic_ops.fit_probabilistic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ProbabilisticFitResult
            Serializable fit summary including backend and conformal disclosures.

        See Also
        --------
        :func:`buildml.session.probabilistic_ops.fit_probabilistic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ProbabilisticFitResult", probabilistic_ops.fit_probabilistic_op(
            self,
            backend=backend,
            estimator=estimator,
            task=task,
            columns=columns,
            random_state=random_state,
            alpha=alpha,
            conformal=conformal,
            conformal_calibration_fraction=conformal_calibration_fraction,
            interval_method=interval_method,
            prefer_reduce_components=prefer_reduce_components,
            n_restarts_optimizer=n_restarts_optimizer,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        ))

    def evaluate_probabilistic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
        alpha: float | None = None,
    ) -> ProbabilisticEvalResult:
        """Evaluate the probabilistic plan on a holdout partition.

        Session facade over :func:`buildml.session.probabilistic_ops.evaluate_probabilistic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ProbabilisticEvalResult
            Calibration, coverage, and sharpness metrics on the partition.

        See Also
        --------
        :func:`buildml.session.probabilistic_ops.evaluate_probabilistic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ProbabilisticEvalResult", probabilistic_ops.evaluate_probabilistic_op(
            self,
            partition=partition,
            alpha=alpha,
        ))

    def predict_probabilistic(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_std: bool = True,
        return_proba: bool = True,
    ) -> ProbabilisticPredictResult:
        """Predict with the probabilistic estimator without updating the plan.

        Session facade over :func:`buildml.session.probabilistic_ops.predict_probabilistic_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ProbabilisticPredictResult
            Point predictions with optional uncertainty outputs.

        See Also
        --------
        :func:`buildml.session.probabilistic_ops.predict_probabilistic_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ProbabilisticPredictResult", probabilistic_ops.predict_probabilistic_op(
            self,
            partition=partition,
            return_std=return_std,
            return_proba=return_proba,
        ))

    def predict_interval(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        alpha: float | None = None,
        method: str | None = None,
    ) -> ProbabilisticIntervalResult:
        """Predict predictive intervals or conformal prediction sets on a partition.

        Session facade over :func:`buildml.session.probabilistic_ops.predict_interval_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ProbabilisticIntervalResult
            Interval bounds or conformal sets per row.

        See Also
        --------
        :func:`buildml.session.probabilistic_ops.predict_interval_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ProbabilisticIntervalResult", probabilistic_ops.predict_interval_op(
            self,
            partition=partition,
            alpha=alpha,
            method=method,
        ))

    @property
    def probabilistic_plan(self) -> ProbabilisticPlan | None:
        """Return the last probabilistic plan, if any.

        Stored on this Session after :meth:`fit_probabilistic` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        ProbabilisticPlan or None
            ``None`` before the first :meth:`fit_probabilistic` call on this session.
        """
        return cast("ProbabilisticPlan | None", self._probabilistic_plan)

    @property
    def probabilistic_fit_result(self) -> ProbabilisticFitResult | None:
        """Return the last probabilistic fit result, if any.

        Session-held result for ``probabilistic_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ProbabilisticFitResult | None", self._probabilistic_fit_result)

    @property
    def probabilistic_eval_result(self) -> ProbabilisticEvalResult | None:
        """Return the last probabilistic evaluation result, if any.

        Session-held result for ``probabilistic_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ProbabilisticEvalResult | None", self._probabilistic_eval_result)

    @property
    def probabilistic_predict_result(self) -> ProbabilisticPredictResult | None:
        """Return the last probabilistic prediction result, if any.

        Session-held result for ``probabilistic_predict_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ProbabilisticPredictResult | None", self._probabilistic_predict_result)

    @property
    def probabilistic_interval_result(self) -> ProbabilisticIntervalResult | None:
        """Return the last probabilistic interval result, if any.

        Session-held result for ``probabilistic_interval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ProbabilisticIntervalResult | None", self._probabilistic_interval_result)

    def save_probabilistic_bundle(self, path: str | Path) -> Path:
        """Persist the active ProbabilisticPlan as ``buildml.probabilistic_bundle.v1``.

        Session facade over :func:`buildml.session.probabilistic_ops.save_probabilistic_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.probabilistic_ops.save_probabilistic_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", probabilistic_ops.save_probabilistic_bundle_op(self, path=path))

    def load_probabilistic_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a probabilistic bundle into this Session.

        Session facade over :func:`buildml.session.probabilistic_ops.load_probabilistic_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with ProbabilisticPlan attached for chaining.

        See Also
        --------
        :func:`buildml.session.probabilistic_ops.load_probabilistic_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", probabilistic_ops.load_probabilistic_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def probabilistic_capability_matrix() -> dict[str, Any]:
        """
        Report which probabilistic prediction backends are available on this machine.

        Call before conformal or distributional fit methods to confirm mapie,
        torch quantile heads, or sklearn fallbacks on this install. Read-only.

        Returns
        -------
        dict[str, Any]
            Probabilistic backends, interval methods, and install hints from
            :func:`buildml.probabilistic.catalog.probabilistic_capability_matrix`.
        """
        from buildml.probabilistic.catalog import probabilistic_capability_matrix

        return cast("dict[str, Any]", probabilistic_capability_matrix())
