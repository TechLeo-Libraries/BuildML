"""Session mixin: forecast domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import forecast_ops
from buildml.session.mixins._shared import *  # noqa: F403


class ForecastSessionMixin:
    """Public Session methods for the forecast domain.

    Preferred namespaced API: ``session.forecast.*`` (domain flat actions emit DeprecationWarning until BuildML 3.0).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _forecast_eval_result: Any
        _forecast_fit_result: Any
        _forecast_generate_result: Any
        _forecast_plan: Any

    def fit_forecast(
        self,
        *,
        method: ForecastMethod = "auto",
        horizon: int = 1,
        lags: list[int] | tuple[int, ...] | None = None,
        seasonal_period: int | None = None,
        exog_columns: list[str] | None = None,
        target_column: str | None = None,
        time_column: str | None = None,
        random_state: int | None = 0,
        alpha: float = 1.0,
        max_iter: int = 100,
        max_depth: int | None = 3,
        learning_rate: float = 0.1,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        nbeats_input_size: int = 24,
        nbeats_horizon: int | None = None,
    ) -> ForecastFitResult:
        """Fit a classical forecaster on the train partition only.

        Session facade over :func:`buildml.session.forecast_ops.fit_forecast`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ForecastFitResult
            Serializable fit summary including method and horizon disclosures.

        See Also
        --------
        :func:`buildml.session.forecast_ops.fit_forecast`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast(
            "ForecastFitResult",
            forecast_ops.fit_forecast(
                self,
                method=method,
                horizon=horizon,
                lags=lags,
                seasonal_period=seasonal_period,
                exog_columns=exog_columns,
                target_column=target_column,
                time_column=time_column,
                random_state=random_state,
                alpha=alpha,
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                order=order,
                seasonal_order=seasonal_order,
                nbeats_input_size=nbeats_input_size,
                nbeats_horizon=nbeats_horizon,
            ),
        )

    def generate_forecast(
        self,
        *,
        horizon: int | None = None,
        origin: str = "train_end",
        future_exog: Any | None = None,
    ) -> ForecastGenerateResult:
        """Generate an H-step forecast from the train-fitted ForecastPlan.

        Session facade over :func:`buildml.session.forecast_ops.generate_forecast_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ForecastGenerateResult
            Point forecasts and optional intervals for the requested horizon.

        See Also
        --------
        :func:`buildml.session.forecast_ops.generate_forecast_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ForecastGenerateResult", forecast_ops.generate_forecast_op(
            self, horizon=horizon, origin=origin, future_exog=future_exog
        ))

    def evaluate_forecast(
        self,
        *,
        partition: PartitionName = "test",
        strategy: ForecastEvalStrategy = "rolling_one_step",
    ) -> ForecastEvalResult:
        """Evaluate the train-fitted ForecastPlan on a holdout partition.

        Session facade over :func:`buildml.session.forecast_ops.evaluate_forecast_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ForecastEvalResult
            Holdout error metrics for the frozen forecast plan.

        See Also
        --------
        :func:`buildml.session.forecast_ops.evaluate_forecast_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ForecastEvalResult", forecast_ops.evaluate_forecast_op(
            self, partition=partition, strategy=strategy
        ))

    @property
    def forecast_plan(self) -> ForecastPlan | None:
        """Return the last forecast plan, if any.

        Stored on this Session after :meth:`fit_forecast` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        ForecastPlan or None
            ``None`` before the first :meth:`fit_forecast` call on this session.
        """
        return cast("ForecastPlan | None", self._forecast_plan)

    @property
    def forecast_fit_result(self) -> ForecastFitResult | None:
        """Return the last forecast fit result, if any.

        Stored on this Session after :meth:`fit_forecast` so later calls can reuse
        the same plan without refitting.

        Returns
        -------
        ForecastFitResult or None
            ``None`` before the first :meth:`fit_forecast` call on this session.
        """
        return cast("ForecastFitResult | None", self._forecast_fit_result)

    @property
    def forecast_generate_result(self) -> ForecastGenerateResult | None:
        """Return the last forecast generation result, if any.

        Session-held result for ``forecast_generate_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ForecastGenerateResult | None", self._forecast_generate_result)

    @property
    def forecast_eval_result(self) -> ForecastEvalResult | None:
        """Return the last forecast evaluation result, if any.

        Session-held result for ``forecast_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ForecastEvalResult | None", self._forecast_eval_result)

    def save_forecast_bundle(self, path: str | Path) -> Path:
        """Persist the active ForecastPlan as ``buildml.forecast_bundle.v2``.

        Session facade over :func:`buildml.session.forecast_ops.save_forecast_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.forecast_ops.save_forecast_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", forecast_ops.save_forecast_bundle_op(self, path=path))

    def load_forecast_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Load a forecast bundle into this Session.

        Session facade over :func:`buildml.session.forecast_ops.load_forecast_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            this Session with ForecastPlan attached for chaining.

        See Also
        --------
        :func:`buildml.session.forecast_ops.load_forecast_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", forecast_ops.load_forecast_bundle_op(self, path=path, trusted=trusted))

    @staticmethod
    def forecast_capability_matrix() -> dict[str, Any]:
        """
        Report which forecasting backends and model families are available here.

        Call before :meth:`fit_forecast` to see whether statsmodels, Prophet,
        neuralforecast, or core fallbacks imported successfully. Read-only.

        Returns
        -------
        dict[str, Any]
            Forecast backends, horizons, and install hints from
            :func:`buildml.forecasting.catalog.forecast_capability_matrix`.
        """
        from buildml.forecasting.catalog import forecast_capability_matrix

        return cast("dict[str, Any]", forecast_capability_matrix())
