"""Session mixin: timeseries domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import timeseries_ops
from buildml.session.mixins._shared import *  # noqa: F403


class TimeseriesSessionMixin:
    """Public Session methods for the timeseries domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _ts_analysis_result: Any

    def analyze_timeseries(
        self,
        *,
        target_column: str | None = None,
        time_column: str | None = None,
        scope: str = "train",
        seasonal_period: int | None = None,
        decompose_method: str | None = None,
        include_decompose: bool = True,
        include_diagnostics: bool = True,
        include_changepoints: bool = True,
        include_features: bool = True,
        acf_lags: int = 40,
        pacf_lags: int = 40,
        changepoint_penalty: float = 10.0,
        rolling_window: int = 7,
    ) -> Any:
        """Run time-series analysis on train-only or full-dataset scope.

        Session facade over :func:`buildml.session.timeseries_ops.analyze_timeseries_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        TimeseriesAnalysisResult
            Decomposition, diagnostics, changepoints, and feature summaries.

        See Also
        --------
        :func:`buildml.session.timeseries_ops.analyze_timeseries_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return timeseries_ops.analyze_timeseries_op(
            self,
            target_column=target_column,
            time_column=time_column,
            scope=scope,  # type: ignore[arg-type]
            seasonal_period=seasonal_period,
            decompose_method=decompose_method,  # type: ignore[arg-type]
            include_decompose=include_decompose,
            include_diagnostics=include_diagnostics,
            include_changepoints=include_changepoints,
            include_features=include_features,
            acf_lags=acf_lags,
            pacf_lags=pacf_lags,
            changepoint_penalty=changepoint_penalty,
            rolling_window=rolling_window,
        )

    def ts_decompose(
        self,
        *,
        target_column: str | None = None,
        time_column: str | None = None,
        scope: str = "train",
        seasonal_period: int | None = None,
        decompose_method: str | None = None,
    ) -> Any:
        """Run decomposition-only time-series analysis on Session data.

        Session facade over :func:`buildml.session.timeseries_ops.ts_decompose_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        TimeseriesAnalysisResult
            Result with decomposition components populated.

        See Also
        --------
        :func:`buildml.session.timeseries_ops.ts_decompose_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return timeseries_ops.ts_decompose_op(
            self,
            target_column=target_column,
            time_column=time_column,
            scope=scope,
            seasonal_period=seasonal_period,
            decompose_method=decompose_method,
        )

    def ts_diagnostics(
        self,
        *,
        target_column: str | None = None,
        time_column: str | None = None,
        scope: str = "train",
        acf_lags: int = 40,
        pacf_lags: int = 40,
    ) -> Any:
        """Run diagnostics-only time-series analysis on Session data.

        Session facade over :func:`buildml.session.timeseries_ops.ts_diagnostics_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        TimeseriesAnalysisResult
            Result with diagnostic tests and ACF/PACF summaries populated.

        See Also
        --------
        :func:`buildml.session.timeseries_ops.ts_diagnostics_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return timeseries_ops.ts_diagnostics_op(
            self,
            target_column=target_column,
            time_column=time_column,
            scope=scope,
            acf_lags=acf_lags,
            pacf_lags=pacf_lags,
        )

    @property
    def ts_analysis_result(self) -> Any | None:
        """Return the last time-series analysis result, if any.

        Session-held result for ``ts_analysis_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("Any | None", self._ts_analysis_result)

    @staticmethod
    def timeseries_capability_matrix() -> dict[str, Any]:
        """
        Report which time-series analysis backends are available on this machine.

        Call before :meth:`analyze_timeseries` to confirm statsmodels STL/ADF and
        ruptures changepoint extras versus core fallbacks. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            Decomposition and changepoint backends from
            :func:`buildml.timeseries.catalog.timeseries_capability_matrix`.
        """
        from buildml.timeseries.catalog import timeseries_capability_matrix

        return cast("dict[str, Any]", timeseries_capability_matrix())
