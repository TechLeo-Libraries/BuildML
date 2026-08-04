"""Session mixin: eda domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import eda_ops
from buildml.session.mixins._shared import *  # noqa: F403


class EdaSessionMixin:
    """Public Session methods for the eda domain.

    Preferred namespaced API: ``session.explore.*`` (classical/core dual: flat methods remain first-class without warnings).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _last_eda: Any

    def eda(
        self,
        *,
        include_plots: bool = False,
        show: bool = False,
        sample_rows: int | None = None,
        max_columns: int = 100,
        max_plots: int = 36,
        export_html: str | Path | None = None,
        export_figures: str | Path | None = None,
        html_format: Literal["studio", "research"] = "studio",
    ) -> EDAReport:
        """Understand the data before you model it.

        Session facade over :func:`buildml.session.eda_ops.eda`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.eda.report.EDAReport
            The findings, their interpretation, the recommendations drawn from

        See Also
        --------
        :func:`buildml.session.eda_ops.eda`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EDAReport", eda_ops.eda(
            self,
            include_plots=include_plots,
            show=show,
            sample_rows=sample_rows,
            max_columns=max_columns,
            max_plots=max_plots,
            export_html=export_html,
            export_figures=export_figures,
            html_format=html_format,
        ))

    def eda_app(
        self,
        *,
        report: EDAReport | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = True,
        title: str = "BuildML EDA Studio",
        sample_rows: int | None = None,
        max_columns: int = 100,
        blocking: bool = False,
    ) -> EDAAppHandle:
        """Explore the data interactively in a browser instead of on paper.

        Session facade over :func:`buildml.session.eda_ops.eda_app`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.dashboard.launch.EDAAppHandle
            A handle exposing ``url``, ``is_running``, and ``stop()``. Call

        See Also
        --------
        :func:`buildml.session.eda_ops.eda_app`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EDAAppHandle", eda_ops.eda_app(
            self,
            report=report,
            host=host,
            port=port,
            open_browser=open_browser,
            title=title,
            sample_rows=sample_rows,
            max_columns=max_columns,
            blocking=blocking,
        ))

    def open_eda_dashboard(
        self,
        *,
        report: EDAReport | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = True,
        title: str = "BuildML EDA Studio",
        sample_rows: int | None = None,
        max_columns: int = 100,
        blocking: bool = False,
    ) -> EDAAppHandle:
        """Open the interactive EDA studio: an alias for :meth:`eda_app`.

        Session facade over :func:`buildml.session.eda_ops.open_eda_dashboard`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.dashboard.launch.EDAAppHandle
            A handle exposing ``url``, ``is_running``, and ``stop()``.

        See Also
        --------
        :func:`buildml.session.eda_ops.open_eda_dashboard`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EDAAppHandle", eda_ops.open_eda_dashboard(
            self,
            report=report,
            host=host,
            port=port,
            open_browser=open_browser,
            title=title,
            sample_rows=sample_rows,
            max_columns=max_columns,
            blocking=blocking,
        ))

    @property
    def last_eda(self) -> EDAReport | None:
        """The most recent exploratory analysis report.

        Session-held result for ``last_eda``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("EDAReport | None", self._last_eda)
