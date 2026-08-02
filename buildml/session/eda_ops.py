"""EDA and local dashboard entry orchestration."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


def eda(
    session,
    *,
    include_plots: bool = False,
    show: bool = False,
    sample_rows: int | None = None,
    max_columns: int = 100,
    max_plots: int = 36,
    export_html: str | Path | None = None,
    export_figures: str | Path | None = None,
    html_format: Literal['studio', 'research'] = "studio",
) -> EDAReport:
    """Run exploratory analysis.

    Includes quality/pattern screens, distributional tests, correlations,
    mutual information, VIF/PCA, target-aware tests, outlier screens,
    train/test drift (if split exists), adaptive visualization planning,
    narrative generation, and optional HTML/figure export.

    Parameters
    ----------
    include_plots:
        Render adaptive plots (requires ``pip install 'buildml[viz]'``).
    show:
        Print the narrative summary.
    sample_rows:
        Optional analysis sample size for large datasets.
    max_columns:
        Maximum columns used by detailed analyzers. Dataset-wide quality
        checks still cover the full schema.
    max_plots:
        Cap on adaptive plot specifications.
    export_html:
        Optional path for a self-contained HTML artifact. Default format is
        an offline Teaching Studio snapshot (same surface as ``eda_app``).
    export_figures:
        Optional directory for saved PNG figures.
    html_format:
        ``"studio"`` (default) writes the offline Teaching Studio; ``"research"``
        writes the layered research HTML shell with matplotlib embeds.
    """
    report = explore_dataset(
        session.dataset,
        split_plan=session._split_plan,
        sample_rows=sample_rows,
        max_columns=max_columns,
        max_plots=max_plots,
        include_plots=include_plots,
        show=show,
        export_html=export_html,
        export_figures=export_figures,
        html_format=html_format,
    )
    from buildml.session.walkthrough import (
        preprocess_scope_status,
        rag_status_for_walkthrough,
        torch_training_status_for_walkthrough,
        warm_start_studies_status,
    )

    warm = warm_start_studies_status(session._history, last_nested_cv=session._last_nested_cv)
    report.overview["warm_start_status"] = warm
    report.overview["preprocess_scope_status"] = preprocess_scope_status(
        session._history,
        session=session,
        last_cv=session._last_cv,
        last_nested_cv=session._last_nested_cv,
    )
    report.overview["torch_training_status"] = torch_training_status_for_walkthrough(session)
    report.overview["rag_status"] = rag_status_for_walkthrough(session)
    session._last_eda = report
    session._record(
        "eda",
        {
            "include_plots": include_plots,
            "show": show,
            "sample_rows": sample_rows,
            "max_columns": max_columns,
            "max_plots": max_plots,
            "export_html": export_html,
            "export_figures": export_figures,
            "html_format": html_format,
        },
        result_summary={
            "n_rows": report.overview.get("n_rows"),
            "n_columns": report.overview.get("n_columns"),
            "recommendations": len(report.recommendations),
            "narrative": len(report.narrative),
            "plots": len(report.figures),
            "html_path": report.html_path,
            "html_format": html_format if export_html is not None else None,
            "warm_start_studies": bool(warm.get("enabled")),
        },
    )
    return report


def eda_app(
    session,
    *,
    report: EDAReport | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    title: str = "BuildML EDA Studio",
    sample_rows: int | None = None,
    max_columns: int = 100,
    blocking: bool = False,
) -> Any:
    """Launch the local EDA Teaching Studio web app.

    Runs a FastAPI process on the local host and opens a browser to an
    interactive product UI (domain boards, Teaching Studio, Concept Academy,
    Plotly charts, PDF/CSV export). Requires ``pip install 'buildml[dashboard]'``.

    Parameters
    ----------
    report:
        Optional existing :class:`~buildml.eda.report.EDAReport`. When omitted,
        uses the last ``eda()`` result or runs a fresh analysis.
    host, port:
        Local bind address for the ASGI server.
    open_browser:
        Open the system browser when the server is ready.
    title:
        App header title.
    sample_rows, max_columns:
        Forwarded to ``eda()`` when a fresh report must be computed.
    blocking:
        If True, serve on the current thread until interrupted.

    Returns
    -------
    EDAAppHandle
        Handle with ``url``, ``stop()``, and ``is_running``.
    """
    from buildml.dashboard.launch import launch_eda_app

    eda_report = report or session._last_eda
    if eda_report is None:
        eda_report = session.eda(
            include_plots=False, show=False, sample_rows=sample_rows, max_columns=max_columns
        )
    roles = {}
    if session._dataset is not None:
        roles = {
            str(column): getattr(role, "value", str(role))
            for column, role in session.dataset.roles.items()
        }
    meta = {
        "has_split": session._split_plan is not None,
        "history_len": len(session._history),
        "roles": roles,
    }
    handle = launch_eda_app(
        eda_report,
        host=host,
        port=port,
        open_browser=open_browser,
        title=title,
        session_meta=meta,
        blocking=blocking,
    )
    session._eda_app_handle = handle
    session._record(
        "eda_app",
        {"host": host, "port": port, "title": title, "url": handle.url},
        result_summary={"url": handle.url},
    )
    return handle


def open_eda_dashboard(
    session,
    *,
    report: EDAReport | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    title: str = "BuildML EDA Studio",
    sample_rows: int | None = None,
    max_columns: int = 100,
    blocking: bool = False,
) -> Any:
    """Alias for :meth:`eda_app`."""
    return session.eda_app(
        report=report,
        host=host,
        port=port,
        open_browser=open_browser,
        title=title,
        sample_rows=sample_rows,
        max_columns=max_columns,
        blocking=blocking,
    )
