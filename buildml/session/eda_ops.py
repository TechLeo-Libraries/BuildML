"""EDA and local dashboard entry orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from buildml.session.session import Session

from buildml.session._imports import (
    EDAReport,
    explore_dataset,
)


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
    """Understand the data before you model it.

    Modelling before looking at the data is how people discover, three
    weeks in, that a column is 80% missing, that two features are the same
    number in different units, or that the target is nearly constant. This
    runs the checks that would have caught it.

    The screens cover data quality (missing values, constant and duplicate
    columns, suspicious cardinality), distributions and their skew,
    correlations between features and multicollinearity via VIF and PCA,
    mutual information against the target, and outlier detection. When a
    split exists it also compares train against test and reports drift :
    systematic differences between the two that would make your holdout
    estimate misleading.

    The output is narrated rather than dumped. Each finding comes with what
    it means and what to consider doing about it.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    include_plots:
        Generate charts alongside the statistics. The plots are chosen to
        suit each column's type and distribution rather than drawn
        uniformly. Requires ``pip install 'buildml[viz]'``.
    show:
        Print the narrative summary to standard output, for notebook use.
    sample_rows:
        Analyse a random sample of this many rows instead of all of them.
        Worth setting on a large table, where the statistics stabilise long
        before the row count is exhausted.
    max_columns:
        How many columns the detailed analysers cover. Dataset-wide quality
        checks still see every column; this caps the expensive per-column
        work on very wide tables.
    max_plots:
        Upper bound on charts generated, so a wide table does not produce
        hundreds of figures.
    export_html:
        Path for a self-contained HTML report: the artefact to share with
        someone who will not run the code.
    export_figures:
        Directory to write individual PNG figures into.
    html_format:
        ``'studio'`` writes the interactive offline studio layout, the same
        surface :meth:`eda_app` serves. ``'research'`` writes a layered
        document with embedded matplotlib figures, better suited to reading
        top to bottom.

    Returns
    -------
    ~buildml.eda.report.EDAReport
        The findings, their interpretation, the recommendations drawn from
        them, and paths to anything exported. Also stored on
        :attr:`last_eda`.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No dataset is attached.
    ~buildml.core.errors.MissingExtraError
        Plots were requested without ``buildml[viz]`` installed.

    Notes
    -----
    **Leakage:** Exploration is how analysts leak without noticing. Every
    pattern you find by looking at the whole dataset: including the test
    rows: informs decisions you then make about the model, so the test set
    stops being independent. Split first, and explore the training rows.
    The drift comparison is the exception: it exists precisely to compare
    partitions and reports only aggregate differences.

    **Scale:** Correlation and mutual-information analysis grows quickly
    with column count. Use ``sample_rows`` and ``max_columns`` on wide or
    tall tables.

    Examples
    --------
    >>> report = session.eda(export_html="reports/eda.html")  # doctest: +SKIP
    >>> report.recommendations[:2]  # doctest: +SKIP

    See Also
    --------
    Session.eda_app : The same analysis, served interactively.
    Session.head : A quick look rather than a full profile.
    Session.error_slices : Where the model fails, after fitting.
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
    """Explore the data interactively in a browser instead of on paper.

    Starts a local web server and opens the EDA studio: the same analysis
    :meth:`eda` produces, but navigable: click into a column to see its
    distribution, sort correlations, filter findings, read the concept
    explanations behind each screen, and export what you find as PDF or
    CSV.

    The advantage over a static report is following a thread. Noticing that
    one column is skewed usually prompts a question about a second column,
    and clicking is faster than re-running an analysis with different
    arguments.

    Nothing leaves your machine: the server binds to localhost by default.
    Requires ``pip install 'buildml[dashboard]'``.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    report:
        An existing :class:`~buildml.eda.report.EDAReport` to display.
        ``None`` reuses :attr:`last_eda` if present, and otherwise runs a
        fresh analysis first.
    host:
        Address to bind to. The default keeps the app on this machine;
        change it only if you intend the app to be reachable from
        elsewhere, and understand that your data becomes reachable too.
    port:
        Port to serve on. Change it if the default is already taken.
    open_browser:
        Open your browser automatically once the server is ready.
    title:
        Heading shown in the app, useful when several are running.
    sample_rows:
        Row sample size, forwarded to :meth:`eda` when a fresh report has
        to be computed.
    max_columns:
        Column cap, forwarded to :meth:`eda` on a fresh computation.
    blocking:
        Serve on the current thread until interrupted, rather than
        returning immediately. Use this in a script that would otherwise
        exit and take the server with it; leave it off in a notebook, where
        you want the cell to finish.

    Returns
    -------
    ~buildml.dashboard.launch.EDAAppHandle
        A handle exposing ``url``, ``is_running``, and ``stop()``. Call
        ``stop()`` when finished: a non-blocking server keeps running
        until you do.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        ``buildml[dashboard]`` is not installed.
    ~buildml.core.errors.ValidationError
        No dataset is attached and no report was supplied.

    Examples
    --------
    >>> app = session.eda_app()  # doctest: +SKIP
    >>> app.url  # doctest: +SKIP
    'http://127.0.0.1:8765'
    >>> app.stop()  # doctest: +SKIP

    See Also
    --------
    Session.eda : The same analysis as a static report.
    Session.open_eda_dashboard : An alias for this method.
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
    """Open the interactive EDA studio: an alias for :meth:`eda_app`.

    Identical behaviour under a more discoverable name. See
    :meth:`eda_app` for the full description of every argument.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    report:
        Existing report to display, or ``None`` to reuse or compute one.
    host:
        Address to bind to.
    port:
        Port to serve on.
    open_browser:
        Open the system browser once the server is ready.
    title:
        Heading shown in the app.
    sample_rows:
        Row sample size when a fresh report must be computed.
    max_columns:
        Column cap when a fresh report must be computed.
    blocking:
        Serve on the current thread until interrupted.

    Returns
    -------
    ~buildml.dashboard.launch.EDAAppHandle
        A handle exposing ``url``, ``is_running``, and ``stop()``.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        ``buildml[dashboard]`` is not installed.
    ~buildml.core.errors.ValidationError
        No dataset is attached and no report was supplied.

    See Also
    --------
    Session.eda_app : The method this delegates to.
    """
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
