"""Publication-oriented Plotly figure builders for the EDA app."""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from buildml.core.errors import MissingExtraError
from buildml.dashboard.serialize import flagged_column_names

ThemeName = Literal["light", "dark"]

# Cool technical palette: not default Plotly blue/orange candy.
_PALETTE_LIGHT: dict[str, Any] = {
    "accent": "#0B6E4F",
    "accent_soft": "#3FAE7F",
    "ink": "#1C2430",
    "muted": "#5B6775",
    "warn": "#B45309",
    "critical": "#B91C1C",
    "info": "#1D4E89",
    "grid": "#D7DEE7",
    "hover_bg": "#FFFFFF",
    "hover_ink": "#1C2430",
    "annotation": "#5B6775",
    "series": [
        "#0B6E4F",
        "#1D4E89",
        "#B45309",
        "#6D28D9",
        "#0F766E",
        "#9F1239",
        "#334155",
        "#A16207",
    ],
    "heatmap": [
        [0.0, "#1D4E89"],
        [0.5, "#F8FAFC"],
        [1.0, "#0B6E4F"],
    ],
    "gauge_steps": [
        {"range": [0, 5], "color": "#E8F5EF"},
        {"range": [5, 10], "color": "#FEF3C7"},
    ],
    "gauge_hot": "#FEE2E2",
}

_PALETTE_DARK: dict[str, Any] = {
    "accent": "#3DDC97",
    "accent_soft": "#2A9B6E",
    "ink": "#E8EEF5",
    "muted": "#9AA8B8",
    "warn": "#F0A35A",
    "critical": "#F07178",
    "info": "#7DB4F0",
    "grid": "#2A3340",
    "hover_bg": "#171C24",
    "hover_ink": "#E8EEF5",
    "annotation": "#9AA8B8",
    "series": [
        "#3DDC97",
        "#7DB4F0",
        "#F0A35A",
        "#C4B5FD",
        "#5EEAD4",
        "#FB7185",
        "#94A3B8",
        "#FBBF24",
    ],
    "heatmap": [
        [0.0, "#7DB4F0"],
        [0.5, "#1D2430"],
        [1.0, "#3DDC97"],
    ],
    "gauge_steps": [
        {"range": [0, 5], "color": "#163528"},
        {"range": [5, 10], "color": "#3A2716"},
    ],
    "gauge_hot": "#3A1719",
}

_PALETTES: dict[str, dict[str, Any]] = {"light": _PALETTE_LIGHT, "dark": _PALETTE_DARK}
PALETTE: dict[str, Any] = _PALETTE_LIGHT

DOMAIN_CHART_IDS: dict[str, list[str]] = {
    "cockpit": ["severity_map", "missing_rates", "role_summary", "mi_vs_target"],
    "quality": ["missing_rates", "role_summary"],
    "features": [
        "skew_profile",
        "quartile_spread",
        "normality_flags",
        "cardinality_entropy",
    ],
    "relationships": [
        "mi_vs_target",
        "correlation_heatmap",
        "spearman_heatmap",
        "cramers_v_bars",
    ],
    "multivariate": ["vif_bars", "pca_variance", "correlation_heatmap", "spearman_heatmap"],
    "target": ["target_balance", "drift_flags"],
    "outliers": [
        "outlier_rates",
        "outlier_bounds",
        "zscore_outlier_rates",
        "multivariate_anomaly",
    ],
    "visuals": [
        "severity_map",
        "missing_rates",
        "skew_profile",
        "quartile_spread",
        "mi_vs_target",
        "vif_bars",
        "pca_variance",
        "target_balance",
        "drift_flags",
        "outlier_rates",
        "outlier_bounds",
        "multivariate_anomaly",
        "correlation_heatmap",
        "spearman_heatmap",
        "cramers_v_bars",
    ],
    "briefing": [
        "severity_map",
        "missing_rates",
        "target_balance",
        "mi_vs_target",
        "outlier_rates",
        "correlation_heatmap",
        "spearman_heatmap",
    ],
}


def charts_for_domain(domain_key: str) -> list[str]:
    """Look up which charts belong on a board.

    The board-to-chart mapping lives in one table, so the web studio, the
    offline export, and the PDF briefing all show the same charts on the same
    board. Three separate lists would eventually disagree.

    Parameters
    ----------
    domain_key:
        The board key, from :mod:`buildml.dashboard.domains`.

    Returns
    -------
    list of str
        Chart ids in display order. Empty for an unknown key, which lets a board
        exist before its charts do.

    Notes
    -----
    **A fresh list every call**, so a caller can reorder or filter it without
    editing the registry.

    **A chart can appear on several boards.** Correlation heatmaps show up under
    both relationships and multivariate, because both questions want them.

    Examples
    --------
    >>> isinstance(charts_for_domain("quality"), list)
    True
    >>> charts_for_domain("no-such-board")
    []

    See Also
    --------
    build_chart_figures : Rendering them.
    """
    return list(DOMAIN_CHART_IDS.get(domain_key, []))


@contextmanager
def theme_palette(theme: ThemeName | str = "light") -> Iterator[dict[str, Any]]:
    """Swap the chart palette for the duration of a block, then put it back.

    Plotly figures bake their colours in at construction, so the palette has to
    be active while the figures are built rather than when they are displayed.
    A context manager makes that lifetime explicit and guarantees the previous
    palette is restored even if building raises.

    Parameters
    ----------
    theme:
        ``'light'`` or ``'dark'``. An unrecognised name falls back to light
        rather than raising, so a typo produces a readable chart.

    Yields
    ------
    dict
        The active palette, for a caller that needs a colour outside the chart
        builders.

    Notes
    -----
    **The palette is module-global, so this is not thread-safe.** Two threads
    building charts in different themes at once will interfere. The dashboard
    builds charts on one thread; if you parallelise, do not mix themes.

    Examples
    --------
    ::

        with theme_palette("dark"):
            figures = build_chart_figures(report)

    See Also
    --------
    build_chart_figures : The usual caller, which handles this for you.
    """
    global PALETTE
    previous = PALETTE
    PALETTE = _PALETTES.get(str(theme), _PALETTE_LIGHT)
    try:
        yield PALETTE
    finally:
        PALETTE = previous


def _require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise MissingExtraError("dashboard", "EDA dashboard charts") from exc
    return go, make_subplots


def _layout(fig: Any, *, title: str, height: int = 420) -> Any:
    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 16, "color": PALETTE["ink"]},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Helvetica Neue, sans-serif", "color": PALETTE["ink"]},
        margin={"l": 56, "r": 28, "t": 56, "b": 64},
        height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        colorway=PALETTE["series"],
        hoverlabel={
            "bgcolor": PALETTE["hover_bg"],
            "font": {"size": 12, "color": PALETTE["hover_ink"]},
            "bordercolor": PALETTE["grid"],
        },
        meta={"buildml_theme": "dark" if PALETTE is _PALETTE_DARK else "light"},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        tickfont={"color": PALETTE["ink"]},
        title_font={"color": PALETTE["ink"]},
        linecolor=PALETTE["grid"],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        tickfont={"color": PALETTE["ink"]},
        title_font={"color": PALETTE["ink"]},
        linecolor=PALETTE["grid"],
    )
    return fig


def _footnote(fig: Any, text: str, *, y: float = -0.22) -> None:
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0,
        y=y,
        showarrow=False,
        font={"size": 11, "color": PALETTE["annotation"]},
        align="left",
    )


def build_chart_figures(
    report: dict[str, Any],
    *,
    theme: ThemeName | str = "light",
) -> dict[str, Any]:
    """Build every dashboard chart from the report, in one theme.

    Renders the full catalogue in a single pass, all under one palette, so the
    set is visually consistent. Each chart reads only the report sections it
    needs and produces an empty-state figure when they are absent: a chart that
    says "no drift analysis was run" rather than a hole in the layout.

    Parameters
    ----------
    report:
        The report as a dict, from
        :meth:`~buildml.eda.report.EDAReport.to_dict`.
    theme:
        ``'light'`` or ``'dark'``.

    Returns
    -------
    dict
        Plotly ``Figure`` objects by chart id. Every id in the catalogue is
        present, including the ones that rendered as empty states.

    Raises
    ------
    MissingExtraError
        If Plotly is not installed. Install with
        ``pip install 'buildml[dashboard]'``.

    Notes
    -----
    **All charts are built, not just the ones you are about to show.** That is
    the right trade for the studio, which navigates between boards without a
    round trip, and wasteful if you want one chart. There is no per-chart entry
    point today.

    **Figure objects are live and mutable**: adjust a title or an axis before
    display if you need to.

    See Also
    --------
    build_chart_catalog : The same charts as JSON.
    render_chart_png : Rasterising one.
    """
    go, make_subplots = _require_plotly()
    with theme_palette(theme):
        figures: dict[str, Any] = {
            "severity_map": _fig_severity(go, report),
            "missing_rates": _fig_missing(go, report),
            "mi_vs_target": _fig_mi(go, report),
            "vif_bars": _fig_vif(go, report),
            "pca_variance": _fig_pca(go, report),
            "target_balance": _fig_target(go, report),
            "drift_flags": _fig_drift(go, report),
            "outlier_rates": _fig_outliers(go, report),
            "outlier_bounds": _fig_outlier_bounds(go, report),
            "zscore_outlier_rates": _fig_zscore_outliers(go, report),
            "multivariate_anomaly": _fig_multivariate_anomaly(go, report),
            "skew_profile": _fig_skew(go, report),
            "cardinality_entropy": _fig_cardinality(go, report),
            "normality_flags": _fig_normality(go, report),
            "quartile_spread": _fig_quartile_spread(go, report),
            "correlation_heatmap": _fig_corr(go, report, method="pearson"),
            "spearman_heatmap": _fig_corr(go, report, method="spearman"),
            "cramers_v_bars": _fig_cramers(go, report),
            "role_summary": _fig_roles(go, report),
        }
    _ = make_subplots
    return figures


def build_chart_catalog(
    report: dict[str, Any],
    *,
    theme: ThemeName | str = "light",
) -> dict[str, dict[str, Any]]:
    """Build the charts and serialise them for the browser.

    Plotly renders in the browser from a JSON description, so the server's job
    is to produce that description rather than an image. The charts stay
    interactive: hover, zoom, legend toggling: and no rasterisation happens
    server-side.

    Parameters
    ----------
    report:
        The report as a dict.
    theme:
        ``'light'`` or ``'dark'``.

    Returns
    -------
    dict
        Chart id to Plotly JSON, ready to embed in a page or return from an
        endpoint.

    Raises
    ------
    MissingExtraError
        If Plotly is not installed.

    Notes
    -----
    **This is not small.** A heatmap over 50 columns carries 2,500 values, and
    the whole catalogue can run to several megabytes: which is what makes the
    offline export a large file.

    See Also
    --------
    build_chart_figures : The figure objects instead.
    """
    return {key: _as_json(fig) for key, fig in build_chart_figures(report, theme=theme).items()}


def render_chart_png(
    fig: Any,
    *,
    width: int = 900,
    height: int | None = None,
    scale: float = 2.0,
) -> bytes | None:
    """Turn a figure into PNG bytes, or return ``None`` and let the caller cope.

    PDF export needs raster images, which means Kaleido: a headless browser
    that renders Plotly figures server-side. It is a heavy optional dependency
    and it fails in ways that are hard to predict: missing system libraries,
    sandboxed environments, containers without the right shared objects.

    So this never raises. A missing or broken Kaleido yields ``None``, and the
    PDF builder emits a placeholder instead of a chart. A briefing with one
    missing figure is worth more than an exception.

    Parameters
    ----------
    fig:
        A Plotly figure.
    width:
        Output width in pixels before scaling.
    height:
        Output height. Defaults to the figure's own layout height, or 420.
    scale:
        Resolution multiplier. 2.0 gives a sharp image on high-density displays
        and in print, at four times the bytes.

    Returns
    -------
    bytes or None
        PNG data, or ``None`` if Kaleido is unavailable or rendering failed.

    Notes
    -----
    **A ``None`` return hides the reason.** Both "not installed" and "installed
    but crashed" look identical here. If figures are silently missing from a
    PDF, test Kaleido directly.

    **Rasterising is slow**: a browser process per figure. A full catalogue
    takes seconds to tens of seconds.

    See Also
    --------
    buildml.dashboard.exports.export_pdf : The consumer.
    """
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return None
    try:
        layout_height = None
        if hasattr(fig, "layout"):
            layout_height = getattr(fig.layout, "height", None)
        out_height = height or int(layout_height or 420)
        buffer = io.BytesIO()
        fig.write_image(buffer, format="png", width=width, height=out_height, scale=scale)
        return buffer.getvalue()
    except Exception:
        return None


def _as_json(fig: Any) -> dict[str, Any]:
    return fig.to_plotly_json()

def _numeric_per_column(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    uni = report.get("univariate") or {}
    per_column = uni.get("per_column") or {}
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(per_column, dict):
        for name, stats in per_column.items():
            if isinstance(stats, dict) and str(stats.get("kind", "numeric")) == "numeric":
                rows.append((str(name), stats))
    return rows


def _categorical_per_column(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    uni = report.get("univariate") or {}
    per_column = uni.get("per_column") or {}
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(per_column, dict):
        for name, stats in per_column.items():
            if isinstance(stats, dict) and str(stats.get("kind")) == "categorical":
                rows.append((str(name), stats))
    return rows


def _outlier_per_column(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    outliers = report.get("outliers") or {}
    per_column = outliers.get("per_column") or outliers.get("univariate") or {}
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(per_column, dict):
        for name, stats in per_column.items():
            if isinstance(stats, dict):
                rows.append((str(name), stats))
    return rows


def _fig_severity(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    findings = report.get("findings") or []
    order = ["info", "low", "medium", "high", "critical"]
    counts = {key: 0 for key in order}
    for item in findings:
        severity = str(item.get("severity", "info")).lower()
        counts[severity] = counts.get(severity, 0) + 1
    colors = {
        "info": PALETTE["info"],
        "low": PALETTE["accent_soft"],
        "medium": PALETTE["warn"],
        "high": "#C2410C",
        "critical": PALETTE["critical"],
    }
    fig = go.Figure(
        data=[
            go.Bar(
                x=order,
                y=[counts[key] for key in order],
                marker_color=[colors[key] for key in order],
                text=[counts[key] for key in order],
                textposition="outside",
                hovertemplate="Severity=%{x}<br>Findings=%{y}<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Findings", rangemode="tozero")
    fig.update_xaxes(title_text="Severity")
    _layout(fig, title="Finding severity map", height=360)
    _footnote(fig, "Severity ranks workflow impact, not visual emphasis.")
    return fig


def _fig_missing(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    quality = report.get("quality") or {}
    rates = quality.get("missing_rate_by_column") or {}
    items = sorted(
        ((str(k), float(v)) for k, v in rates.items() if float(v) > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    if not items:
        return _empty_fig(go, "Missingness by column", "No missing values observed.")
    labels, values = zip(*items, strict=True)
    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(labels),
            orientation="h",
            marker_color=PALETTE["warn"],
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Missing rate", tickformat=".0%")
    fig.update_yaxes(autorange="reversed", title_text="")
    _layout(fig, title="Top missing rates (full frame)", height=max(360, 24 * len(labels) + 120))
    return fig


def _fig_mi(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    mi = (report.get("bivariate") or {}).get("mutual_information_vs_target") or {}
    scored = []
    for key, value in mi.items():
        try:
            scored.append((str(key), float(value)))
        except (TypeError, ValueError):
            if isinstance(value, dict) and value.get("score") is not None:
                scored.append((str(key), float(value["score"])))
    scored.sort(key=lambda item: item[1], reverse=True)
    scored = scored[:15]
    if not scored:
        return _empty_fig(go, "Mutual information vs target", "Target or features unavailable.")
    labels, values = zip(*scored, strict=True)
    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(labels),
            orientation="h",
            marker_color=PALETTE["accent"],
            hovertemplate="%{y}<br>MI=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Mutual information (association, not causation)")
    fig.update_yaxes(autorange="reversed")
    _layout(fig, title="Mutual information vs target", height=max(380, 24 * len(labels) + 120))
    _footnote(fig, "High MI is a review cue for usefulness or leakage-like proxies.")
    return fig


def _fig_vif(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    rows = (report.get("multivariate") or {}).get("vif") or []
    cleaned = [
        (str(row.get("column")), float(row.get("vif")))
        for row in rows
        if isinstance(row, dict) and row.get("vif") is not None
    ]
    cleaned.sort(key=lambda item: item[1], reverse=True)
    cleaned = cleaned[:15]
    if not cleaned:
        return _empty_fig(go, "Variance inflation factors", "VIF was not available.")
    labels, values = zip(*cleaned, strict=True)
    colors = [PALETTE["critical"] if v >= 5 else PALETTE["info"] for v in values]
    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(labels),
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>VIF=%{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=5,
        line_dash="dot",
        line_color=PALETTE["warn"],
        annotation_text="review flag (VIF=5)",
        annotation_position="top",
    )
    fig.update_xaxes(title_text="VIF")
    fig.update_yaxes(autorange="reversed")
    _layout(fig, title="Variance inflation factors", height=max(380, 24 * len(labels) + 140))
    return fig


def _fig_pca(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    pca = (report.get("multivariate") or {}).get("pca") or {}
    ratios = pca.get("explained_variance_ratio") or []
    if not ratios:
        return _empty_fig(go, "PCA explained variance", "PCA summary unavailable.")
    xs = [f"PC{i + 1}" for i in range(len(ratios))]
    cumulative = []
    total = 0.0
    for value in ratios:
        total += float(value)
        cumulative.append(total)
    fig = go.Figure()
    fig.add_bar(
        x=xs,
        y=[float(v) for v in ratios],
        name="Component share",
        marker_color=PALETTE["info"],
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    )
    fig.add_scatter(
        x=xs,
        y=cumulative,
        name="Cumulative",
        mode="lines+markers",
        line={"color": PALETTE["accent"], "width": 2.5},
        hovertemplate="Cumulative %{y:.1%}<extra></extra>",
    )
    fig.update_yaxes(title_text="Explained variance", tickformat=".0%")
    _layout(fig, title="PCA explained variance", height=400)
    return fig


def _fig_target(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    target = report.get("target") or {}
    balance = target.get("class_balance") or target.get("balance") or {}
    if isinstance(balance, dict) and balance:
        labels = [str(k) for k in balance]
        values = [float(balance[k]) for k in balance]
    else:
        counts = target.get("value_counts") or {}
        if not counts:
            return _empty_fig(go, "Target balance", "No classification balance available.")
        labels = [str(k) for k in counts]
        values = [float(counts[k]) for k in counts]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=PALETTE["series"][: len(labels)],
            hovertemplate="Class=%{x}<br>Value=%{y}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Count or rate", rangemode="tozero")
    _layout(fig, title="Target class balance", height=380)
    return fig


def _fig_drift(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    drift = report.get("drift") or {}
    if not drift.get("available"):
        return _empty_fig(go, "Train/test drift", "Drift requires a Session split.")
    scores = drift.get("scores") or drift.get("column_scores") or {}
    items: list[tuple[str, float]] = []
    if isinstance(scores, dict) and scores:
        items = sorted(
            ((str(k), float(v)) for k, v in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:15]
    else:
        for row in drift.get("numeric_drift") or []:
            if isinstance(row, dict) and row.get("column") is not None:
                try:
                    items.append((str(row["column"]), float(row.get("ks_stat") or 0.0)))
                except (TypeError, ValueError):
                    continue
        for row in drift.get("categorical_drift") or []:
            if isinstance(row, dict) and row.get("column") is not None:
                try:
                    items.append((str(row["column"]), float(row.get("js_divergence") or 0.0)))
                except (TypeError, ValueError):
                    continue
        items.sort(key=lambda item: item[1], reverse=True)
        items = items[:15]
        if not items:
            flagged = flagged_column_names(drift.get("flagged_columns"))
            items = [(name, 1.0) for name in flagged[:15]]
    if not items:
        return _empty_fig(go, "Train/test drift", "No drift flags or scores reported.")
    labels, values = zip(*items, strict=True)
    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(labels),
            orientation="h",
            marker_color=PALETTE["critical"],
            hovertemplate="%{y}<br>score=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Drift score / flag intensity")
    _layout(fig, title="Train/test drift signals", height=max(360, 24 * len(labels) + 120))
    return fig


def _fig_outliers(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    items: list[tuple[str, float, int]] = []
    for name, stats in _outlier_per_column(report):
        rate = stats.get("iqr_outlier_rate")
        if rate is None:
            rate = stats.get("rate") or stats.get("outlier_rate")
        if rate is None:
            continue
        count = int(stats.get("iqr_outlier_count") or stats.get("count") or 0)
        items.append((name, float(rate), count))
    items.sort(key=lambda item: item[1], reverse=True)
    items = items[:15]
    if not items:
        return _empty_fig(go, "IQR outlier rates", "No univariate IQR outlier rates found.")
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    counts = [item[2] for item in items]
    colors = [
        PALETTE["critical"] if rate >= 0.05 else PALETTE["warn"] if rate > 0 else PALETTE["accent"]
        for rate in values
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            customdata=counts,
            text=[f"{rate:.1%}" for rate in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>IQR outlier rate=%{x:.1%}<br>count=%{customdata}<extra></extra>",
        )
    )
    fig.add_vline(
        x=0.05,
        line_dash="dot",
        line_color=PALETTE["critical"],
        annotation_text="5% review flag",
        annotation_position="top",
    )
    fig.update_yaxes(autorange="reversed")
    xmax = max(0.05, max(values) * 1.25 if values else 0.05)
    fig.update_xaxes(
        title_text="IQR outlier rate (1.5×IQR rule)",
        tickformat=".0%",
        range=[0, xmax],
    )
    _layout(fig, title="Univariate IQR outlier screens", height=max(380, 24 * len(labels) + 140))
    _footnote(fig, "Flags are review candidates, not automatic deletion criteria.")
    return fig


def _fig_outlier_bounds(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    items: list[tuple[str, float, float, float]] = []
    for name, stats in _outlier_per_column(report):
        bounds = stats.get("iqr_bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        try:
            lower, upper = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError):
            continue
        if upper < lower:
            continue
        mid = (lower + upper) / 2.0
        items.append((name, lower, upper, mid))
    items.sort(key=lambda item: abs(item[2] - item[1]), reverse=True)
    items = items[:12]
    if not items:
        return _empty_fig(go, "IQR fences by column", "IQR bounds unavailable.")
    labels = [item[0] for item in items]
    lowers = [item[1] for item in items]
    uppers = [item[2] for item in items]
    mids = [item[3] for item in items]
    fig = go.Figure()
    fig.add_scatter(
        x=mids,
        y=labels,
        mode="markers",
        marker={"size": 9, "color": PALETTE["accent"]},
        error_x={
            "type": "data",
            "symmetric": False,
            "array": [u - m for u, m in zip(uppers, mids, strict=True)],
            "arrayminus": [m - lo for m, lo in zip(mids, lowers, strict=True)],
            "color": PALETTE["info"],
            "thickness": 2.2,
            "width": 6,
        },
        customdata=list(zip(lowers, uppers, strict=True)),
        hovertemplate=(
            "%{y}<br>lower=%{customdata[0]:.4g}<br>upper=%{customdata[1]:.4g}<extra></extra>"
        ),
        name="IQR fence",
    )
    fig.update_yaxes(autorange="reversed", title_text="")
    fig.update_xaxes(title_text="Value scale (IQR fence midpoint ± bounds)")
    _layout(fig, title="IQR fences by feature", height=max(380, 28 * len(labels) + 140))
    _footnote(fig, "Fence width reflects robust scale; compare with domain units.")
    return fig


def _fig_zscore_outliers(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    items: list[tuple[str, float, int]] = []
    for name, stats in _outlier_per_column(report):
        rate = stats.get("zscore_abs_gt_3_rate")
        if rate is None:
            continue
        count = int(stats.get("zscore_abs_gt_3") or 0)
        items.append((name, float(rate), count))
    items.sort(key=lambda item: item[1], reverse=True)
    items = items[:15]
    if not items:
        return _empty_fig(go, "|z| > 3 rates", "No z-score outlier rates found.")
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    counts = [item[2] for item in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=PALETTE["info"],
            customdata=counts,
            text=[f"{rate:.1%}" for rate in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>|z|>3 rate=%{x:.1%}<br>count=%{customdata}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    xmax = max(0.05, max(values) * 1.25 if values else 0.05)
    fig.update_xaxes(
        title_text="Share of rows with |z-score| > 3",
        tickformat=".0%",
        range=[0, xmax],
    )
    _layout(fig, title="Gaussian |z| > 3 screens", height=max(360, 24 * len(labels) + 120))
    _footnote(fig, "Gaussian z-scores are brittle for skewed or heavy-tailed columns.")
    return fig


def _fig_multivariate_anomaly(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    multi = (report.get("outliers") or {}).get("multivariate") or {}
    if not isinstance(multi, dict) or not multi:
        return _empty_fig(
            go,
            "Multivariate anomaly screen",
            "IsolationForest screen needs ≥2 numeric columns and ≥30 complete rows.",
        )
    rate = float(multi.get("anomaly_rate") or 0.0)
    count = int(multi.get("anomaly_count") or 0)
    n_rows = int(multi.get("n_rows_scored") or 0)
    method = str(multi.get("method") or "isolation_forest")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=rate * 100.0,
            number={
                "suffix": "%",
                "font": {"size": 36, "color": PALETTE["ink"]},
            },
            title={
                "text": f"{method.replace('_', ' ').title()} anomaly rate",
                "font": {"size": 14, "color": PALETTE["ink"]},
            },
            delta={"font": {"color": PALETTE["muted"]}},
            gauge={
                "axis": {
                    "range": [0, max(15.0, rate * 100.0 * 1.4)],
                    "tickfont": {"color": PALETTE["ink"]},
                    "tickcolor": PALETTE["grid"],
                },
                "bar": {"color": PALETTE["accent"]},
                "bordercolor": PALETTE["grid"],
                "steps": [
                    *PALETTE["gauge_steps"],
                    {
                        "range": [10, max(15.0, rate * 100.0 * 1.4)],
                        "color": PALETTE["gauge_hot"],
                    },
                ],
                "threshold": {
                    "line": {"color": PALETTE["critical"], "width": 2},
                    "thickness": 0.75,
                    "value": 5,
                },
            },
        )
    )
    _layout(fig, title="Multivariate anomaly summary", height=360)
    _footnote(
        fig,
        f"{count:,} anomalies in {n_rows:,} scored rows · 5% gauge mark is a review cue.",
        y=-0.08,
    )
    return fig


def _fig_skew(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    items: list[tuple[str, float]] = []
    for name, stats in _numeric_per_column(report):
        skew = stats.get("skew")
        if skew is None:
            continue
        items.append((name, float(skew)))
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    items = items[:15]
    if not items:
        return _empty_fig(go, "Skewness profile", "No numeric skew values available.")
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    colors = [
        PALETTE["critical"]
        if abs(v) >= 1.5
        else PALETTE["warn"]
        if abs(v) >= 0.75
        else PALETTE["accent"]
        for v in values
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>skew=%{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=PALETTE["muted"], line_width=1)
    fig.add_vline(x=0.75, line_dash="dot", line_color=PALETTE["warn"])
    fig.add_vline(x=-0.75, line_dash="dot", line_color=PALETTE["warn"])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Sample skewness (analysis frame)")
    _layout(fig, title="Numeric skewness profile", height=max(380, 24 * len(labels) + 140))
    _footnote(fig, "Dashed lines at ±0.75 mark moderate skew review cues.")
    return fig


def _fig_cardinality(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    items: list[tuple[str, int, float]] = []
    for name, stats in _categorical_per_column(report):
        nunique = stats.get("nunique")
        if nunique is None:
            continue
        entropy = float(stats.get("entropy_bits") or 0.0)
        items.append((name, int(nunique), entropy))
    items.sort(key=lambda item: item[1], reverse=True)
    items = items[:15]
    if not items:
        # Fall back to categorical_uniques map when per_column empty.
        uniques = (report.get("univariate") or {}).get("categorical_uniques") or {}
        items = [(str(k), int(v), 0.0) for k, v in uniques.items()]
        items.sort(key=lambda item: item[1], reverse=True)
        items = items[:15]
    if not items:
        return _empty_fig(go, "Categorical cardinality", "No categorical columns profiled.")
    labels = [item[0] for item in items]
    nuniques = [item[1] for item in items]
    entropies = [item[2] for item in items]
    fig = go.Figure()
    fig.add_bar(
        x=nuniques,
        y=labels,
        orientation="h",
        name="nunique",
        marker_color=PALETTE["info"],
        customdata=entropies,
        hovertemplate="%{y}<br>nunique=%{x}<br>entropy=%{customdata:.3f} bits<extra></extra>",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Distinct levels (analysis frame)")
    _layout(fig, title="Categorical cardinality & entropy", height=max(380, 24 * len(labels) + 140))
    _footnote(fig, "High cardinality widens one-hot encodings; inspect rare-level rates.")
    return fig


def _fig_normality(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    items: list[tuple[str, float, str, bool | None]] = []
    for name, stats in _numeric_per_column(report):
        pvalue = stats.get("normality_pvalue")
        if pvalue is None:
            continue
        method = str(stats.get("normality_method") or "normality screen")
        flag = stats.get("appears_non_normal")
        items.append((name, float(pvalue), method, flag if isinstance(flag, bool) else None))
    items.sort(key=lambda item: item[1])
    items = items[:15]
    if not items:
        return _empty_fig(go, "Normality screens", "No normality p-values available.")
    labels = [item[0] for item in items]
    pvalues = [item[1] for item in items]
    methods = [item[2] for item in items]
    colors = [
        PALETTE["critical"] if (flag is True or p < 0.05) else PALETTE["accent"]
        for p, flag in ((item[1], item[3]) for item in items)
    ]
    fig = go.Figure(
        go.Bar(
            x=pvalues,
            y=labels,
            orientation="h",
            marker_color=colors,
            customdata=methods,
            hovertemplate="%{y}<br>p=%{x:.4g}<br>%{customdata}<extra></extra>",
        )
    )
    fig.add_vline(
        x=0.05,
        line_dash="dot",
        line_color=PALETTE["warn"],
        annotation_text="α=0.05",
        annotation_position="top",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Normality screen p-value", type="log")
    _layout(fig, title="Normality screens (log p-scale)", height=max(380, 24 * len(labels) + 140))
    _footnote(fig, "p-values are unadjusted screens; large n makes tiny effects 'significant'.")
    return fig


def _fig_quartile_spread(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    items: list[tuple[str, dict[str, float]]] = []
    for name, stats in _numeric_per_column(report):
        needed = ("q05", "q25", "median", "q75", "q95")
        if any(stats.get(key) is None for key in needed):
            continue
        items.append(
            (
                name,
                {key: float(stats[key]) for key in needed},
            )
        )
    items = items[:12]
    if not items:
        return _empty_fig(go, "Quartile spread", "Numeric quartile summaries unavailable.")
    labels = [item[0] for item in items]
    # Synthetic box traces from analyzer quantiles (no raw samples required).
    fig = go.Figure()
    for index, (name, qs) in enumerate(items):
        fig.add_box(
            name=name,
            y=[name],
            orientation="h",
            q1=[qs["q25"]],
            median=[qs["median"]],
            q3=[qs["q75"]],
            lowerfence=[qs["q05"]],
            upperfence=[qs["q95"]],
            marker_color=PALETTE["series"][index % len(PALETTE["series"])],
            hovertemplate=(
                f"{name}<br>q05=%{{lowerfence}}<br>q25=%{{q1}}<br>"
                "median=%{median}<br>q75=%{q3}<br>q95=%{upperfence}<extra></extra>"
            ),
        )
    fig.update_layout(showlegend=False, boxmode="group")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels)
    fig.update_xaxes(title_text="Observed quantiles (q05–q95)")
    _layout(
        fig,
        title="Numeric quartile spread (analysis frame)",
        height=max(400, 34 * len(labels) + 140),
    )
    _footnote(fig, "Boxes use analyzer quantiles, not re-sampled whiskers from raw rows.")
    return fig


def _fig_corr(
    go: Any,
    report: dict[str, Any],
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> dict[str, Any]:
    bivariate = report.get("bivariate") or {}
    if method == "spearman":
        matrix = bivariate.get("spearman") or bivariate.get("correlation_spearman")
        title = "Spearman correlation (numeric ranks)"
        empty = "Spearman matrix unavailable."
        coeff = "ρ"
    else:
        matrix = (
            bivariate.get("pearson")
            or bivariate.get("correlation_pearson")
            or bivariate.get("correlation")
        )
        title = "Pearson correlation (numeric)"
        empty = "Pearson matrix unavailable."
        coeff = "r"
    if not isinstance(matrix, dict) or not matrix:
        return _empty_fig(go, title, empty)
    # Accept either nested dict or {columns,matrix}
    if "columns" in matrix and "matrix" in matrix:
        columns = [str(c) for c in matrix["columns"]]
        z = matrix["matrix"]
    else:
        columns = [str(c) for c in matrix]
        z = []
        for row_key in columns:
            row = matrix.get(row_key) or {}
            z.append([_safe_float(row.get(col)) for col in columns])
    if len(columns) > 18:
        columns = columns[:18]
        z = [row[:18] for row in z[:18]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=columns,
            y=columns,
            colorscale=PALETTE["heatmap"],
            zmin=-1,
            zmax=1,
            colorbar={
                "title": {"text": coeff, "font": {"color": PALETTE["ink"]}},
                "tickfont": {"color": PALETTE["ink"]},
                "outlinecolor": PALETTE["grid"],
            },
            hovertemplate=f"%{{y}} vs %{{x}}<br>{coeff}=%{{z:.3f}}<extra></extra>",
        )
    )
    _layout(fig, title=title, height=520)
    fig.update_xaxes(tickangle=45, tickfont={"color": PALETTE["ink"]})
    fig.update_yaxes(tickfont={"color": PALETTE["ink"]})
    _footnote(
        fig,
        (
            "Rank association; less sensitive to linear-scale outliers than Pearson."
            if method == "spearman"
            else "Linear association on raw numeric scale; pairwise complete observations."
        ),
        y=-0.18,
    )
    return fig


def _fig_cramers(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    pairs = (report.get("bivariate") or {}).get("categorical_pairs") or []
    scored: list[tuple[str, float]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        value = item.get("cramers_v")
        if value is None:
            continue
        label = f"{item.get('a')} × {item.get('b')}"
        scored.append((label, float(value)))
    scored.sort(key=lambda item: item[1], reverse=True)
    scored = scored[:15]
    if not scored:
        return _empty_fig(
            go,
            "Cramér's V (categorical pairs)",
            "No categorical-pair Cramér's V values available.",
        )
    labels = [item[0] for item in scored]
    values = [item[1] for item in scored]
    colors = [
        PALETTE["critical"] if v >= 0.5 else PALETTE["warn"] if v >= 0.25 else PALETTE["accent"]
        for v in values
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Cramér's V=%{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.25, line_dash="dot", line_color=PALETTE["warn"])
    fig.add_vline(x=0.5, line_dash="dot", line_color=PALETTE["critical"])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Cramér's V (0–1)", range=[0, 1])
    _layout(fig, title="Cramér's V for categorical pairs", height=max(380, 24 * len(labels) + 140))
    _footnote(fig, "0.25 / 0.5 dashed lines are conventional association review cues, not proof.")
    return fig


def _fig_roles(go: Any, report: dict[str, Any]) -> dict[str, Any]:
    overview = report.get("overview") or {}
    role_counts = overview.get("role_counts")
    roles = overview.get("roles") or {}
    labels: list[str]
    values: list[float]
    if isinstance(role_counts, dict) and role_counts:
        labels = [str(key) for key in role_counts]
        values = [float(role_counts[key]) for key in role_counts]
    elif isinstance(roles, dict) and roles:
        # Column -> role name map from Session metadata / overview.
        tallies: dict[str, float] = {}
        sample_value = next(iter(roles.values()), None)
        if isinstance(sample_value, (int, float)) and not isinstance(sample_value, bool):
            labels = [str(key) for key in roles]
            values = [float(roles[key]) for key in roles]
        else:
            for role_name in roles.values():
                key = str(role_name)
                tallies[key] = tallies.get(key, 0.0) + 1.0
            labels = list(tallies)
            values = [tallies[key] for key in labels]
    else:
        eligible = len(overview.get("eligible_feature_columns") or [])
        excluded = len(overview.get("excluded_from_feature_analysis") or [])
        labels = ["eligible features", "excluded from feature analysis"]
        values = [float(eligible), float(excluded)]
    if not any(values):
        return _empty_fig(go, "Role / eligibility summary", "No role summary available.")
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={"colors": PALETTE["series"]},
            hovertemplate="%{label}<br>n=%{value}<br>%{percent}<extra></extra>",
            textinfo="label+value",
        )
    )
    _layout(fig, title="Role / eligibility summary", height=400)
    return fig


def _empty_fig(go: Any, title: str, message: str) -> dict[str, Any]:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": PALETTE["muted"]},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    _layout(fig, title=title, height=300)
    return fig


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
