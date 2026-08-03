"""Draw the charts an EDA plan asked for, and never fail the whole report.

The rendering half of the plan-then-render split. The planner in
:mod:`buildml.eda.adaptive` decided what is worth drawing; this draws it.

Two things shape the implementation. Matplotlib is forced onto the ``Agg``
backend, because a report generated on a server or in CI must not try to open a
window: that hangs, or crashes, depending on the environment. And every plot is
wrapped: a figure that fails records its error and the rest still render. A
report missing one chart is useful; an exception on chart nineteen of
twenty-four wastes everything before it.

Matplotlib and Seaborn are optional extras, imported at call time so the numeric
EDA path never pays for them.

See Also
--------
buildml.eda.adaptive : Deciding what to draw.
buildml.eda.html_report : Embedding the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from buildml.core.errors import MissingExtraError
from buildml.data.dataset import Dataset


def _require_viz() -> tuple[Any, Any]:
    try:
        import matplotlib

        if str(matplotlib.get_backend()).lower() != "agg":
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise MissingExtraError("viz", "EDA visualization") from exc
    sns.set_theme(style="whitegrid", context="notebook")
    return plt, sns


def render_adaptive_plots(
    frame: pd.DataFrame,
    plan: list[dict[str, Any]],
    *,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    """Execute a plot plan, keeping going when an individual chart fails.

    Walks the plan and draws each specification. A chart that raises is recorded
    as an error entry in the results rather than propagating, so one awkward
    column cannot cost you the whole report.

    Parameters
    ----------
    frame:
        The analysis frame: the same one the plan was built against, usually
        sampled. Plotting a different frame will not match the plan's
        assumptions about cardinality and dtype.
    plan:
        Specifications from
        :func:`~buildml.eda.adaptive.build_adaptive_plan`.
    dataset:
        Supplies roles for chart titles, so a target is labelled as one.

    Returns
    -------
    dict
        Keyed by plot identifier. Each value is a Matplotlib figure, or a dict
        with ``error`` and ``spec`` when that chart failed.

    Raises
    ------
    MissingExtraError
        If Matplotlib or Seaborn is not installed. Install with
        ``pip install 'buildml[viz]'``.

    Notes
    -----
    **Check for error entries.** ``isinstance(value, dict)`` distinguishes a
    failure from a figure, and the entry carries both the message and the spec
    that produced it.

    **The figures are open and hold memory.** Save them with
    :func:`save_figures` or close them; a few dozen open figures is a warning
    from Matplotlib and a real allocation.

    **The backend is forced to Agg**, so nothing tries to display. Figures are
    returned as objects for you to save or show deliberately.

    See Also
    --------
    save_figures : Writing them out.
    """
    plt, sns = _require_viz()
    figures: dict[str, Any] = {}
    _ = dataset

    for idx, spec in enumerate(plan):
        kind = spec.get("kind")
        key = f"{idx:02d}_{kind}"
        try:
            if kind == "missingness_matrix":
                figures[key] = _plot_missingness(frame, plt, sns, title=spec.get("title"))
            elif kind == "dtype_overview":
                figures[key] = _plot_dtype_overview(frame, plt, sns, title=spec.get("title"))
            elif kind == "correlation_heatmap":
                figures[key] = _plot_corr(
                    frame,
                    plt,
                    sns,
                    columns=spec.get("columns"),
                    title=spec.get("title"),
                )
            elif kind == "numeric_distribution":
                figures[key] = _plot_numeric_dist(
                    frame, plt, sns, column=spec["column"], title=spec.get("title")
                )
            elif kind == "pair_sample":
                figures[key] = _plot_pair_sample(
                    frame, plt, sns, columns=spec.get("columns", []), title=spec.get("title")
                )
            elif kind in {"categorical_bars", "categorical_topk"}:
                figures[key] = _plot_categorical(
                    frame, plt, sns, column=spec["column"], title=spec.get("title")
                )
            elif kind == "target_balance":
                figures[key] = _plot_target_balance(
                    frame, plt, sns, column=spec["column"], title=spec.get("title")
                )
            elif kind == "target_vs_numeric":
                figures[key] = _plot_target_vs_numeric(
                    frame,
                    plt,
                    sns,
                    feature=spec["feature"],
                    target=spec["target"],
                    title=spec.get("title"),
                )
            elif kind == "target_vs_categorical":
                figures[key] = _plot_target_vs_categorical(
                    frame,
                    plt,
                    sns,
                    feature=spec["feature"],
                    target=spec["target"],
                    title=spec.get("title"),
                )
            elif kind == "temporal_density":
                figures[key] = _plot_temporal(
                    frame, plt, sns, column=spec["column"], title=spec.get("title")
                )
            elif kind == "outlier_board":
                figures[key] = _plot_outlier_board(frame, plt, sns, title=spec.get("title"))
        except Exception as exc:  # noqa: BLE001 - keep EDA resilient
            figures[key] = {"error": str(exc), "spec": spec}
    return figures


def render_analysis_plots(sections: dict[str, Any]) -> dict[str, Any]:
    """Draw the charts that need the analysis, not the data.

    Some charts visualise results rather than rows: a PCA scree plot, a
    correlation heatmap, a drift summary. They are built from the analyzer
    sections, which means they can be produced without the frame at all.

    That matters more than it sounds. The analyzer output is small and
    serialisable, so these charts can be regenerated from a saved report long
    after the data has gone, or on a machine that never had access to it.

    Parameters
    ----------
    sections:
        The analyzer outputs, as held on an
        :class:`~buildml.eda.report.EDAReport`. Sections that are absent simply
        produce no chart.

    Returns
    -------
    dict
        Keyed by chart identifier, holding Matplotlib figures. Charts whose
        input section was missing or empty are absent from the result.

    Raises
    ------
    MissingExtraError
        If Matplotlib or Seaborn is not installed.

    Notes
    -----
    **A missing key means the section was empty**, not that the chart failed.
    No PCA in the multivariate section, no scree plot.

    **The figures are open.** Save or close them.

    See Also
    --------
    render_adaptive_plots : Charts drawn from the rows themselves.
    """
    plt, sns = _require_viz()
    figures: dict[str, Any] = {}
    pca = (sections.get("multivariate") or {}).get("pca") or {}
    variance = pca.get("explained_variance_ratio") or []
    if variance:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            components = list(range(1, len(variance) + 1))
            sns.barplot(x=components, y=variance, ax=ax, color="#2a6f97")
            ax.plot(
                [component - 1 for component in components],
                pca.get("cumulative_explained_variance") or [],
                marker="o",
                color="#c24d00",
                label="Cumulative",
            )
            ax.set_xlabel("Principal component")
            ax.set_ylabel("Explained variance ratio")
            ax.set_title("PCA explained variance")
            ax.legend()
            fig.tight_layout()
            figures["analysis_pca"] = fig
        except Exception as exc:  # noqa: BLE001
            figures["analysis_pca"] = {"error": str(exc)}

    outlier_rows = (sections.get("outliers") or {}).get("per_column") or {}
    if outlier_rows:
        try:
            rates = pd.Series(
                {
                    column: values.get("iqr_outlier_rate", 0)
                    for column, values in outlier_rows.items()
                }
            ).sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(x=rates.values, y=rates.index, ax=ax, color="#8a5a44")
            ax.set_xlabel("IQR outlier rate")
            ax.set_title("Univariate outlier screen")
            fig.tight_layout()
            figures["analysis_outliers"] = fig
        except Exception as exc:  # noqa: BLE001
            figures["analysis_outliers"] = {"error": str(exc)}

    drift = sections.get("drift") or {}
    drift_rows = [
        *list(drift.get("numeric_drift") or []),
        *list(drift.get("categorical_drift") or []),
    ]
    if drift.get("available") and drift_rows:
        try:
            labels = [str(row["column"]) for row in drift_rows[:20]]
            values = [
                row.get("ks_stat", row.get("js_divergence", 0))
                for row in drift_rows[:20]
            ]
            colors = ["#a62b2b" if row.get("flag") else "#4f7c75" for row in drift_rows[:20]]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh(labels[::-1], values[::-1], color=colors[::-1])
            ax.set_xlabel("KS statistic or Jensen-Shannon divergence")
            ax.set_title("Train/test drift screen")
            fig.tight_layout()
            figures["analysis_drift"] = fig
        except Exception as exc:  # noqa: BLE001
            figures["analysis_drift"] = {"error": str(exc)}
    return figures


def _plot_missingness(frame: pd.DataFrame, plt: Any, sns: Any, title: str | None) -> Any:
    fig, ax = plt.subplots(figsize=(10, 4))
    miss = frame.isna().mean().sort_values(ascending=False)
    labels = miss.index.astype(str)
    sns.barplot(x=miss.values, y=labels, hue=labels, ax=ax, palette="crest", legend=False)
    ax.set_xlabel("Missing rate")
    ax.set_title(title or "Missingness")
    fig.tight_layout()
    return fig


def _plot_dtype_overview(frame: pd.DataFrame, plt: Any, sns: Any, title: str | None) -> Any:
    kinds = frame.dtypes.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = kinds.index.astype(str)
    sns.barplot(x=kinds.values, y=labels, hue=labels, ax=ax, palette="mako", legend=False)
    ax.set_xlabel("Column count")
    ax.set_title(title or "Dtype overview")
    fig.tight_layout()
    return fig


def _plot_corr(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    columns: list[str] | None,
    title: str | None,
) -> Any:
    cols = columns or list(frame.select_dtypes(include="number").columns.astype(str))
    corr = frame[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=False)
    ax.set_title(title or "Correlation")
    fig.tight_layout()
    return fig


def _plot_numeric_dist(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    column: str,
    title: str | None,
) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sns.histplot(frame[column].dropna(), kde=True, ax=axes[0], color="#2a6f97")
    axes[0].set_title("Histogram + KDE")
    sns.boxplot(x=frame[column], ax=axes[1], color="#61a5c2")
    axes[1].set_title("Box")
    fig.suptitle(title or column)
    fig.tight_layout()
    return fig


def _plot_pair_sample(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    columns: list[str],
    title: str | None,
) -> Any:
    cols = [c for c in columns if c in frame.columns][:4]
    if len(cols) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need ≥2 numeric columns", ha="center")
        return fig
    complete = frame[cols].dropna()
    if complete.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No complete rows for numeric pairs", ha="center")
        return fig
    grid = sns.pairplot(
        complete.sample(n=min(400, len(complete)), random_state=0)
    )
    grid.figure.suptitle(title or "Pair relationships", y=1.02)
    return grid.figure


def _plot_categorical(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    column: str,
    title: str | None,
) -> Any:
    counts = frame[column].astype(str).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        hue=counts.index,
        ax=ax,
        palette="flare",
        legend=False,
    )
    ax.set_title(title or column)
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


def _plot_target_balance(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    column: str,
    title: str | None,
) -> Any:
    counts = frame[column].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        ax=ax,
        palette="Spectral",
        legend=False,
    )
    ax.set_title(title or "Target balance")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _plot_target_vs_numeric(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    feature: str,
    target: str,
    title: str | None,
) -> Any:
    fig, ax = plt.subplots(figsize=(8, 4))
    if frame[target].nunique(dropna=True) <= 12:
        sns.violinplot(
            data=frame,
            x=target,
            y=feature,
            hue=target,
            ax=ax,
            palette="coolwarm",
            legend=False,
        )
    else:
        sns.scatterplot(data=frame, x=feature, y=target, ax=ax, alpha=0.5)
    ax.set_title(title or f"{feature} vs {target}")
    fig.tight_layout()
    return fig


def _plot_target_vs_categorical(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    feature: str,
    target: str,
    title: str | None,
) -> Any:
    fig, ax = plt.subplots(figsize=(9, 4))
    ct = pd.crosstab(frame[feature].astype(str), frame[target].astype(str), normalize="index")
    ct.head(15).plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title(title or f"{target} composition by {feature}")
    ax.set_ylabel("Rate")
    fig.tight_layout()
    return fig


def _plot_temporal(
    frame: pd.DataFrame,
    plt: Any,
    sns: Any,
    column: str,
    title: str | None,
) -> Any:
    s = pd.to_datetime(frame[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.histplot(s, bins=40, ax=ax, color="#014f86")
    ax.set_title(title or column)
    fig.tight_layout()
    return fig


def _plot_outlier_board(frame: pd.DataFrame, plt: Any, sns: Any, title: str | None) -> Any:
    numeric = frame.select_dtypes(include="number")
    cols = list(numeric.columns.astype(str))[:8]
    if not cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric columns", ha="center")
        return fig
    melted = numeric[cols].melt(var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=melted,
        x="feature",
        y="value",
        hue="feature",
        ax=ax,
        palette="icefire",
        legend=False,
    )
    ax.set_title(title or "Outlier board")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def save_figures(figures: dict[str, Any], directory: str | Path) -> Path:
    """Write figures to a directory as PNGs, skipping whatever cannot be saved.

    Each figure becomes ``{key}.png`` at 140 DPI with tight bounding boxes :
    sharp enough to read on a high-resolution screen and in print, without the
    file sizes that come from going higher.

    Error entries from a failed render are skipped, as is anything that fails to
    save. Consistent with the rest of the visualization path: a partial set of
    figures beats an exception that loses all of them.

    Parameters
    ----------
    figures:
        The figures, as returned by :func:`render_adaptive_plots` or
        :func:`render_analysis_plots`. Error entries are ignored.
    directory:
        Where to write. Created if absent, including parents.

    Returns
    -------
    Path
        The directory written to.

    Raises
    ------
    OSError
        If the directory itself cannot be created. Individual file failures are
        skipped rather than raised.

    Notes
    -----
    **Existing files with the same names are overwritten.**

    **Figures are not closed afterwards.** They remain open and holding memory;
    close them if you are rendering many in a loop.

    **Silent skips are possible.** A figure that fails to save produces no file
    and no error, so compare the directory contents against the keys you passed
    if completeness matters.

    See Also
    --------
    render_adaptive_plots : Producing the figures.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for key, fig in figures.items():
        if fig is None or isinstance(fig, dict):
            continue
        out = root / f"{key}.png"
        try:
            fig.savefig(out, dpi=140, bbox_inches="tight")
        except Exception:  # noqa: BLE001
            continue
    return root
