"""Adaptive high-impact visualization rendering for EDA."""

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
    """Render adaptive plot specs into matplotlib figure objects.

    Parameters
    ----------
    frame:
        Analysis frame (possibly sampled).
    plan:
        Specs from :func:`buildml.eda.adaptive.build_adaptive_plan`.
    dataset:
        Optional dataset for role-aware titles.
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
    """Render plots that use retained analyzer statistics rather than raw rows."""
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
    """Save rendered figures to a directory as PNG files."""
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
