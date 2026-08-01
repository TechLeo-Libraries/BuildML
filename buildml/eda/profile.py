"""Exploratory data analysis orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import MissingExtraError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.eda.adaptive import build_adaptive_plan
from buildml.eda.analyzers.bivariate import analyze_bivariate
from buildml.eda.analyzers.drift import analyze_drift
from buildml.eda.analyzers.multivariate import analyze_multivariate
from buildml.eda.analyzers.outliers import analyze_outliers
from buildml.eda.analyzers.quality import analyze_quality
from buildml.eda.analyzers.target import analyze_target
from buildml.eda.analyzers.univariate import analyze_univariate
from buildml.eda.findings import (
    build_findings,
    build_recommendations,
    narrative_view,
    recommendation_view,
)
from buildml.eda.html_report import export_eda_html
from buildml.eda.report import EDAReport

DEFAULT_ANALYSIS_CAP = 100_000
DEFAULT_COLUMN_CAP = 100
HtmlFormat = Literal["studio", "research"]


def explore_dataset(
    dataset: Dataset,
    *,
    split_plan: SplitPlan | None = None,
    sample_rows: int | None = None,
    max_columns: int = DEFAULT_COLUMN_CAP,
    max_plots: int = 36,
    include_plots: bool = False,
    show: bool = False,
    random_state: int = 42,
    export_html: str | Path | None = None,
    export_figures: str | Path | None = None,
    html_format: HtmlFormat = "studio",
) -> EDAReport:
    """Run the exploratory analysis pipeline.

    Includes quality, univariate diagnostics, bivariate/MI associations,
    multicollinearity/PCA, target-aware tests, outlier screens, train/test
    drift (when split exists), adaptive visualization planning, narrative
    generation, and optional HTML/figure export.

    ``html_format="studio"`` (default) writes an offline Teaching Studio
    snapshot (same product surface as ``session.eda_app()``). Use
    ``html_format="research"`` for the layered research HTML shell with
    matplotlib figure embeds.
    """
    full = dataset._ensure_pandas()
    warnings: list[str] = []
    analysis_cap = DEFAULT_ANALYSIS_CAP if sample_rows is None else sample_rows
    if analysis_cap < 1:
        raise ValueError("sample_rows must be positive or None")
    if max_columns < 1:
        raise ValueError("max_columns must be positive")
    if max_plots < 0:
        raise ValueError("max_plots must be non-negative")
    if len(full) > analysis_cap:
        frame = full.sample(n=analysis_cap, random_state=random_state)
        warnings.append(
            f"Heavy EDA sections used a sample of {analysis_cap:,} rows from {len(full):,}."
        )
    else:
        frame = full

    target_cols = dataset.role_columns(ColumnRole.TARGET)
    target = target_cols[0] if target_cols else None

    overview = _overview(dataset, full, frame)
    quality = analyze_quality(full, frame)
    analysis_columns = _bounded_analysis_columns(dataset, frame, max_columns=max_columns)
    if len(analysis_columns) < len(frame.columns):
        omitted = len(frame.columns) - len(analysis_columns)
        warnings.append(
            f"Column budget limited detailed analyzers to {len(analysis_columns):,} of "
            f"{len(frame.columns):,} columns; {omitted:,} columns remain covered by "
            "dataset-wide quality checks."
        )
    analysis_frame = frame.loc[:, analysis_columns]
    overview["analysis_columns"] = analysis_columns
    overview["analysis_column_count"] = len(analysis_columns)
    overview["analysis_column_budget"] = max_columns
    feature_columns = _eligible_feature_columns(dataset, analysis_frame, quality)
    overview["eligible_feature_columns"] = feature_columns
    exclusion_reasons = _feature_exclusion_reasons(dataset, analysis_frame, quality)
    overview["feature_exclusion_reasons"] = exclusion_reasons
    overview["heuristic_id_exclusions"] = sorted(
        column
        for column, reasons in exclusion_reasons.items()
        if "heuristic identifier-like detection" in reasons
    )
    overview["explicit_role_exclusions"] = sorted(
        column
        for column, reasons in exclusion_reasons.items()
        if any(reason.startswith("explicit role:") for reason in reasons)
    )
    overview["excluded_from_feature_analysis"] = sorted(exclusion_reasons)
    univariate = analyze_univariate(analysis_frame)
    bivariate = analyze_bivariate(analysis_frame, target=target, feature_columns=feature_columns)
    multivariate = analyze_multivariate(analysis_frame, bivariate, feature_columns=feature_columns)
    target_info = analyze_target(dataset, analysis_frame, feature_columns=feature_columns)
    outliers = analyze_outliers(analysis_frame, feature_columns=feature_columns)
    drift = analyze_drift(dataset, split_plan, feature_columns=feature_columns)
    adaptive_plan = build_adaptive_plan(
        dataset,
        analysis_frame,
        feature_columns=feature_columns,
        max_plots=max_plots,
    )
    sections = {
        "overview": overview,
        "quality": quality,
        "target": target_info,
        "bivariate": bivariate,
        "multivariate": multivariate,
        "outliers": outliers,
        "drift": drift,
    }
    findings = build_findings(sections)
    recommendation_details = build_recommendations(findings)
    narrative = narrative_view(findings)
    recommendations = recommendation_view(recommendation_details)

    figures: dict[str, Any] = {}
    figure_dir = None
    figure_paths: dict[str, str] = {}
    needs_matplotlib = (
        include_plots
        or export_figures is not None
        or (export_html is not None and html_format == "research")
    )
    if needs_matplotlib:
        from buildml.eda.visualize import (
            render_adaptive_plots,
            render_analysis_plots,
            save_figures,
        )

        try:
            figures = render_adaptive_plots(analysis_frame, adaptive_plan, dataset=dataset)
            figures.update(render_analysis_plots(sections))
        except MissingExtraError as exc:
            figures = {"visualization_unavailable": {"error": str(exc)}}
        plot_errors = [
            f"Plot '{key}' skipped: {value.get('error', 'unknown error')}"
            for key, value in figures.items()
            if isinstance(value, dict) and value.get("error")
        ]
        warnings.extend(plot_errors)
        if export_figures is not None:
            root = save_figures(figures, export_figures)
            figure_dir = str(root)
            figure_paths = {
                key: str(root / f"{key}.png")
                for key, fig in figures.items()
                if fig is not None and not isinstance(fig, dict) and (root / f"{key}.png").exists()
            }

    report = EDAReport(
        overview=overview,
        quality=quality,
        univariate=univariate,
        bivariate=bivariate,
        multivariate=multivariate,
        target=target_info,
        outliers=outliers,
        drift=drift,
        findings=findings,
        recommendation_details=recommendation_details,
        narrative=narrative,
        adaptive_plan=adaptive_plan,
        recommendations=recommendations,
        figures=figures,
        warnings=warnings,
        figure_dir=figure_dir,
        figure_paths=figure_paths,
    )

    if export_html is not None:
        report.html_path = str(
            _export_eda_html_path(
                report,
                export_html,
                html_format=html_format,
                max_plots=max_plots,
                warnings=warnings,
            )
        )
        _close_exported_figures(report.figures)

    if show:
        report.show()
    return report


def _export_eda_html_path(
    report: EDAReport,
    path: str | Path,
    *,
    html_format: HtmlFormat,
    max_plots: int,
    warnings: list[str],
) -> Path:
    """Route HTML export to Teaching Studio offline or research shell."""
    if html_format == "studio":
        try:
            from buildml.dashboard.offline import export_studio_html

            return export_studio_html(report.to_dict(), path, title="BuildML EDA Studio")
        except MissingExtraError as exc:
            warnings.append(
                f"Teaching Studio offline export unavailable ({exc}); "
                "falling back to research HTML. Install buildml[dashboard]."
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.append(
                f"Teaching Studio offline export failed ({exc}); falling back to research HTML."
            )
    return export_eda_html(
        report.to_dict(),
        path,
        figures=report.figures,
        max_figures=max_plots,
    )


def summarize_dataset(dataset: Dataset) -> dict[str, Any]:
    """Compact summary subset of :func:`explore_dataset`."""
    report = explore_dataset(dataset, include_plots=False, show=False)
    return {
        "n_rows": report.overview["n_rows"],
        "n_columns": report.overview["n_columns"],
        "columns": report.overview["columns"],
        "dtypes": report.overview["dtypes"],
        # ``missing`` remains a compatibility alias for counts.
        "missing": report.quality["missing_by_column"],
        "missing_counts": report.quality["missing_by_column"],
        "missing_rates": report.quality["missing_rate_by_column"],
        "roles": report.overview["roles"],
        "numeric_describe": report.univariate.get("numeric_describe", {}),
        "categorical_uniques": report.univariate.get("categorical_uniques", {}),
        "recommendations": report.recommendations,
        "narrative": report.narrative,
        "adaptive_plan_count": len(report.adaptive_plan),
        "completeness_score": report.quality.get("completeness_score"),
    }


def _overview(dataset: Dataset, full: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    has_lazy = bool(dataset.has_lazy_native)
    has_native = bool(dataset.has_native)
    disclosures: list[str] = []
    if has_lazy:
        disclosures.append(
            "has_lazy_native=True: Polars LazyFrame is attached. EDA analyzers run on "
            "the promoted/sampled Pandas analysis frame. Collect-on-promote applies for "
            "to_pandas(); this is not out-of-core sklearn training."
        )
    elif has_native:
        disclosures.append(
            "has_native=True: a Polars/DuckDB handle is attached for project/filter/"
            "sample before materialization. Sklearn still needs an in-memory design matrix."
        )
    return {
        "n_rows": int(len(full)),
        "n_columns": int(full.shape[1]),
        "analysis_rows": int(len(sample)),
        "columns": list(full.columns.astype(str)),
        "dtypes": {str(k): str(v) for k, v in full.dtypes.items()},
        "roles": {k: v.value for k, v in dataset.roles.items()},
        "mode": dataset.mode.value,
        "engine": dataset.engine.value,
        "has_native": has_native,
        "has_lazy_native": has_lazy,
        "pandas_stale": bool(dataset.pandas_stale),
        "engine_disclosures": disclosures,
        "memory_bytes_approx": int(full.memory_usage(deep=True).sum()),
        "numeric_columns": list(full.select_dtypes(include="number").columns.astype(str)),
        "categorical_columns": [
            str(c)
            for c in full.columns
            if not pd.api.types.is_numeric_dtype(full[c])
            and not pd.api.types.is_datetime64_any_dtype(full[c])
        ],
        "datetime_columns": list(
            full.select_dtypes(include=["datetime", "datetimetz"]).columns.astype(str)
        ),
    }


def _eligible_feature_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    quality: dict[str, Any],
) -> list[str]:
    """Return columns valid for default feature-only rankings."""
    excluded = set(_feature_exclusion_reasons(dataset, frame, quality))
    return [str(column) for column in frame.columns if str(column) not in excluded]


def _feature_exclusion_reasons(
    dataset: Dataset,
    frame: pd.DataFrame,
    quality: dict[str, Any],
) -> dict[str, list[str]]:
    """Explain explicit-role and heuristic feature exclusions separately."""
    disallowed_roles = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.IGNORE,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    available = set(map(str, frame.columns))
    reasons: dict[str, list[str]] = {}
    for column in map(str, quality.get("constant_columns", [])):
        if column in available:
            reasons.setdefault(column, []).append("constant-column detection")
    for column in map(str, quality.get("id_like_columns", [])):
        if column in available and dataset.roles.get(column) is not ColumnRole.ID:
            reasons.setdefault(column, []).append("heuristic identifier-like detection")
    for column, role in dataset.roles.items():
        if column in available and role in disallowed_roles:
            reasons.setdefault(column, []).append(f"explicit role: {role.value}")
    return reasons


def _bounded_analysis_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    *,
    max_columns: int,
) -> list[str]:
    """Select a stable bounded schema while always retaining assigned targets."""
    columns = list(map(str, frame.columns))
    targets = [column for column in dataset.role_columns(ColumnRole.TARGET) if column in columns]
    selected = columns[:max_columns]
    for target in targets:
        if target in selected:
            continue
        if len(selected) >= max_columns:
            selected[-1] = target
        else:
            selected.append(target)
    return list(dict.fromkeys(selected))


def _close_exported_figures(figures: dict[str, Any]) -> None:
    """Release pyplot managers after assets have been embedded in HTML."""
    if not figures:
        return
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return
    for figure in figures.values():
        if not isinstance(figure, dict) and figure is not None:
            plt.close(figure)
