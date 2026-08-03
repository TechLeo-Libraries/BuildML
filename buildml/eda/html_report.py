# ruff: noqa: E501
"""Lay an EDA report out so a reader can stop at any depth and be satisfied.

The research shell. It orders the report by how much a reader needs rather than
by which analyzer produced what: orientation first, then data quality, features,
relationships, target and validation, figures, next steps, methods, and finally
the raw appendix.

That ordering is the design. Someone who reads only the first section should
come away with an accurate impression; someone who reads everything should find
the evidence for it. The reverse order: analyzer dumps first, conclusions
buried: is what most generated reports do, and it is why most generated reports
go unread.

Each section uses the shared five-part reading frame from
:mod:`buildml.reporting`, so a claim always arrives with what was examined, what
was found, why it matters, what it cannot tell you, and what to do next.

A size budget applies. Every figure is base64-inlined, and a report with three
dozen of them can reach tens of megabytes: past which browsers struggle and
mail gateways refuse. The budget drops figures rather than producing a file
nobody can open, and says in the document that it did.

See Also
--------
buildml.reporting.html : The components and the document shell.
buildml.dashboard.offline : The studio alternative.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from buildml.reporting.html import (
    ReportSection,
    element_id,
    encode_asset,
    escape,
    render_badge,
    render_reading_frame,
    render_report,
    render_table,
    severity_tone,
)

DEFAULT_MAX_HTML_BYTES = 12 * 1024 * 1024


def export_eda_html(
    report_dict: dict[str, Any],
    path: str | Path,
    *,
    title: str = "BuildML EDA Report",
    figures: Mapping[str, Any] | None = None,
    include_raw_appendix: bool = True,
    max_figures: int = 36,
    max_html_bytes: int = DEFAULT_MAX_HTML_BYTES,
) -> Path:
    """Write the whole report as one HTML file with nothing left outside it.

    Assembles the layered sections, inlines every figure, and writes a single
    self-contained document. No CDN, no sidecar images, no server: the file
    opens from a USB stick on a machine that has never heard of Python.

    Sections run from orientation to appendix, each in the five-part reading
    frame, so the document can be read to any depth and still make sense.

    Parameters
    ----------
    report_dict:
        The report as plain data, from
        :meth:`~buildml.eda.report.EDAReport.to_dict`. Missing sections are
        skipped rather than erroring, so a partial pass still exports.
    path:
        Where to write. Parent directories are created.
    title:
        The document title. Worth setting to something identifying: the dataset
        and the date: since these files accumulate.
    figures:
        Rendered figures to embed. Error entries are handled and reported in the
        document rather than skipped silently.
    include_raw_appendix:
        Append the full analyzer output as formatted JSON. Makes the file
        larger and makes every number checkable.
    max_figures:
        Ceiling on embedded figures.
    max_html_bytes:
        Size budget, 12 MiB by default. Figures are dropped to stay under it,
        and the document states what was dropped.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If ``max_figures`` is negative or ``max_html_bytes`` is not positive.
    OSError
        If the file cannot be written.

    Notes
    -----
    **Check the degraded section in the output.** It lists figures that failed
    to render and figures dropped for the size budget, so a report that is
    missing charts says why in the document rather than only in a log.

    **Figures are the size driver.** Base64 costs a third over the raw bytes,
    and it is all in one file. Fewer, smaller figures export better than many
    large ones.

    Examples
    --------
    ::

        report = explore_dataset(dataset, include_plots=True)
        export_eda_html(
            report.to_dict(),
            "artifacts/eda.html",
            title="Churn dataset · 2026-08",
            figures=report.figures,
        )

    See Also
    --------
    buildml.eda.report.EDAReport.save_html : The usual way to call this.
    """
    if max_figures < 0:
        raise ValueError("max_figures must be non-negative")
    if max_html_bytes < 1:
        raise ValueError("max_html_bytes must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overview = report_dict.get("overview") or {}
    quality = report_dict.get("quality") or {}
    target = report_dict.get("target") or {}
    drift = report_dict.get("drift") or {}
    sections = [
        _orientation_section(overview, quality, report_dict),
        _quality_section(quality),
        _feature_section(report_dict),
        _relationship_section(report_dict),
        _target_validation_section(target, drift),
        _figure_section(report_dict, figures or {}, max_figures=max_figures),
        _next_steps_section(report_dict),
        _methods_section(report_dict),
        _degraded_section(report_dict, figures or {}),
    ]
    if include_raw_appendix:
        sections.append(_appendix_section(report_dict))
    document = render_report(
        title,
        sections,
        subtitle=(
            "Descriptive evidence, limitations, and next steps. Associations in this "
            "report do not establish causality."
        ),
        metadata={
            "Rows": overview.get("n_rows", "not available"),
            "Columns": overview.get("n_columns", "not available"),
            "Analysis rows": overview.get("analysis_rows", "not available"),
            "Engine": overview.get("engine", "not available"),
        },
    )
    encoded_size = len(document.encode("utf-8"))
    if encoded_size > max_html_bytes and include_raw_appendix:
        sections[-1] = _appendix_omitted_section(encoded_size, max_html_bytes)
        document = render_report(
            title,
            sections,
            subtitle=(
                "Descriptive evidence, limitations, and next steps. Associations in this "
                "report do not establish causality."
            ),
            metadata={
                "Rows": overview.get("n_rows", "not available"),
                "Columns": overview.get("n_columns", "not available"),
                "Analysis rows": overview.get("analysis_rows", "not available"),
                "Engine": overview.get("engine", "not available"),
            },
        )
    final_size = len(document.encode("utf-8"))
    if final_size > max_html_bytes:
        raise ValueError(
            f"Report size {final_size:,} bytes exceeds max_html_bytes={max_html_bytes:,}; "
            "reduce sample_rows, max_columns, or max_figures."
        )
    destination.write_text(document, encoding="utf-8")
    return destination


def _orientation_section(
    overview: dict[str, Any],
    quality: dict[str, Any],
    report: dict[str, Any],
) -> ReportSection:
    total = overview.get("n_rows")
    analyzed = overview.get("analysis_rows")
    sampled = isinstance(total, int) and isinstance(analyzed, int) and analyzed < total
    roles = overview.get("roles") or {}
    body = _frame(
        examined="Dataset shape, analysis scope, sampling, assigned roles, and basic readiness indicators.",
        observed=(
            f"{_fmt(total)} rows × {_fmt(overview.get('n_columns'))} columns; "
            f"{_fmt(analyzed)} rows used by sampled analyzers. "
            f"Observed completeness: {_pct(quality.get('completeness_score'))}."
        ),
        why="Scope and role assignments determine which statistics are valid inputs to later feature analyses.",
        limits=(
            "Heavy analyses used a sample, so rare patterns may differ in the full data."
            if sampled
            else "The analyzed row count matches the dataset row count; analyzer-specific complete-case filters may still reduce samples."
        ),
        next_step="Confirm roles and validation design before preprocessing or modeling.",
    )
    body += "<h3>Scope and readiness</h3>" + render_table(
        [
            {"field": "Source mode", "value": overview.get("mode")},
            {"field": "Approximate memory bytes", "value": overview.get("memory_bytes_approx")},
            {"field": "Assigned roles", "value": roles},
            {
                "field": "Detailed-analysis columns",
                "value": (
                    f"{overview.get('analysis_column_count', overview.get('n_columns'))} "
                    f"of {overview.get('n_columns')} "
                    f"(budget={overview.get('analysis_column_budget', 'not set')})"
                ),
            },
            {"field": "Eligible feature columns", "value": overview.get("eligible_feature_columns", [])},
            {"field": "Excluded from feature analysis", "value": overview.get("excluded_from_feature_analysis", [])},
            {
                "field": "Heuristic identifier exclusions",
                "value": overview.get("heuristic_id_exclusions", []),
            },
            {
                "field": "Explicit role exclusions",
                "value": overview.get("explicit_role_exclusions", []),
            },
            {
                "field": "Feature exclusion reasons",
                "value": overview.get("feature_exclusion_reasons", {}),
            },
            {"field": "Warnings", "value": report.get("warnings", [])},
        ],
        caption="Orientation details",
    )
    return ReportSection("orientation", "Orientation, scope, sampling, and readiness", body)


def _quality_section(quality: dict[str, Any]) -> ReportSection:
    missing_counts = quality.get("missing_by_column") or {}
    missing_rates = quality.get("missing_rate_by_column") or {}
    rows = [
        {
            "column": column,
            "missing_count": count,
            "missing_rate": missing_rates.get(column),
        }
        for column, count in missing_counts.items()
    ]
    rows.sort(key=lambda row: float(row["missing_rate"] or 0), reverse=True)
    flags = [
        {"check": label, "columns": quality.get(key, [])}
        for key, label in (
            ("constant_columns", "Constant"),
            ("quasi_constant_columns", "Quasi-constant"),
            ("id_like_columns", "Identifier-like (heuristic)"),
            ("high_cardinality_columns", "High cardinality"),
            ("mixed_type_suspect_columns", "Mixed-type suspect"),
        )
    ]
    body = _frame(
        examined="Cell and row missingness, duplicate rows, cardinality, constancy, and string-pattern hints.",
        observed=(
            f"{int(quality.get('missing_cell_count', 0)):,} missing cells and "
            f"{int(quality.get('duplicate_row_count', 0)):,} duplicate rows were observed."
        ),
        why="These conditions can affect usable sample size, leakage, encoding, and estimator behavior.",
        limits="Heuristic identifier, cardinality, and pattern flags require domain confirmation.",
        next_step="Review high-rate columns and define train-fitted cleaning decisions.",
    )
    body += "<h3>Missingness by column</h3>" + render_table(
        rows, caption="Missing counts and rates"
    )
    body += "<h3>Quality flags</h3>" + render_table(flags, caption="Heuristic quality flags")
    body += "<details><summary>Row missingness and pattern details</summary>"
    body += render_table(
        [
            {"metric": "Rows with any missing", "value": quality.get("rows_with_any_missing")},
            {"metric": "Rows with any missing rate", "value": quality.get("rows_with_any_missing_rate")},
            {"metric": "Missing fields per row quantiles", "value": quality.get("missingness_by_row_quantiles")},
            {"metric": "String pattern hints", "value": quality.get("string_pattern_hints")},
        ],
        caption="Additional quality evidence",
    ) + "</details>"
    return ReportSection("quality", "Data quality", body)


def _feature_section(report: dict[str, Any]) -> ReportSection:
    profiles = (report.get("univariate") or {}).get("per_column") or {}
    roles = (report.get("overview") or {}).get("roles") or {}
    missing_counts = (report.get("quality") or {}).get("missing_by_column") or {}
    missing_rates = (report.get("quality") or {}).get("missing_rate_by_column") or {}
    exclusion_reasons = (report.get("overview") or {}).get("feature_exclusion_reasons") or {}
    rows = []
    for column, stats in profiles.items():
        rows.append(
            {
                "column": column,
                "role": roles.get(column, "unassigned"),
                "kind": stats.get("kind"),
                "missing_count": missing_counts.get(column, 0),
                "missing_rate": missing_rates.get(column, 0),
                "count": stats.get("count"),
                "unique": stats.get("nunique"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "median": stats.get("median"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "skew": stats.get("skew"),
                "mode": stats.get("mode"),
                "feature_eligibility": (
                    "excluded" if column in exclusion_reasons else "eligible"
                ),
                "exclusion_reason": exclusion_reasons.get(column),
                "normality_method": stats.get("normality_method"),
                "normality_sample_size": stats.get("normality_sample_size"),
                "normality_stat": stats.get("normality_stat"),
                "normality_pvalue": stats.get("normality_pvalue"),
                "appears_non_normal": stats.get("appears_non_normal"),
                "normality_caveats": stats.get("normality_assumptions"),
            }
        )
    body = _frame(
        examined="Per-column distributions, central tendency, spread, shape, cardinality, and missingness.",
        observed=f"Profiles are available for {len(rows):,} columns.",
        why="Feature-level distributions guide validation, transformations, encoding, and error checks.",
        limits=(
            "Normality tests are unadjusted screens with independence and continuity assumptions; "
            "non-significance does not prove normality."
        ),
        next_step="Use the table search and column filters to inspect features relevant to the planned model.",
    )
    body += render_table(rows, caption="Searchable feature profiles")
    body += "<details><summary>Full per-column technical profiles</summary>"
    body += _json_block(profiles) + "</details>"
    return ReportSection("features", "Feature profiles", body)


def _relationship_section(report: dict[str, Any]) -> ReportSection:
    bivariate = report.get("bivariate") or {}
    multivariate = report.get("multivariate") or {}
    mi = bivariate.get("mutual_information_vs_target") or {}
    mi_rows = [{"feature": key, "mutual_information": value} for key, value in mi.items()]
    cat_rows = bivariate.get("categorical_pairs") or []
    clusters = [
        {"cluster": index + 1, "columns": columns}
        for index, columns in enumerate(multivariate.get("correlation_clusters") or [])
    ]
    pca = multivariate.get("pca") or {}
    body = _frame(
        examined="Pearson, Spearman, selected Kendall correlations, Cramér's V, target mutual information, VIF, correlation clusters, and PCA.",
        observed=(
            f"{len(bivariate.get('top_abs_pearson_pairs') or []):,} Pearson pairs, "
            f"{len(cat_rows):,} categorical pairs, and {len(mi_rows):,} MI estimates are shown."
        ),
        why="Different measures capture different forms of association and redundancy.",
        limits="Association is not causation. Pairwise deletion, complete-case PCA/VIF, encoding, and multiple testing can affect results.",
        next_step="Compare conclusions across measures and validate any feature decision out of sample.",
    )
    body += "<h3>Ranked associations</h3>"
    body += render_table(bivariate.get("top_abs_pearson_pairs") or [], caption="Pearson pairs by absolute coefficient")
    body += render_table(bivariate.get("kendall_top_pairs") or [], caption="Selected Kendall pairs")
    body += render_table(cat_rows, caption="Categorical pairs: Cramér's V")
    body += render_table(mi_rows, caption="Mutual information versus target")
    body += "<h3>Multivariate structure</h3>"
    body += render_table(multivariate.get("vif") or [], caption="Variance inflation factors")
    body += render_table(clusters, caption="Correlation clusters")
    body += render_table(
        [
            {
                "component": index + 1,
                "explained_variance_ratio": ratio,
                "cumulative_explained_variance": (pca.get("cumulative_explained_variance") or [None] * len(pca.get("explained_variance_ratio") or []))[index],
                "top_loadings": (pca.get("components_top_loadings") or {}).get(f"pc{index + 1}"),
            }
            for index, ratio in enumerate(pca.get("explained_variance_ratio") or [])
        ],
        caption="Principal component summary",
    )
    body += "<details><summary>Pearson and Spearman matrices</summary><h4>Pearson</h4>"
    body += _matrix_table(bivariate.get("pearson") or {})
    body += "<h4>Spearman</h4>" + _matrix_table(bivariate.get("spearman") or {}) + "</details>"
    return ReportSection("relationships", "Relationships and multivariate structure", body)


def _target_validation_section(target: dict[str, Any], drift: dict[str, Any]) -> ReportSection:
    target_rows = [{"metric": key, "value": value} for key, value in (target.get("summary") or {}).items()]
    drift_rows = [
        *list(drift.get("numeric_drift") or []),
        *list(drift.get("categorical_drift") or []),
    ]
    body = _frame(
        examined="Target distribution and feature associations, plus train/test drift when a split is available.",
        observed=(
            f"Target: {target.get('column', 'not assigned')}; "
            f"drift status: {drift.get('summary', drift.get('reason', 'not available'))}."
        ),
        why="Target balance and validation-set shift affect metric choice and generalization estimates.",
        limits="Univariate target tests and drift flags are descriptive screens; p-values are not causal evidence and were not multiplicity-adjusted.",
        next_step="Confirm split strategy, investigate flagged columns, and choose metrics appropriate to the target.",
    )
    body += render_table(target_rows, caption="Target summary")
    body += render_table(target.get("top_numeric_associations") or [], caption="Regression-target numeric associations")
    body += render_table(target.get("categorical_effect_tests") or [], caption="Categorical effect screens")
    body += render_table(target.get("numeric_separation_tests") or [], caption="Classification separation screens")
    body += render_table(drift_rows, caption="Train/test drift statistics and flags")
    if drift.get("settings"):
        body += "<details><summary>Drift settings</summary>" + _json_block(drift["settings"]) + "</details>"
    return ReportSection("target-validation", "Target, validation, and drift", body)


def _figure_section(
    report: dict[str, Any],
    figures: Mapping[str, Any],
    *,
    max_figures: int,
) -> ReportSection:
    assets: list[tuple[str, str]] = []
    skipped: list[str] = []
    for name, figure in figures.items():
        if len(assets) >= max_figures:
            skipped.append(f"{name}: figure budget of {max_figures} reached")
            continue
        if isinstance(figure, Mapping):
            skipped.append(f"{name}: {figure.get('error', 'not rendered')}")
            continue
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            assets.append((str(name), encode_asset(buffer.getvalue(), media_type="image/png")))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{name}: {exc}")
    for name, raw_path in (report.get("figure_paths") or {}).items():
        if any(existing == str(name) for existing, _ in assets):
            continue
        if len(assets) >= max_figures:
            skipped.append(f"{name}: figure budget of {max_figures} reached")
            continue
        try:
            assets.append((str(name), encode_asset(Path(raw_path))))
        except OSError as exc:
            skipped.append(f"{name}: {exc}")
    gallery = "".join(
        (
            '<figure class="bml-figure">'
            f'<button type="button" class="bml-figure__expand" aria-label="Expand {escape(name)}">'
            f'<img src="{uri}" alt="{escape(name)}"></button>'
            f"<figcaption>{escape(name)}</figcaption></figure>"
        )
        for name, uri in assets
    )
    body = _frame(
        examined="Automatically selected missingness, schema, correlation, distribution, target, temporal, relationship, and outlier views.",
        observed=f"{len(assets):,} plots were embedded; {len(skipped):,} plot renders were skipped.",
        why="Plots expose shape and local patterns that compact statistics can hide.",
        limits="Plot selection is capped and sampled plots may omit rare observations.",
        next_step="Expand plots for inspection and use the statistics tables for exact values.",
    )
    body += f'<div class="bml-gallery">{gallery}</div>'
    if skipped:
        body += "<details><summary>Skipped plots</summary><ul>" + "".join(
            f"<li>{escape(item)}</li>" for item in skipped
        ) + "</ul></details>"
    return ReportSection("figures", "Visual evidence", body)


def _next_steps_section(report: dict[str, Any]) -> ReportSection:
    findings = report.get("findings") or []
    recommendations = report.get("recommendation_details") or []
    finding_html = []
    for finding in findings:
        key = str(finding.get("key", "finding"))
        evidence = finding.get("evidence") or []
        evidence_html = "".join(
            (
                f'<li id="{escape(element_id(item.get("key", "evidence"), prefix="evidence"))}">'
                f"<strong>{escape(item.get('summary'))}</strong>: {escape(_fmt(item.get('value')))}"
                + (
                    "<br><small>Limits: "
                    + escape("; ".join(item.get("limitations") or []))
                    + "</small>"
                    if item.get("limitations")
                    else ""
                )
                + "</li>"
            )
            for item in evidence
        )
        finding_html.append(
            f'<article class="bml-finding" id="{escape(element_id(key, prefix="finding"))}">'
            f"<h3>{escape(finding.get('title'))} {render_badge(finding.get('severity', 'info'), tone=severity_tone(finding.get('severity')))}</h3>"
            f"<p>{escape(finding.get('detail'))}</p>"
            f"<details><summary>Supporting evidence</summary><ul>{evidence_html}</ul></details></article>"
        )
    rec_html = []
    for recommendation in recommendations:
        based = recommendation.get("based_on") or []
        links = ", ".join(
            f'<a href="#{escape(element_id(item, prefix="finding"))}">{escape(item)}</a>'
            for item in based
        )
        caveats = "; ".join(recommendation.get("caveats") or [])
        rec_html.append(
            "<li>"
            f"<strong>{escape(recommendation.get('title'))}</strong> "
            f"{render_badge(recommendation.get('priority'), tone='info')}"
            f"<p>{escape(recommendation.get('rationale'))}</p>"
            f"<p>Based on: {links or 'no linked finding'}</p>"
            + (f"<p><small>Limits: {escape(caveats)}</small></p>" if caveats else "")
            + "</li>"
        )
    body = _frame(
        examined="Structured findings and recommendations linked to explicit analyzer evidence.",
        observed=f"{len(findings):,} findings support {len(recommendations):,} suggested next steps.",
        why="Evidence links separate observations from interpretations and make advice auditable.",
        limits="Recommendations are defaults and do not replace domain, cost, fairness, or deployment review.",
        next_step="Open each finding's evidence before accepting or rejecting its linked recommendation.",
    )
    body += "<h3>Top findings</h3>" + "".join(finding_html)
    body += (
        "<h3>Evidence-linked next steps</h3>"
        '<ol class="bml-recommendations">'
        + "".join(rec_html)
        + "</ol>"
    )
    return ReportSection("next-steps", "Findings and next steps", body)


def _methods_section(report: dict[str, Any]) -> ReportSection:
    multivariate = report.get("multivariate") or {}
    drift = report.get("drift") or {}
    profiles = (report.get("univariate") or {}).get("per_column") or {}
    normality_methods = sorted(
        {
            str(profile["normality_method"])
            for profile in profiles.values()
            if profile.get("normality_method")
        }
    )
    normality_sample_sizes = [
        int(profile["normality_sample_size"])
        for profile in profiles.values()
        if profile.get("normality_method") and profile.get("normality_sample_size") is not None
    ]
    body = _frame(
        examined="Analyzer methods, sample sizes, feature eligibility, and known interpretation boundaries.",
        observed="Methods and settings below are those retained in the report payload.",
        why="Reproducible interpretation requires the statistic, sample, eligibility rules, and thresholds.",
        limits="Library versions and all low-level estimator defaults are not currently recorded in this report.",
        next_step="Retain the report with the data/split checkpoint when reproducibility is required.",
    )
    body += render_table(
        [
            {"method": "Correlation", "setting": "Pearson and Spearman matrices; Kendall on selected leading pairs"},
            {"method": "Categorical association", "setting": "Cramér's V for low-cardinality pairs"},
            {"method": "Mutual information", "setting": "scikit-learn estimators, random_state=0"},
            {
                "method": "Normality screening",
                "setting": (
                    f"methods={normality_methods or ['none eligible']}; "
                    f"sample sizes={normality_sample_sizes or ['none']}; alpha=0.05; "
                    "sample cap=5000; assumes independent continuous observations; "
                    "unadjusted p-values are sample-size sensitive and non-significance "
                    "does not prove normality"
                ),
            },
            {"method": "VIF/PCA", "setting": f"complete cases={multivariate.get('complete_case_rows', 0)}; standardized PCA"},
            {"method": "Outliers", "setting": "IQR, absolute z-score > 3, Isolation Forest random_state=0"},
            {"method": "Drift", "setting": drift.get("settings", drift.get("reason"))},
        ],
        caption="Methods and retained settings",
    )
    return ReportSection("methods", "Methods and limitations", body)


def _degraded_section(report: dict[str, Any], figures: Mapping[str, Any]) -> ReportSection:
    rows = [{"analysis": "report warning", "reason": warning} for warning in report.get("warnings") or []]
    for name, value in figures.items():
        if isinstance(value, Mapping):
            rows.append({"analysis": f"plot: {name}", "reason": value.get("error", "not rendered")})
    drift = report.get("drift") or {}
    if not drift.get("available"):
        rows.append({"analysis": "train/test drift", "reason": drift.get("reason", "No split available")})
    for key, label in (
        ("pearson", "Pearson matrix"),
        ("spearman", "Spearman matrix"),
        ("categorical_pairs", "Cramér's V"),
        ("mutual_information_vs_target", "Mutual information"),
    ):
        if not (report.get("bivariate") or {}).get(key):
            rows.append({"analysis": label, "reason": "No eligible result was produced"})
    if not (report.get("multivariate") or {}).get("pca"):
        rows.append({"analysis": "PCA", "reason": "Insufficient eligible complete numeric data"})
    body = _frame(
        examined="Warnings and analyses that were unavailable, inapplicable, or failed gracefully.",
        observed=f"{len(rows):,} degraded or skipped items are listed.",
        why="Absence of a result should not be mistaken for evidence that no issue exists.",
        limits="Some analyzers intentionally return an empty result after internal numerical errors.",
        next_step="Resolve prerequisites or run focused diagnostics when a skipped item matters.",
    )
    body += render_table(rows, caption="Skipped and degraded analyses")
    return ReportSection("degraded", "Skipped and degraded analyses", body)


def _appendix_section(report: dict[str, Any]) -> ReportSection:
    body = _frame(
        examined="Serializable technical analyzer payload.",
        observed="The payload is preserved without the in-memory matplotlib objects.",
        why="Raw values support audit, troubleshooting, and downstream extraction.",
        limits="JSON formatting is for inspection and may be large.",
        next_step="Use the structured EDAReport object for programmatic access.",
    )
    body += "<details><summary>Open raw technical payload</summary>" + _json_block(report) + "</details>"
    return ReportSection("appendix", "Raw technical appendix", body)


def _appendix_omitted_section(original_bytes: int, budget_bytes: int) -> ReportSection:
    body = _frame(
        examined="Serializable technical analyzer payload.",
        observed=(
            f"The raw appendix was omitted because the first render was "
            f"{original_bytes:,} bytes against a {budget_bytes:,}-byte output budget."
        ),
        why="Output budgets keep wide reports practical to store, open, and review.",
        limits="The structured EDAReport object still contains the complete analyzer payload.",
        next_step="Use EDAReport.to_dict() or raise max_html_bytes for an intentional larger export.",
    )
    return ReportSection("appendix", "Raw appendix omitted by output budget", body)


def _frame(*, examined: str, observed: str, why: str, limits: str, next_step: str) -> str:
    return render_reading_frame(
        examined=examined,
        observed=observed,
        why=why,
        limits=limits,
        next_step=next_step,
    )


def _matrix_table(matrix: dict[str, Any]) -> str:
    columns = list(matrix)
    rows = []
    for row_name in columns:
        row = {"feature": row_name}
        for column in columns:
            column_values = matrix.get(column) or {}
            row[column] = column_values.get(row_name)
        rows.append(row)
    return render_table(rows, caption="Association matrix")


def _json_block(value: Any) -> str:
    return f'<pre class="bml-json">{escape(json.dumps(value, indent=2, default=str))}</pre>'


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return str(value)


def _pct(value: Any) -> str:
    return f"{float(value):.3%}" if isinstance(value, (int, float)) else "not available"
