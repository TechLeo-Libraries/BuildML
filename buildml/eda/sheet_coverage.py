"""Shared readiness-sheet analytic coverage for App Cockpit and Static EDA.

Both surfaces need the same measured depth: a full ledger of computed numbers,
methods/limitations cards, degraded/skipped rows, and domain-board briefs.
Keeping the builders here means an analyzer that lands in the report dict is
visible on every Industry spine without duplicating truncation logic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

def _flagged_names(flagged: Any) -> list[str]:
    """Accept plain name lists or ``{column: ...}`` rows from analyzers."""
    if not isinstance(flagged, list):
        return []
    names: list[str] = []
    for item in flagged:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Mapping) and item.get("column") is not None:
            names.append(str(item["column"]))
    return names


def fmt_metric(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return str(value)


def fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.{digits}f}%"
    return "not available"


def build_ledger_groups(
    report: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Every meaningful computed number, grouped for the Industry ledger.

    Empty theater is omitted: a group appears only when its source section
    produced values. Items are ``(label, display)`` tuples.
    """
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    bivariate = report.get("bivariate") or {}
    multivariate = report.get("multivariate") or {}
    target = report.get("target") or {}
    outliers = report.get("outliers") or {}
    drift = report.get("drift") or {}
    findings_list = list(findings if findings is not None else (report.get("findings") or []))

    roles = overview.get("roles") or {}
    role_counts: dict[str, int] = {}
    for role in roles.values():
        role_counts[str(role)] = role_counts.get(str(role), 0) + 1

    sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for finding in findings_list:
        sev = str(finding.get("severity", "info")).lower()
        if sev in {"crit"}:
            sev = "critical"
        if sev in {"med"}:
            sev = "medium"
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    groups: list[dict[str, Any]] = [
        {
            "key": "frame",
            "title": "Frame",
            "items": [
                ("rows analysed", fmt_int(overview.get("analysis_rows"))),
                ("rows in frame", fmt_int(overview.get("n_rows"))),
                ("columns", fmt_int(overview.get("n_columns"))),
                (
                    "eligible features",
                    fmt_int(len(overview.get("eligible_feature_columns") or [])),
                ),
                ("missing cells", fmt_int(quality.get("missing_cell_count"))),
                ("cell completeness", fmt_pct(quality.get("completeness_score"))),
                ("duplicate rows", fmt_int(quality.get("duplicate_row_count"))),
                (
                    "sampling",
                    (
                        f"{fmt_int(overview.get('analysis_rows'))} of "
                        f"{fmt_int(overview.get('n_rows'))}"
                        if overview.get("analysis_rows") != overview.get("n_rows")
                        else "none disclosed"
                    ),
                ),
                ("engine", str(overview.get("engine") or "pandas")),
                ("mode", str(overview.get("mode") or "eager")),
            ],
        },
        {
            "key": "roles",
            "title": "Roles & severity",
            "items": [
                *[(f"role · {name}", str(count)) for name, count in sorted(role_counts.items())],
                (
                    "findings · crit / high",
                    f"{sev_counts.get('critical', 0)} / {sev_counts.get('high', 0)}",
                ),
                (
                    "findings · med / low",
                    f"{sev_counts.get('medium', 0)} / {sev_counts.get('low', 0)}",
                ),
                ("findings · info", str(sev_counts.get("info", 0))),
                ("findings · total", fmt_int(len(findings_list))),
            ],
        },
    ]

    missing_rates = quality.get("missing_rate_by_column") or {}
    if isinstance(missing_rates, dict) and missing_rates:
        miss_items = sorted(
            (
                (str(col), float(rate))
                for col, rate in missing_rates.items()
                if float(rate or 0) > 0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        n_cols = int(overview.get("n_columns") or 0)
        ledger_miss = [(name, fmt_pct(rate)) for name, rate in miss_items[:40]]
        complete_n = max(0, n_cols - len(miss_items))
        if complete_n:
            ledger_miss.append(
                (
                    f"{complete_n} other column{'s' if complete_n != 1 else ''}",
                    "0.000%",
                )
            )
        if ledger_miss:
            groups.append(
                {
                    "key": "missing",
                    "title": "Missing rate by column",
                    "items": ledger_miss,
                }
            )

    quality_flags = [
        ("duplicate rows", fmt_int(quality.get("duplicate_row_count"))),
        ("constant columns", fmt_int(len(quality.get("constant_columns") or []))),
        (
            "near-constant columns",
            fmt_int(len(quality.get("quasi_constant_columns") or [])),
        ),
        (
            "high-cardinality columns",
            fmt_int(len(quality.get("high_cardinality_columns") or [])),
        ),
        ("identifier-like columns", fmt_int(len(_flagged_names(quality.get("id_like_columns"))))),
        (
            "mixed-type suspects",
            fmt_int(len(quality.get("mixed_type_suspect_columns") or [])),
        ),
        ("rows with any missing", fmt_int(quality.get("rows_with_any_missing"))),
        (
            "rows with any missing rate",
            fmt_pct(quality.get("rows_with_any_missing_rate")),
        ),
    ]
    if quality:
        groups.append({"key": "quality-flags", "title": "Quality flags", "items": quality_flags})

    mi = bivariate.get("mutual_information_vs_target") or bivariate.get("mutual_information")
    if isinstance(mi, dict) and mi:
        mi_sorted = sorted(mi.items(), key=lambda pair: float(pair[1] or 0), reverse=True)
        groups.append(
            {
                "key": "mi",
                "title": "Mutual information vs target",
                "items": [(str(name), fmt_metric(value)) for name, value in mi_sorted[:48]],
            }
        )
    elif isinstance(mi, list) and mi:
        mi_items: list[tuple[str, str]] = []
        for row in mi[:48]:
            if not isinstance(row, Mapping):
                continue
            name = row.get("feature") or row.get("column") or row.get("key")
            score = row.get("mi") or row.get("score") or row.get("value")
            if name is None:
                continue
            mi_items.append((str(name), fmt_metric(score)))
        if mi_items:
            groups.append(
                {
                    "key": "mi",
                    "title": "Mutual information vs target",
                    "items": mi_items,
                }
            )

    pairs = bivariate.get("top_abs_pearson_pairs") or []
    if pairs:
        pair_items = []
        for pair in pairs[:24]:
            if not isinstance(pair, Mapping):
                continue
            label = f"{pair.get('a')} ↔ {pair.get('b')}"
            pair_items.append((label, fmt_metric(pair.get("corr"))))
        if len(pairs) > 24:
            pair_items.append((f"{len(pairs) - 24} additional pairs", "truncated in ledger"))
        groups.append({"key": "pearson", "title": "Top |Pearson| pairs", "items": pair_items})

    spearman = bivariate.get("top_abs_spearman_pairs") or []
    if spearman:
        groups.append(
            {
                "key": "spearman",
                "title": "Top |Spearman| pairs",
                "items": [
                    (
                        f"{row.get('a')} ↔ {row.get('b')}",
                        fmt_metric(row.get("corr")),
                    )
                    for row in spearman[:16]
                    if isinstance(row, Mapping)
                ],
            }
        )

    cat_pairs = bivariate.get("categorical_pairs") or []
    if cat_pairs:
        groups.append(
            {
                "key": "cramers",
                "title": "Categorical association (Cramér's V)",
                "items": [
                    (
                        f"{row.get('a')} ↔ {row.get('b')}",
                        fmt_metric(row.get("cramers_v", row.get("v"))),
                    )
                    for row in cat_pairs[:24]
                    if isinstance(row, Mapping)
                ],
            }
        )

    kendall = bivariate.get("kendall_top_pairs") or []
    if kendall:
        groups.append(
            {
                "key": "kendall",
                "title": "Leading Kendall pairs",
                "items": [
                    (
                        f"{row.get('a')} ↔ {row.get('b')}",
                        fmt_metric(row.get("corr", row.get("tau"))),
                    )
                    for row in kendall[:12]
                    if isinstance(row, Mapping)
                ],
            }
        )

    vif_rows = multivariate.get("vif") or []
    if isinstance(vif_rows, dict) and vif_rows:
        vif_items = [
            (str(name), fmt_metric(score))
            for name, score in sorted(
                vif_rows.items(),
                key=lambda kv: float(kv[1] or 0),
                reverse=True,
            )[:48]
        ]
        groups.append(
            {
                "key": "vif",
                "title": "Variance inflation (complete case)",
                "items": vif_items,
            }
        )
    elif isinstance(vif_rows, list) and vif_rows:
        groups.append(
            {
                "key": "vif",
                "title": "Variance inflation (complete case)",
                "items": [
                    (
                        str(row.get("column") or row.get("feature") or row.get("key")),
                        fmt_metric(row.get("vif") or row.get("value")),
                    )
                    for row in vif_rows[:48]
                    if isinstance(row, Mapping)
                ],
            }
        )

    clusters = multivariate.get("correlation_clusters") or []
    if clusters:
        groups.append(
            {
                "key": "clusters",
                "title": "Correlation clusters",
                "items": [
                    (f"cluster {index}", ", ".join(str(col) for col in group[:10]))
                    for index, group in enumerate(clusters[:16], start=1)
                ],
            }
        )

    pca = multivariate.get("pca") or {}
    variance = pca.get("explained_variance_ratio") or []
    if variance:
        cum = pca.get("cumulative_explained_variance") or []
        groups.append(
            {
                "key": "pca",
                "title": "PCA explained variance",
                "items": [
                    (f"PC{index}", fmt_pct(value))
                    for index, value in enumerate(variance, start=1)
                ]
                + [
                    ("components", fmt_int(pca.get("n_components"))),
                    (
                        "cumulative last",
                        fmt_pct(cum[-1] if cum else None),
                    ),
                    (
                        "complete-case rows",
                        fmt_int(multivariate.get("complete_case_rows")),
                    ),
                ],
            }
        )

    summary = target.get("summary") or {}
    screens: list[tuple[str, str]] = [
        ("target column", str(target.get("column") or "not declared")),
        ("task", str(summary.get("type") or summary.get("task") or "—")),
    ]
    class_counts = summary.get("class_counts") or {}
    if isinstance(class_counts, dict) and class_counts:
        total = sum(int(value) for value in class_counts.values()) or 1
        for label, count in list(class_counts.items())[:16]:
            screens.append(
                (f"class · {label}", f"{fmt_int(count)} · {fmt_pct(int(count) / total)}")
            )
        if summary.get("imbalance_ratio") is not None:
            screens.append(("imbalance ratio", fmt_metric(summary.get("imbalance_ratio"))))
    elif str(summary.get("type") or "").startswith("regression"):
        for key in ("mean", "std", "skew", "min", "max", "median"):
            if summary.get(key) is not None:
                screens.append((key, fmt_metric(summary.get(key))))

    for assoc_key, title in (
        ("top_numeric_associations", "target numeric associations"),
        ("numeric_separation_tests", "target numeric separation tests"),
        ("categorical_effect_tests", "target categorical effect tests"),
    ):
        rows = target.get(assoc_key) or []
        if rows:
            screens.append((title, fmt_int(len(rows))))

    if outliers.get("multivariate"):
        multi = outliers["multivariate"]
        flagged = multi.get(
            "anomaly_count",
            multi.get("flagged_row_count", multi.get("n_flagged", multi.get("flagged"))),
        )
        scored = multi.get(
            "n_rows_scored",
            multi.get("scored_row_count", multi.get("n_scored", multi.get("scored"))),
        )
        if flagged is not None:
            screens.append(
                (
                    "anomalies / scored",
                    f"{fmt_int(flagged)} / {fmt_int(scored)}" if scored is not None else fmt_int(flagged),
                )
            )
            if multi.get("anomaly_rate") is not None:
                screens.append(("anomaly rate", fmt_pct(multi.get("anomaly_rate"))))

    outlier_cols = outliers.get("per_column") or {}
    if isinstance(outlier_cols, dict) and outlier_cols:
        hot = sum(
            1
            for stats in outlier_cols.values()
            if isinstance(stats, Mapping)
            and float(stats.get("iqr_outlier_rate") or 0) >= 0.05
        )
        screens.append(("columns with IQR rate ≥5%", fmt_int(hot)))
        screens.append(("outlier columns scored", fmt_int(len(outlier_cols))))

    flagged_drift = _flagged_names(drift.get("flagged_columns"))
    screens.append(("drift flags", fmt_int(len(flagged_drift))))
    if drift.get("available"):
        screens.append(
            (
                "drift train / test rows",
                f"{fmt_int(drift.get('train_rows'))} / {fmt_int(drift.get('test_rows'))}",
            )
        )
    screens.append(
        ("complete-case rows (VIF/PCA)", fmt_int(multivariate.get("complete_case_rows")))
    )
    groups.append({"key": "screens", "title": "Target & screens", "items": screens})

    if isinstance(outlier_cols, dict) and outlier_cols:
        rate_raw = sorted(
            (
                (str(name), float(stats.get("iqr_outlier_rate") or 0))
                for name, stats in outlier_cols.items()
                if isinstance(stats, Mapping)
                and float(stats.get("iqr_outlier_rate") or 0) > 0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:24]
        if rate_raw:
            groups.append(
                {
                    "key": "outlier-rates",
                    "title": "IQR outlier rates by column",
                    "items": [(name, fmt_pct(rate)) for name, rate in rate_raw],
                }
            )

    profiles = (report.get("univariate") or {}).get("per_column") or {}
    if isinstance(profiles, dict) and profiles:
        non_normal = [
            name
            for name, profile in profiles.items()
            if isinstance(profile, Mapping) and profile.get("appears_non_normal") is True
        ]
        skew_raw = sorted(
            (
                (str(name), float(profile.get("skew")))
                for name, profile in profiles.items()
                if isinstance(profile, Mapping) and profile.get("skew") is not None
            ),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )[:16]
        uni_items: list[tuple[str, str]] = [
            ("columns profiled", fmt_int(len(profiles))),
            ("appears_non_normal", fmt_int(len(non_normal))),
            *((name, "non-normal flag") for name in non_normal[:16]),
        ]
        if skew_raw:
            uni_items.extend(
                (f"skew · {name}", fmt_metric(value)) for name, value in skew_raw[:8]
            )
        groups.append(
            {
                "key": "univariate",
                "title": "Univariate screens",
                "items": uni_items,
            }
        )

    drift_rows = [
        *list(drift.get("numeric_drift") or []),
        *list(drift.get("categorical_drift") or []),
    ]
    if drift.get("available") and drift_rows:
        groups.append(
            {
                "key": "drift",
                "title": "Drift screen (measured columns)",
                "items": [
                    (
                        str(row.get("column")),
                        (
                            f"flag · {fmt_metric(row.get('ks_stat', row.get('js_divergence')))}"
                            if row.get("flag")
                            else fmt_metric(row.get("ks_stat", row.get("js_divergence")))
                        ),
                    )
                    for row in drift_rows[:32]
                    if isinstance(row, Mapping)
                ],
            }
        )

    exclusions = overview.get("feature_exclusion_reasons") or {}
    if isinstance(exclusions, dict) and exclusions:
        groups.append(
            {
                "key": "exclusions",
                "title": "Feature analysis exclusions",
                "items": [
                    (str(name), str(reason))
                    for name, reason in list(exclusions.items())[:24]
                ],
            }
        )

    return groups


def build_methods_catalog(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """Analyzer family cards: ran / skipped / not_applicable with limitations."""
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    multivariate = report.get("multivariate") or {}
    bivariate = report.get("bivariate") or {}
    drift = report.get("drift") or {}
    target = report.get("target") or {}
    outliers = report.get("outliers") or {}
    profiles = (report.get("univariate") or {}).get("per_column") or {}
    normality_methods = sorted(
        {
            str(profile["normality_method"])
            for profile in profiles.values()
            if isinstance(profile, Mapping) and profile.get("normality_method")
        }
    )
    analysis_rows = overview.get("analysis_rows", overview.get("n_rows"))
    frame_rows = overview.get("n_rows")
    columns = overview.get("analysis_column_count", overview.get("n_columns"))
    eligible = len(overview.get("eligible_feature_columns") or [])
    sampled = analysis_rows != frame_rows

    cards: list[dict[str, str]] = [
        {
            "family": "Analysis scope",
            "status": "ran",
            "summary": (
                f"Examined {fmt_int(analysis_rows)} of {fmt_int(frame_rows)} rows across "
                f"{fmt_int(columns)} columns, with {fmt_int(eligible)} eligible features "
                "after role exclusions."
            ),
            "detail": (
                f"Sampling: {'yes — later screens use the analysis frame' if sampled else 'none — full frame used'}. "
                f"Mode: {overview.get('mode') or 'session'} · engine: {overview.get('engine') or 'pandas'}."
            ),
            "why": "",
        },
        {
            "family": "Quality",
            "status": "ran" if quality else "skipped",
            "summary": (
                "Screened full-frame missingness, duplicates, constants, cardinality, "
                "identifier-like columns, and mixed types."
                if quality
                else "Quality analyzer did not return a payload for this pass."
            ),
            "detail": (
                f"Missing cells: {fmt_int(quality.get('missing_cell_count'))}; "
                f"completeness: {fmt_pct(quality.get('completeness_score'))}; "
                f"duplicate rows: {fmt_int(quality.get('duplicate_row_count'))}."
                if quality
                else "No quality metrics were attached to the report dict."
            ),
            "why": "" if quality else "Analyzer section absent from report payload.",
        },
    ]

    if profiles:
        cards.append(
            {
                "family": "Univariate profiles",
                "status": "ran",
                "summary": (
                    f"Profiled {fmt_int(len(profiles))} columns for shape, cardinality, "
                    "and normality screens where eligible."
                ),
                "detail": (
                    f"Normality methods: {', '.join(normality_methods) if normality_methods else 'none applied'}."
                ),
                "why": "",
            }
        )
    else:
        cards.append(
            {
                "family": "Univariate profiles",
                "status": "skipped",
                "summary": "No per-column univariate profiles were attached.",
                "detail": "Univariate screens need eligible feature columns on the analysis frame.",
                "why": "Univariate section absent or empty.",
            }
        )

    if bivariate.get("pearson") or bivariate.get("spearman"):
        cards.append(
            {
                "family": "Correlation",
                "status": "ran",
                "summary": (
                    "Computed Pearson and Spearman matrices on eligible numeric features, "
                    "with Kendall reserved for selected leading pairs."
                ),
                "detail": (
                    f"Top |Pearson| pairs retained: "
                    f"{fmt_int(len(bivariate.get('top_abs_pearson_pairs') or []))}."
                ),
                "why": "",
            }
        )
    else:
        cards.append(
            {
                "family": "Correlation",
                "status": "not_applicable",
                "summary": "No eligible numeric correlation matrix was produced.",
                "detail": "Pearson/Spearman require at least two eligible numeric feature columns.",
                "why": "Insufficient eligible numeric features for a pairwise matrix.",
            }
        )

    cat_pairs = bivariate.get("categorical_pairs")
    if cat_pairs:
        cards.append(
            {
                "family": "Categorical association",
                "status": "ran",
                "summary": (
                    f"Estimated Cramér's V for {fmt_int(len(cat_pairs))} categorical pairs "
                    "on the analysis frame."
                ),
                "detail": "Pairwise association only; not adjusted for multiple testing.",
                "why": "",
            }
        )
    elif cat_pairs is not None:
        cards.append(
            {
                "family": "Categorical association",
                "status": "not_applicable",
                "summary": "Categorical association ran but found no eligible pairs.",
                "detail": "Needs at least two eligible categorical columns with usable levels.",
                "why": "No categorical pairs met eligibility for Cramér's V.",
            }
        )
    else:
        cards.append(
            {
                "family": "Categorical association",
                "status": "skipped",
                "summary": "Categorical association was not computed for this pass.",
                "detail": "The bivariate payload did not include a categorical_pairs section.",
                "why": "Analyzer section omitted from report payload.",
            }
        )

    mi = bivariate.get("mutual_information_vs_target") or {}
    if mi:
        cards.append(
            {
                "family": "Mutual information",
                "status": "ran",
                "summary": (
                    f"Ranked {fmt_int(len(mi))} eligible features by mutual information "
                    f"against target '{target.get('column')}'."
                ),
                "detail": "scikit-learn estimators with random_state=0. Descriptive ranking only.",
                "why": "",
            }
        )
    elif not target.get("column"):
        cards.append(
            {
                "family": "Mutual information",
                "status": "not_applicable",
                "summary": "Mutual information requires a declared target role.",
                "detail": "Assign a target with session.set_roles before expecting MI rankings.",
                "why": "No target role declared.",
            }
        )
    else:
        cards.append(
            {
                "family": "Mutual information",
                "status": "skipped",
                "summary": (
                    f"Target '{target.get('column')}' is declared, but no eligible "
                    "mutual-information result was produced."
                ),
                "detail": "Usually means no eligible features remained after role exclusions.",
                "why": "No eligible feature/target pairs for the MI estimator.",
            }
        )

    if normality_methods:
        cards.append(
            {
                "family": "Normality screening",
                "status": "ran",
                "summary": (
                    f"Applied normality screens ({', '.join(normality_methods)}) "
                    f"on {fmt_int(len(profiles))} profiled columns at alpha=0.05."
                ),
                "detail": (
                    "Unadjusted p-values are sample-size sensitive; "
                    "non-significance does not prove normality."
                ),
                "why": "",
            }
        )
    else:
        cards.append(
            {
                "family": "Normality screening",
                "status": "not_applicable",
                "summary": "No columns were eligible for normality screening.",
                "detail": "Screens require numeric columns with enough non-missing observations.",
                "why": "No eligible univariate numeric profiles with a normality method.",
            }
        )

    vif_rows = multivariate.get("vif") or []
    pca = multivariate.get("pca") or {}
    if vif_rows or pca:
        vif_n = len(vif_rows) if isinstance(vif_rows, list) else len(vif_rows or {})
        cards.append(
            {
                "family": "VIF / PCA",
                "status": "ran",
                "summary": (
                    f"Complete-case multivariate screen used "
                    f"{fmt_int(multivariate.get('complete_case_rows'))} rows, "
                    f"{fmt_int(vif_n)} VIF entries, and "
                    f"{fmt_int(pca.get('n_components'))} PCA components."
                ),
                "detail": "Complete-case numeric matrix only; not a fitted Session transform.",
                "why": "",
            }
        )
    else:
        cards.append(
            {
                "family": "VIF / PCA",
                "status": "not_applicable",
                "summary": "VIF and PCA were not produced for this pass.",
                "detail": "Both require enough complete-case numeric features.",
                "why": "Insufficient eligible complete numeric data.",
            }
        )

    per_col = outliers.get("per_column") or {}
    multi = outliers.get("multivariate") or {}
    if per_col or multi:
        cards.append(
            {
                "family": "Outliers",
                "status": "ran",
                "summary": (
                    f"Univariate IQR and |z|>3 screens covered {fmt_int(len(per_col))} columns; "
                    "Isolation Forest (random_state=0) screened complete cases when eligible."
                ),
                "detail": "Flags are screening labels, not proof of data error.",
                "why": "",
            }
        )
    else:
        cards.append(
            {
                "family": "Outliers",
                "status": "not_applicable",
                "summary": "No univariate or multivariate outlier screen was produced.",
                "detail": "Requires eligible numeric columns and enough complete cases for Isolation Forest.",
                "why": "No eligible numeric outlier payload.",
            }
        )

    if drift.get("available"):
        cards.append(
            {
                "family": "Drift",
                "status": "ran",
                "summary": (
                    f"Compared train/test partitions "
                    f"({fmt_int(drift.get('train_rows'))} / {fmt_int(drift.get('test_rows'))} rows) "
                    f"and flagged {fmt_int(len(_flagged_names(drift.get('flagged_columns'))))} columns."
                ),
                "detail": f"Settings: {fmt_metric(drift.get('settings'))}.",
                "why": "",
            }
        )
    else:
        cards.append(
            {
                "family": "Drift",
                "status": "skipped",
                "summary": "Train/test drift was not available for this pass.",
                "detail": fmt_metric(drift.get("reason", "No split available")),
                "why": str(drift.get("reason") or "No split available"),
            }
        )

    if target.get("column"):
        summary = target.get("summary") or {}
        cards.append(
            {
                "family": "Target",
                "status": "ran",
                "summary": (
                    f"Profiled target '{target.get('column')}' "
                    f"as {summary.get('type') or summary.get('task') or 'declared'}."
                ),
                "detail": "Target associations in EDA remain descriptive, not causal.",
                "why": "",
            }
        )
    else:
        cards.append(
            {
                "family": "Target",
                "status": "not_applicable",
                "summary": "No target role was declared for this pass.",
                "detail": "Target balance, MI, and target-linked screens stay unavailable until a target is set.",
                "why": "No target role declared.",
            }
        )

    plan = report.get("adaptive_plan") or []
    if plan:
        cards.append(
            {
                "family": "Adaptive figure plan",
                "status": "ran",
                "summary": f"Produced {fmt_int(len(plan))} adaptive plot specs for this frame.",
                "detail": "Specs describe recommended visuals; rendering depends on include_plots / studio charts.",
                "why": "",
            }
        )

    return cards


def build_degraded_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """Analyses that were skipped, unavailable, or empty — no empty-theater claims."""
    rows: list[dict[str, str]] = [
        {"analysis": "report warning", "reason": str(warning)}
        for warning in report.get("warnings") or []
    ]
    drift = report.get("drift") or {}
    if not drift.get("available"):
        rows.append(
            {
                "analysis": "train/test drift",
                "reason": str(drift.get("reason", "No split available")),
            }
        )
    bivariate = report.get("bivariate") or {}
    for key, label in (
        ("pearson", "Pearson matrix"),
        ("spearman", "Spearman matrix"),
        ("categorical_pairs", "Cramér's V"),
        ("mutual_information_vs_target", "Mutual information"),
        ("top_abs_pearson_pairs", "Top Pearson pairs"),
    ):
        if not bivariate.get(key):
            rows.append({"analysis": label, "reason": "No eligible result was produced"})
    multivariate = report.get("multivariate") or {}
    if not multivariate.get("pca"):
        rows.append(
            {
                "analysis": "PCA",
                "reason": "Insufficient eligible complete numeric data",
            }
        )
    if not multivariate.get("vif"):
        rows.append(
            {
                "analysis": "VIF",
                "reason": "Insufficient eligible complete numeric data",
            }
        )
    if not multivariate.get("correlation_clusters"):
        rows.append(
            {
                "analysis": "Correlation clusters",
                "reason": "No clusters above threshold or insufficient numerics",
            }
        )
    if not (report.get("target") or {}).get("column"):
        rows.append({"analysis": "Target profile", "reason": "No target role declared"})
    if not (report.get("outliers") or {}).get("multivariate"):
        rows.append(
            {
                "analysis": "Multivariate anomaly screen",
                "reason": "Insufficient complete numeric rows or features",
            }
        )
    if not (report.get("outliers") or {}).get("per_column"):
        rows.append(
            {
                "analysis": "Univariate outlier screen",
                "reason": "No eligible numeric columns",
            }
        )
    if not (report.get("univariate") or {}).get("per_column"):
        rows.append(
            {
                "analysis": "Univariate profiles",
                "reason": "No per-column profiles attached",
            }
        )
    return rows


def build_domain_briefs(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Board-level EDA insights for the readiness sheet (dataset-adaptive)."""
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    univariate = report.get("univariate") or {}
    bivariate = report.get("bivariate") or {}
    multivariate = report.get("multivariate") or {}
    target = report.get("target") or {}
    drift = report.get("drift") or {}
    outliers = report.get("outliers") or {}
    findings = list(report.get("findings") or [])

    def _findings_for(*prefixes: str) -> list[str]:
        out: list[str] = []
        for item in findings:
            key = str(item.get("key") or "")
            if any(key.startswith(prefix) for prefix in prefixes):
                title = item.get("title") or item.get("detail") or key
                sev = str(item.get("severity") or "info")
                out.append(f"{sev}: {title}")
        return out[:6]

    briefs: list[dict[str, Any]] = []

    if quality:
        constants = _flagged_names(quality.get("constant_columns"))
        id_like = _flagged_names(quality.get("id_like_columns"))
        briefs.append(
            {
                "key": "quality",
                "title": "Data quality",
                "board": "quality",
                "status": "ran",
                "summary": (
                    f"Completeness {fmt_pct(quality.get('completeness_score'))} · "
                    f"{fmt_int(quality.get('missing_cell_count'))} missing cells · "
                    f"{fmt_int(quality.get('duplicate_row_count'))} duplicate rows."
                ),
                "highlights": [
                    f"{fmt_int(len(constants))} constant columns"
                    + (f" ({', '.join(constants[:3])})" if constants else ""),
                    f"{fmt_int(len(id_like))} identifier-like columns"
                    + (f" ({', '.join(id_like[:3])})" if id_like else ""),
                    f"{fmt_int(len(quality.get('high_cardinality_columns') or []))} high-cardinality flags",
                    f"{fmt_int(len(quality.get('quasi_constant_columns') or []))} near-constant flags",
                ],
                "findings": _findings_for("quality."),
                "metrics": [
                    {"k": "completeness", "v": fmt_pct(quality.get("completeness_score"))},
                    {"k": "missing cells", "v": fmt_int(quality.get("missing_cell_count"))},
                    {"k": "duplicates", "v": fmt_int(quality.get("duplicate_row_count"))},
                ],
            }
        )

    profiles = univariate.get("per_column") or {}
    if profiles:
        non_normal = sum(
            1
            for profile in profiles.values()
            if isinstance(profile, Mapping) and profile.get("appears_non_normal") is True
        )
        briefs.append(
            {
                "key": "features",
                "title": "Feature profiles",
                "board": "features",
                "status": "ran",
                "summary": (
                    f"{fmt_int(len(profiles))} columns profiled · "
                    f"{fmt_int(non_normal)} flagged non-normal."
                ),
                "highlights": [
                    f"{fmt_int(len(overview.get('numeric_columns') or []))} numeric columns in frame",
                    f"{fmt_int(len(overview.get('categorical_columns') or []))} categorical columns in frame",
                    f"{fmt_int(non_normal)} normality flags (unadjusted)",
                ],
                "findings": _findings_for("univariate.", "quality.high_cardinality"),
                "metrics": [
                    {"k": "profiled", "v": fmt_int(len(profiles))},
                    {"k": "non-normal", "v": fmt_int(non_normal)},
                ],
            }
        )

    if bivariate:
        mi = bivariate.get("mutual_information_vs_target") or {}
        top_mi = None
        if isinstance(mi, dict) and mi:
            top_name, top_score = max(mi.items(), key=lambda kv: float(kv[1] or 0))
            top_mi = f"top MI · {top_name} = {fmt_metric(top_score)}"
        pearson_n = len(bivariate.get("top_abs_pearson_pairs") or [])
        cramers_n = len(bivariate.get("categorical_pairs") or [])
        briefs.append(
            {
                "key": "relationships",
                "title": "Relationships",
                "board": "relationships",
                "status": "ran",
                "summary": (
                    f"{fmt_int(pearson_n)} leading |Pearson| pairs · "
                    f"{fmt_int(cramers_n)} Cramér's V pairs · "
                    f"{fmt_int(len(mi) if isinstance(mi, dict) else 0)} MI ranks."
                ),
                "highlights": [
                    *( [top_mi] if top_mi else [] ),
                    f"Pearson matrix {'present' if bivariate.get('pearson') else 'absent'}",
                    f"Spearman matrix {'present' if bivariate.get('spearman') else 'absent'}",
                ],
                "findings": _findings_for("relationships."),
                "metrics": [
                    {"k": "|Pearson| pairs", "v": fmt_int(pearson_n)},
                    {"k": "Cramér's V pairs", "v": fmt_int(cramers_n)},
                    {"k": "MI features", "v": fmt_int(len(mi) if isinstance(mi, dict) else 0)},
                ],
            }
        )

    if multivariate.get("vif") or multivariate.get("pca") or multivariate.get("correlation_clusters"):
        vif = multivariate.get("vif") or []
        vif_n = len(vif) if isinstance(vif, list) else len(vif or {})
        pca = multivariate.get("pca") or {}
        cum = (pca.get("cumulative_explained_variance") or [None])[-1]
        briefs.append(
            {
                "key": "multivariate",
                "title": "Multivariate structure",
                "board": "multivariate",
                "status": "ran",
                "summary": (
                    f"{fmt_int(vif_n)} VIF entries · "
                    f"{fmt_int(pca.get('n_components'))} PCA components · "
                    f"{fmt_int(len(multivariate.get('correlation_clusters') or []))} clusters."
                ),
                "highlights": [
                    f"complete-case rows {fmt_int(multivariate.get('complete_case_rows'))}",
                    f"cumulative variance explained {fmt_pct(cum)}" if cum is not None else "PCA variance unavailable",
                ],
                "findings": _findings_for("multivariate."),
                "metrics": [
                    {"k": "VIF entries", "v": fmt_int(vif_n)},
                    {"k": "PCA components", "v": fmt_int(pca.get("n_components"))},
                    {"k": "clusters", "v": fmt_int(len(multivariate.get("correlation_clusters") or []))},
                ],
            }
        )

    if target.get("column"):
        summary = target.get("summary") or {}
        class_counts = summary.get("class_counts") or {}
        balance_bits = []
        if isinstance(class_counts, dict) and class_counts:
            total = sum(int(v) for v in class_counts.values()) or 1
            for label, count in list(class_counts.items())[:4]:
                balance_bits.append(f"{label}={fmt_pct(int(count) / total, 1)}")
        drift_flags = _flagged_names(drift.get("flagged_columns"))
        briefs.append(
            {
                "key": "target",
                "title": "Target & validation",
                "board": "target",
                "status": "ran",
                "summary": (
                    f"Target '{target.get('column')}' · "
                    f"{summary.get('type') or summary.get('task') or 'declared'}"
                    + (f" · {' / '.join(balance_bits)}" if balance_bits else "")
                    + (
                        f" · drift flags {fmt_int(len(drift_flags))}"
                        if drift.get("available")
                        else " · drift n/a"
                    )
                ),
                "highlights": [
                    *(balance_bits[:4]),
                    (
                        f"drift train/test {fmt_int(drift.get('train_rows'))}/"
                        f"{fmt_int(drift.get('test_rows'))}"
                        if drift.get("available")
                        else str(drift.get("reason") or "no split drift")
                    ),
                ],
                "findings": _findings_for("target.", "validation."),
                "metrics": [
                    {"k": "target", "v": str(target.get("column"))},
                    {"k": "drift flags", "v": fmt_int(len(drift_flags))},
                ],
            }
        )

    if outliers.get("per_column") or outliers.get("multivariate"):
        per_col = outliers.get("per_column") or {}
        multi = outliers.get("multivariate") or {}
        flagged = multi.get(
            "anomaly_count",
            multi.get("flagged_row_count", multi.get("n_flagged", multi.get("flagged"))),
        )
        briefs.append(
            {
                "key": "outliers",
                "title": "Outliers & anomalies",
                "board": "outliers",
                "status": "ran",
                "summary": (
                    f"{fmt_int(len(per_col))} columns with univariate screens"
                    + (
                        f" · {fmt_int(flagged)} multivariate anomalies flagged"
                        if flagged is not None
                        else ""
                    )
                ),
                "highlights": [
                    "IQR and |z|>3 are screening labels, not automatic drops",
                    (
                        f"anomaly rate {fmt_pct(multi.get('anomaly_rate'))}"
                        if multi.get("anomaly_rate") is not None
                        else "multivariate screen unavailable or empty"
                    ),
                ],
                "findings": _findings_for("outliers."),
                "metrics": [
                    {"k": "univariate cols", "v": fmt_int(len(per_col))},
                    {"k": "anomalies", "v": fmt_int(flagged) if flagged is not None else "—"},
                ],
            }
        )

    return briefs
