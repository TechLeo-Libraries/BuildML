"""Evidence-linked interpretations for exploratory analysis outputs."""

from __future__ import annotations

from typing import Any

from buildml.explain import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)


def build_findings(sections: dict[str, Any]) -> list[Finding]:
    """Build conservative findings whose claims point to analyzer evidence."""
    overview = sections.get("overview", {})
    quality = sections.get("quality", {})
    bivariate = sections.get("bivariate", {})
    multivariate = sections.get("multivariate", {})
    target = sections.get("target", {})
    outliers = sections.get("outliers", {})
    drift = sections.get("drift", {})
    findings = [
        Finding(
            key="eda.scope",
            title="Analysis scope",
            detail=(
                f"EDA examined {overview.get('analysis_rows', 0):,} of "
                f"{overview.get('n_rows', 0):,} rows across "
                f"{overview.get('n_columns', 0):,} columns."
            ),
            evidence=(
                _evidence(
                    "eda.scope.rows",
                    "Dataset and analysis row counts",
                    {
                        "dataset_rows": overview.get("n_rows"),
                        "analysis_rows": overview.get("analysis_rows"),
                        "columns": overview.get("n_columns"),
                    },
                    "overview",
                    ("Sampled analyses may not reproduce full-data tail behavior.",),
                ),
            ),
        )
    ]

    missing = int(quality.get("missing_cell_count", 0))
    findings.append(
        Finding(
            key="quality.completeness",
            title="Observed completeness",
            detail=(
                f"{missing:,} cells are missing; observed cell completeness is "
                f"{float(quality.get('completeness_score', 1.0)):.3%}."
            ),
            severity=FindingSeverity.MEDIUM if missing else FindingSeverity.INFO,
            evidence=(
                _evidence(
                    "quality.missing_cells",
                    "Missing cell count and completeness",
                    {
                        "missing_cell_count": missing,
                        "completeness_score": quality.get("completeness_score"),
                        "missing_rate_by_column": quality.get("missing_rate_by_column", {}),
                    },
                    "quality",
                    ("Missingness mechanisms were not inferred.",),
                ),
            ),
            affected_columns=tuple(
                column
                for column, rate in quality.get("missing_rate_by_column", {}).items()
                if rate
            ),
        )
    )

    for key, title, severity, detail in (
        (
            "quality.constants",
            "Constant columns",
            FindingSeverity.MEDIUM,
            "Constant columns contain no observed variation in this dataset.",
        ),
        (
            "quality.identifiers",
            "Identifier-like columns",
            FindingSeverity.MEDIUM,
            "Identifier-like columns have near-unique observed values and are "
            "not valid default predictors.",
        ),
    ):
        field = "constant_columns" if key.endswith("constants") else "id_like_columns"
        columns = tuple(map(str, quality.get(field, [])))
        if columns:
            findings.append(
                Finding(
                    key=key,
                    title=title,
                    detail=f"{detail} Observed columns: {', '.join(columns[:10])}.",
                    severity=severity,
                    evidence=(
                        _evidence(
                            f"{key}.columns",
                            title,
                            list(columns),
                            f"quality.{field}",
                        ),
                    ),
                    affected_columns=columns,
                )
            )

    mi = bivariate.get("mutual_information_vs_target") or {}
    if mi:
        feature, value = next(iter(mi.items()))
        findings.append(
            Finding(
                key="relationships.mi_leader",
                title="Highest measured mutual information",
                detail=(
                    f"Among eligible features, '{feature}' had the highest estimated "
                    f"mutual information with the target ({value:.6g}). This is an "
                    "association measure, not evidence of causality."
                ),
                evidence=(
                    _evidence(
                        "relationships.mi",
                        "Mutual-information ranking",
                        mi,
                        "bivariate.mutual_information_vs_target",
                        (
                            "Estimator values depend on encoding, sample, and "
                            "random-state settings.",
                            "Mutual information does not establish direction or causality.",
                        ),
                    ),
                ),
                affected_columns=(str(feature),),
            )
        )

    vif = multivariate.get("vif") or []
    if vif:
        top = vif[0]
        value = float(top.get("vif", 0))
        findings.append(
            Finding(
                key="relationships.vif",
                title="Largest variance inflation factor",
                detail=(
                    f"'{top.get('column')}' has VIF={value:.6g} among complete-case "
                    "eligible numeric features."
                ),
                severity=FindingSeverity.HIGH if value >= 10 else (
                    FindingSeverity.MEDIUM if value >= 5 else FindingSeverity.INFO
                ),
                evidence=(
                    _evidence(
                        "relationships.vif.rows",
                        "VIF estimates",
                        vif,
                        "multivariate.vif",
                        (
                            f"Based on {multivariate.get('complete_case_rows', 0)} complete rows.",
                            "VIF is sensitive to the included feature set.",
                        ),
                    ),
                ),
                affected_columns=(str(top.get("column")),),
            )
        )

    mv = outliers.get("multivariate") or {}
    if mv:
        rate = float(mv.get("anomaly_rate", 0))
        findings.append(
            Finding(
                key="outliers.multivariate",
                title="Multivariate anomaly screen",
                detail=(
                    f"Isolation Forest marked {int(mv.get('anomaly_count', 0)):,} of "
                    f"{int(mv.get('n_rows_scored', 0)):,} scored rows ({rate:.3%}) "
                    "as anomalies."
                ),
                severity=FindingSeverity.MEDIUM if rate > 0.05 else FindingSeverity.INFO,
                evidence=(
                    _evidence(
                        "outliers.isolation_forest",
                        "Isolation Forest screen",
                        mv,
                        "outliers.multivariate",
                        ("Anomaly labels are screening signals, not confirmed data errors.",),
                    ),
                ),
            )
        )

    if drift.get("available"):
        flagged = drift.get("flagged_columns") or []
        findings.append(
            Finding(
                key="validation.drift",
                title="Train/test distribution comparison",
                detail=(
                    f"{len(flagged):,} eligible columns met the configured drift flag "
                    "thresholds."
                ),
                severity=FindingSeverity.HIGH if flagged else FindingSeverity.INFO,
                evidence=(
                    _evidence(
                        "validation.drift.rows",
                        "Flagged drift rows",
                        flagged,
                        "drift.flagged_columns",
                        (
                            "Statistical shift does not identify its cause.",
                            "Multiple-testing adjustment was not applied.",
                        ),
                    ),
                ),
                affected_columns=tuple(str(row.get("column")) for row in flagged),
            )
        )

    summary = target.get("summary") or {}
    if summary:
        target_type = str(summary.get("type", "target"))
        findings.append(
            Finding(
                key="target.summary",
                title="Target profile",
                detail=f"Observed {target_type.replace('_', ' ')} for '{target.get('column')}'.",
                evidence=(
                    _evidence(
                        "target.summary.values",
                        "Target summary statistics",
                        summary,
                        "target.summary",
                        ("Target associations are descriptive and do not establish causality.",),
                    ),
                ),
                affected_columns=(str(target.get("column")),),
            )
        )
    return findings


def build_recommendations(findings: list[Finding]) -> list[Recommendation]:
    """Derive next steps from finding keys rather than duplicate conditions."""
    by_key = {finding.key: finding for finding in findings}
    recommendations: list[Recommendation] = []
    if (
        "quality.completeness" in by_key
        and by_key["quality.completeness"].severity != FindingSeverity.INFO
    ):
        recommendations.append(
            Recommendation(
                key="next.impute",
                title="Define a train-fitted missing-data strategy",
                rationale="Missing values were observed in one or more columns.",
                priority=ActionPriority.BEFORE_MODELING,
                action=Action("action.impute", "Fit imputation on training data", "impute"),
                based_on=("quality.completeness",),
                caveats=("Choose methods by feature meaning and missingness pattern.",),
            )
        )
    for finding_key, rec_key, title in (
        ("quality.constants", "next.drop_constants", "Exclude constant columns"),
        ("quality.identifiers", "next.exclude_ids", "Keep identifiers outside feature matrices"),
    ):
        if finding_key in by_key:
            recommendations.append(
                Recommendation(
                    key=rec_key,
                    title=title,
                    rationale=by_key[finding_key].detail,
                    priority=ActionPriority.BEFORE_MODELING,
                    based_on=(finding_key,),
                )
            )
    if (
        "relationships.vif" in by_key
        and by_key["relationships.vif"].severity != FindingSeverity.INFO
    ):
        recommendations.append(
            Recommendation(
                key="next.collinearity",
                title="Review correlated feature groups",
                rationale="The VIF screen found an elevated estimate.",
                priority=ActionPriority.NEXT,
                based_on=("relationships.vif",),
                caveats=("Compare regularization or feature reduction with validation.",),
            )
        )
    if "validation.drift" in by_key and by_key["validation.drift"].severity != FindingSeverity.INFO:
        recommendations.append(
            Recommendation(
                key="next.drift",
                title="Investigate flagged train/test shifts",
                rationale="Configured distribution-shift thresholds were met.",
                priority=ActionPriority.BEFORE_MODELING,
                based_on=("validation.drift",),
                caveats=("Check split construction and time/group effects before changing data.",),
            )
        )
    if not recommendations:
        recommendations.append(
            Recommendation(
                key="next.validate",
                title="Proceed with leakage-safe validation",
                rationale="No finding generated a configured blocking recommendation.",
                priority=ActionPriority.NEXT,
                action=Action("action.split", "Confirm validation design", "split"),
                based_on=("eda.scope",),
            )
        )
    return recommendations


def narrative_view(findings: list[Finding]) -> list[str]:
    """Compatibility view for callers that consume narrative strings."""
    return [finding.detail for finding in findings]


def recommendation_view(recommendations: list[Recommendation]) -> list[str]:
    """Compatibility view for callers that consume recommendation strings."""
    return [f"{item.title}: {item.rationale}" for item in recommendations]


def _evidence(
    key: str,
    summary: str,
    value: Any,
    source: str,
    limitations: tuple[str, ...] = (),
) -> Evidence:
    return Evidence(
        key=key,
        kind=EvidenceKind.METRIC,
        summary=summary,
        value=value,
        source=source,
        limitations=limitations,
    )
