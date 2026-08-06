"""Turn analyzer numbers into claims, each carrying the evidence behind it.

The analyzers produce measurements. This produces statements about them: "12%
of rows are incomplete" becomes a finding with a severity, an affected-column
list, and a pointer back to the number it came from.

The evidence link is the part that makes this trustworthy rather than
authoritative. Every finding carries the value it was derived from, the analyzer
section it came from, and the limitations of that measurement. A reader who
doubts a claim can follow it back rather than take it on faith, which is the
only honest way to present automated interpretation.

Two deliberate constraints. Findings are conservative: a threshold is crossed or
it is not, and nothing is inferred beyond what was measured. And recommendations
derive from finding *keys*, not from the raw numbers: so the condition for a
recommendation is written once, in the finding, and cannot drift out of sync
with the advice it produces.

See Also
--------
buildml.explain : The ``Finding``, ``Evidence``, and ``Recommendation`` types.
buildml.eda.profile : Where these are assembled into a report.
"""

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
    """Read the analyzer sections and state what they show, with citations.

    Walks the sections an EDA pass produced and emits a finding wherever a
    threshold was crossed or a fact is worth stating outright. Each finding
    carries a stable key, a title, a detail sentence, a severity, the columns
    affected, and the evidence: the actual value, its source section, and what
    that measurement cannot tell you.

    Every finding is generated from a measurement that was taken. Nothing is
    inferred, nothing is extrapolated, and a section that is missing simply
    produces no findings rather than a guess.

    Parameters
    ----------
    sections:
        The analyzer outputs, keyed by section name: ``overview``,
        ``quality``, ``bivariate``, ``multivariate``, ``target``, ``outliers``,
        ``drift``. Missing sections are skipped, so a partial pass works.
        Findings are emitted only when the corresponding measurement exists
        (for example MI requires a target; PCA requires enough complete numerics).

    Returns
    -------
    list of Finding
        In the order the sections were examined, not by severity. Always
        includes ``eda.scope``, which records how many rows were examined out of
        how many exist: the first thing to check before believing anything
        else.

    Notes
    -----
    **The keys are stable and are the API.** Recommendations, tests, and reports
    all reference findings by key, so a key is not renamed once published.

    **Severity is threshold-based, and thresholds are conventions.**
    ``FindingSeverity.INFO`` means measured and unremarkable, not unmeasured.

    **Read ``evidence.limitations``.** It is where a finding says what it cannot
    support: that an association is not causation, that a p-value is
    unadjusted, that a screen was run on a sample.

    See Also
    --------
    build_recommendations : Turning these into next steps.
    """
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

    for key, title, severity, detail, field in (
        (
            "quality.constants",
            "Constant columns",
            FindingSeverity.MEDIUM,
            "Constant columns contain no observed variation in this dataset.",
            "constant_columns",
        ),
        (
            "quality.identifiers",
            "Identifier-like columns",
            FindingSeverity.MEDIUM,
            "Identifier-like columns have near-unique observed values and are "
            "not valid default predictors.",
            "id_like_columns",
        ),
        (
            "quality.near_constant",
            "Near-constant columns",
            FindingSeverity.LOW,
            "Near-constant columns are dominated by a single observed level.",
            "quasi_constant_columns",
        ),
        (
            "quality.high_cardinality",
            "High-cardinality categoricals",
            FindingSeverity.MEDIUM,
            "High-cardinality categorical columns can explode encodings and "
            "fragment rare levels.",
            "high_cardinality_columns",
        ),
        (
            "quality.mixed_types",
            "Mixed-type suspect columns",
            FindingSeverity.LOW,
            "Text columns mix numeric-looking and non-numeric values, which "
            "usually signals sentinels or concatenated sources.",
            "mixed_type_suspect_columns",
        ),
    ):
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

    pattern_hits = quality.get("string_pattern_hints") or {}
    hygiene_cols = [
        column
        for column, rates in pattern_hits.items()
        if any(float(rates.get(name) or 0) >= 0.2 for name in ("email_rate", "url_rate", "phone_rate"))
        or float(rates.get("blank_like_rate") or 0) >= 0.1
    ]
    if hygiene_cols:
        findings.append(
            Finding(
                key="quality.text_hygiene",
                title="String hygiene signals",
                detail=(
                    f"{len(hygiene_cols)} text column(s) show elevated email/URL/phone "
                    f"or blank-like rates. Observed columns: {', '.join(hygiene_cols[:10])}."
                ),
                severity=FindingSeverity.LOW,
                evidence=(
                    _evidence(
                        "quality.string_patterns",
                        "String pattern hint rates",
                        {column: pattern_hits[column] for column in hygiene_cols[:20]},
                        "quality.string_pattern_hints",
                        ("Pattern detection samples text columns; rare tokens may be missed.",),
                    ),
                ),
                affected_columns=tuple(hygiene_cols),
            )
        )

    duplicate_count = int(quality.get("duplicate_row_count") or 0)
    if duplicate_count:
        findings.append(
            Finding(
                key="quality.duplicates",
                title="Duplicate rows",
                detail=(
                    f"{duplicate_count:,} exact duplicate rows were observed "
                    f"({float(quality.get('duplicate_row_rate') or 0):.3%} of the frame)."
                ),
                severity=(
                    FindingSeverity.HIGH
                    if float(quality.get("duplicate_row_rate") or 0) >= 0.05
                    else FindingSeverity.MEDIUM
                ),
                evidence=(
                    _evidence(
                        "quality.duplicate_rows",
                        "Duplicate row count and rate",
                        {
                            "duplicate_row_count": duplicate_count,
                            "duplicate_row_rate": quality.get("duplicate_row_rate"),
                        },
                        "quality.duplicate_row_count",
                        ("Only exact full-row duplicates are counted.",),
                    ),
                ),
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

    top_pairs = bivariate.get("top_abs_pearson_pairs") or []
    strong_pairs = [
        pair
        for pair in top_pairs
        if abs(float(pair.get("corr") or 0)) >= 0.8
        and str(pair.get("a")) != str(pair.get("b"))
    ]
    if strong_pairs:
        lead = strong_pairs[0]
        findings.append(
            Finding(
                key="relationships.correlated_pairs",
                title="Strongly correlated feature pairs",
                detail=(
                    f"{len(strong_pairs)} eligible numeric pair(s) reached |Pearson|≥0.8; "
                    f"strongest was '{lead.get('a')}' vs '{lead.get('b')}' "
                    f"({float(lead.get('corr') or 0):.6g})."
                ),
                severity=FindingSeverity.MEDIUM,
                evidence=(
                    _evidence(
                        "relationships.pearson_pairs",
                        "Top absolute Pearson pairs",
                        strong_pairs[:20],
                        "bivariate.top_abs_pearson_pairs",
                        (
                            "Pearson captures linear co-movement and is outlier-sensitive.",
                            "Association is not evidence of causality.",
                        ),
                    ),
                ),
                affected_columns=tuple(
                    {
                        str(lead.get("a")),
                        str(lead.get("b")),
                        *(
                            str(pair.get("a"))
                            for pair in strong_pairs[:6]
                            if pair.get("a") is not None
                        ),
                        *(
                            str(pair.get("b"))
                            for pair in strong_pairs[:6]
                            if pair.get("b") is not None
                        ),
                    }
                ),
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

    clusters = multivariate.get("correlation_clusters") or []
    if clusters:
        findings.append(
            Finding(
                key="multivariate.clusters",
                title="Correlation clusters",
                detail=(
                    f"{len(clusters)} correlation cluster(s) were formed among eligible "
                    f"numeric features (largest size {max(len(group) for group in clusters)})."
                ),
                severity=FindingSeverity.INFO,
                evidence=(
                    _evidence(
                        "multivariate.correlation_clusters",
                        "Correlation clusters",
                        clusters,
                        "multivariate.correlation_clusters",
                        ("Clusters link pairs above threshold via union-find.",),
                    ),
                ),
                affected_columns=tuple(
                    str(column)
                    for group in clusters[:4]
                    for column in group[:6]
                ),
            )
        )

    pca = multivariate.get("pca") or {}
    variance = pca.get("explained_variance_ratio") or []
    if variance:
        findings.append(
            Finding(
                key="multivariate.pca",
                title="PCA variance screen",
                detail=(
                    f"PCA on complete-case numeric features reported "
                    f"{len(variance)} component(s); first component explained "
                    f"{float(variance[0]):.3%} of variance."
                ),
                severity=FindingSeverity.INFO,
                evidence=(
                    _evidence(
                        "multivariate.pca.variance",
                        "PCA explained variance",
                        pca,
                        "multivariate.pca",
                        (
                            "Variance explained is not predictive utility.",
                            f"Based on {multivariate.get('complete_case_rows', 0)} complete rows.",
                        ),
                    ),
                ),
            )
        )

    per_column_outliers = outliers.get("per_column") or {}
    hot_outlier_cols = [
        (column, float(stats.get("iqr_outlier_rate") or 0))
        for column, stats in per_column_outliers.items()
        if float(stats.get("iqr_outlier_rate") or 0) >= 0.05
    ]
    hot_outlier_cols.sort(key=lambda item: item[1], reverse=True)
    if hot_outlier_cols:
        lead_col, lead_rate = hot_outlier_cols[0]
        findings.append(
            Finding(
                key="outliers.univariate",
                title="Univariate outlier screen",
                detail=(
                    f"{len(hot_outlier_cols)} numeric column(s) had IQR-fence rates "
                    f"≥5%; highest was '{lead_col}' at {lead_rate:.3%}."
                ),
                severity=(
                    FindingSeverity.MEDIUM
                    if lead_rate >= 0.15
                    else FindingSeverity.LOW
                ),
                evidence=(
                    _evidence(
                        "outliers.iqr_rates",
                        "IQR outlier rates",
                        {
                            column: per_column_outliers[column]
                            for column, _ in hot_outlier_cols[:20]
                        },
                        "outliers.per_column",
                        (
                            "IQR fences are a screening convention, not confirmed errors.",
                            "Skewed columns naturally flag heavy tails.",
                        ),
                    ),
                ),
                affected_columns=tuple(column for column, _ in hot_outlier_cols[:12]),
            )
        )

    mv = outliers.get("multivariate") or {}
    if mv:
        rate = float(mv.get("anomaly_rate", 0))
        anomaly_count = int(mv.get("anomaly_count", mv.get("flagged_row_count", 0)) or 0)
        scored = int(mv.get("n_rows_scored", mv.get("scored_row_count", 0)) or 0)
        findings.append(
            Finding(
                key="outliers.multivariate",
                title="Multivariate anomaly screen",
                detail=(
                    f"Isolation Forest marked {anomaly_count:,} of "
                    f"{scored:,} scored rows ({rate:.3%}) "
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
    """Propose next steps, keyed to the findings that justify them.

    Reads the findings by key rather than re-examining the data. That is a
    design choice worth naming: if this re-tested the raw numbers, the condition
    for "missing data is a problem" would exist in two places and would
    eventually disagree with itself. Here a recommendation exists exactly when
    its finding does, and ``based_on`` records the link.

    Recommendations carry a priority. ``BEFORE_MODELING`` marks things that
    change what the data means: imputation, dropping constants, keeping
    identifiers out of the feature matrix, investigating drift. ``NEXT`` marks
    improvements that can wait.

    When nothing triggered, a recommendation is still returned: confirm the
    validation design and proceed. An empty list would read as "nothing to do
    here", and there is always something to do.

    Parameters
    ----------
    findings:
        The output of :func:`build_findings`.

    Returns
    -------
    list of Recommendation
        Each with a key, title, rationale, priority, the finding keys it rests
        on, and any caveats. Never empty.

    Notes
    -----
    **A recommendation is a prompt, not an instruction.** "Review correlated
    feature groups" means look; it does not mean drop columns. The caveats say
    what to weigh.

    **Only some findings produce recommendations.** A finding at ``INFO``
    severity is a measurement worth stating, not a problem worth acting on.

    See Also
    --------
    build_findings : Produces the input.
    """
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
    if "quality.duplicates" in by_key:
        recommendations.append(
            Recommendation(
                key="next.deduplicate",
                title="Fix the table grain before splitting",
                rationale=by_key["quality.duplicates"].detail,
                priority=ActionPriority.BEFORE_MODELING,
                based_on=("quality.duplicates",),
                caveats=("Confirm intended grain; exact-row duplicates are only one form of duplication.",),
            )
        )
    if (
        "quality.high_cardinality" in by_key
        and "quality.identifiers" not in by_key
    ):
        recommendations.append(
            Recommendation(
                key="next.high_cardinality",
                title="Group or encode high-cardinality categoricals carefully",
                rationale=by_key["quality.high_cardinality"].detail,
                priority=ActionPriority.BEFORE_MODELING,
                based_on=("quality.high_cardinality",),
                caveats=("Fit grouping or target encoding on training folds only.",),
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
    elif "relationships.correlated_pairs" in by_key:
        recommendations.append(
            Recommendation(
                key="next.correlated_pairs",
                title="Resolve near-duplicate feature pairs",
                rationale=by_key["relationships.correlated_pairs"].detail,
                priority=ActionPriority.NEXT,
                based_on=("relationships.correlated_pairs",),
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
    if "outliers.univariate" in by_key and by_key["outliers.univariate"].severity != FindingSeverity.INFO:
        recommendations.append(
            Recommendation(
                key="next.outliers",
                title="Inspect IQR-fence outliers before excluding any",
                rationale=by_key["outliers.univariate"].detail,
                priority=ActionPriority.NEXT,
                based_on=("outliers.univariate",),
                caveats=("Skewed columns naturally flag heavy tails; do not drop blindly.",),
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
    """Flatten findings to their detail sentences, dropping the structure.

    A compatibility shim for code written before findings carried evidence,
    severity, and column lists. It returns the sentences and discards
    everything else.

    Parameters
    ----------
    findings:
        The findings to flatten.

    Returns
    -------
    list of str
        The detail sentences, in order.

    Notes
    -----
    **Prefer the findings themselves in new code.** The severity tells a reader
    what matters, and the evidence lets them check it; a bare sentence has
    neither, and a report built from these strings cannot be filtered or sorted.

    See Also
    --------
    build_findings : The structured form.
    """
    return [finding.detail for finding in findings]


def recommendation_view(recommendations: list[Recommendation]) -> list[str]:
    """Flatten recommendations to ``"title: rationale"`` strings.

    The counterpart shim to :func:`narrative_view`, for callers predating
    structured recommendations. Priority, caveats, and the ``based_on`` links
    are dropped.

    Parameters
    ----------
    recommendations:
        The recommendations to flatten.

    Returns
    -------
    list of str
        One ``"title: rationale"`` line each, in order.

    Notes
    -----
    **The caveats are lost, and they are the important part.** "Review
    correlated feature groups" without "compare regularization or feature
    reduction with validation" reads as an instruction to delete columns. Use
    the structured form where you can.

    See Also
    --------
    build_recommendations : The structured form.
    """
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
