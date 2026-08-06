"""Assumption footnotes for Static EDA findings.

Short, scannable teaching notes derived from finding keys and the concrete
evidence on *this* report. This is not Concept Academy: there is no curriculum
navigation, only footnotes attached to measured findings.

Each note uses a fixed scan structure:

* What this means — plain language
* Why it matters — modeling / validation consequence
* What to check next — concrete follow-up
* Technical note — precise estimator / scope caveat
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_THEME_BY_PREFIX: dict[str, str] = {
    "eda": "Scope",
    "quality": "Data quality",
    "relationships": "Relationships",
    "multivariate": "Relationships",
    "outliers": "Outliers & anomalies",
    "target": "Target",
    "validation": "Validation & drift",
}

_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# slug, means, matters, next_check, technical
_NOTE_TEMPLATES: dict[str, tuple[str, str, str, str, str]] = {
    "eda.scope": (
        "data-splitting",
        "This pass reports how many rows and columns were actually examined, "
        "which may be a sample or a column-budgeted subset of the full frame.",
        "Every later metric, association, and screen is conditioned on that scope. "
        "Tail behavior and rare levels can look different on the full frame.",
        "Compare analysed rows to frame rows, and read any sampling or column-budget "
        "warnings before treating a finding as frame-wide.",
        "Association and multivariate screens typically run on the analysis frame; "
        "quality checks usually run on the full frame. Sampled analyses may not "
        "reproduce full-data tail behavior.",
    ),
    "quality.completeness": (
        "missing-data",
        "Some cells are empty. Completeness is the share of non-missing cells "
        "observed in the frame; it does not say why values are missing.",
        "Most learners cannot ingest raw gaps. How you fill or drop them changes "
        "what the model sees, and train-fitted rules must not peek at holdout rows.",
        "Inspect missing rate by column and whether incomplete rows cluster. "
        "Choose a train-fitted imputation or drop strategy before fitting.",
        "Imputation fills gaps with a train-learned rule; it does not prove "
        "missingness is harmless. Missingness mechanisms (MCAR/MAR/MNAR) were "
        "not inferred.",
    ),
    "quality.identifiers": (
        "column-roles",
        "One or more columns look like keys: nearly unique values per row, so "
        "they identify records rather than describe them.",
        "Left as features, identifiers let a model memorise training rows and "
        "produce optimistic validation scores that collapse in production.",
        "Confirm each flagged column is truly a key or high-cardinality label, "
        "then assign role 'id' or 'ignore' before feature matrices are built.",
        "Roles label how each column participates in modeling, not what its "
        "dtype happens to be. Identifier heuristics use distinctness thresholds "
        "and can flag legitimate high-cardinality features.",
    ),
    "quality.constants": (
        "column-roles",
        "One or more columns take a single observed value across the frame.",
        "A constant column adds width without information and can confuse "
        "scaling, encoding, and importance rankings.",
        "Exclude constants with role 'ignore', or confirm they are intentional "
        "sentinels before leaving them in the feature set.",
        "Constant detection uses the observed frame including missingness "
        "encoding choices. Near-constant columns are reported separately.",
    ),
    "quality.near_constant": (
        "constant-and-near-constant",
        "One or more columns are dominated by a single level (about 95% or more "
        "of rows share one value).",
        "Trees can still split on the rare level and overfit a handful of rows; "
        "linear models gain almost no signal.",
        "Inspect the rare level: is it a real minority class, a sentinel, or "
        "noise? Decide to keep, group, or ignore before encoding.",
        "Quasi-constant threshold is a convention (≥95% modal share), not a "
        "proof that the column is useless in every task.",
    ),
    "quality.duplicates": (
        "duplicate-records",
        "Some rows are exact duplicates of other rows in the frame.",
        "Duplicates inflate apparent sample size and can leak across a random "
        "split if copies land in both train and test.",
        "Decide the intended table grain (one row per entity/event). Deduplicate "
        "or aggregate before splitting when the grain is wrong.",
        "Only exact full-row duplicates are counted here. Soft duplicates "
        "(same key, differing fields) need a key-aware check.",
    ),
    "quality.high_cardinality": (
        "high-cardinality",
        "A categorical column has a very large number of distinct levels "
        "relative to the row count.",
        "Naïve one-hot encoding explodes width and fragments rare levels into "
        "unstable coefficients or splits.",
        "Consider grouping rare levels, target encoding with train-only fitting, "
        "or treating the column as an identifier if it is a key.",
        "Cardinality thresholds are heuristic. Grouping or target encoding must "
        "be fit on training folds only to avoid leakage.",
    ),
    "quality.mixed_types": (
        "measurement-units-and-ranges",
        "A text-like column mixes values that look numeric with values that do not.",
        "That usually means sentinels, concatenated sources, or a numeric field "
        "stored as text — all of which break silent casting.",
        "Inspect unique non-numeric tokens, replace sentinels, and cast with an "
        "explicit rule before modeling.",
        "Detection samples text columns and flags when roughly 5–95% of values "
        "look numeric. It is a hygiene screen, not a type inference.",
    ),
    "quality.text_hygiene": (
        "text-hygiene",
        "A text column shows many values that look like emails, URLs, phone numbers, "
        "or blank-like strings.",
        "Those patterns often mark PII or unclean categories that should not be "
        "naive features without explicit consent and cleaning.",
        "Inspect the column, strip/normalize strings, and decide whether the field "
        "belongs in modeling at all.",
        "Pattern rates are estimated on sampled text values (capped columns/rows).",
    ),
    "relationships.vif": (
        "variance-inflation",
        "At least one numeric feature is highly linearly predictable from the "
        "other numeric features included in the screen.",
        "Linear models then have unstable coefficients; correlated features also "
        "compete in importance rankings and confuse ablations.",
        "Review the correlated group: drop, combine, regularize, or reduce "
        "dimensions, and confirm the choice with validation.",
        "VIF estimates how much linear dependence inflates coefficient variance. "
        "It is sensitive to the included feature set and uses complete cases only.",
    ),
    "relationships.mi_leader": (
        "mutual-information",
        "Among eligible features, one shared the most estimated information with "
        "the target under the mutual-information screen.",
        "That ranks what to inspect first. It does not prove the feature should "
        "be kept, nor that it causes the target.",
        "Compare the leader with nearby ranks, check leakage and availability at "
        "score time, and validate any selection on held-out folds.",
        "Scores shared information without assuming linearity. Values depend on "
        "encoding, sample, and estimator random_state. No direction or causality.",
    ),
    "relationships.correlated_pairs": (
        "correlation",
        "Two numeric features move together strongly under Pearson correlation "
        "on the analysis frame.",
        "Near-duplicate features waste capacity and can make coefficient signs "
        "and importances brittle.",
        "Inspect the pair scatter or ranks, then decide whether to keep one, "
        "engineer a combined signal, or use regularization / PCA.",
        "Pearson captures linear co-movement and is outlier-sensitive. Compare "
        "with Spearman when relationships may be monotone but curved.",
    ),
    "outliers.multivariate": (
        "diagnostic-uncertainty",
        "A multivariate anomaly screen labeled some complete-case rows as unusual "
        "relative to the fitted Isolation Forest.",
        "Unusual is not wrong. Deleting flagged rows without review can remove "
        "the very cases the model should learn.",
        "Inspect flagged rows against domain rules. Treat the rate as a screening "
        "signal shaped by contamination settings, not as an error census.",
        "Metrics and anomaly labels are sample estimates. Isolation Forest "
        "contamination controls how many points are marked.",
    ),
    "outliers.univariate": (
        "outlier-screens",
        "One or more numeric columns have values beyond the IQR fence "
        "(1.5×IQR past the quartiles) on the analysis frame.",
        "For skewed variables the upper fence flags a large natural tail. "
        "Blind capping or dropping can distort the distribution the model needs.",
        "Compare IQR flags with skew and z-score counts, then decide per column "
        "whether to keep, winsorize on train, or transform.",
        "IQR fences are distribution-free but still a convention. Z-score > 3 "
        "assumes roughly symmetric scale and is pulled by extremes.",
    ),
    "target.summary": (
        "class-imbalance",
        "The declared target was profiled: class balance for classification, or "
        "shape statistics for a continuous target.",
        "Target shape drives metric choice, split strategy, and whether a trivial "
        "baseline already looks strong.",
        "For classification, compute majority-class accuracy as a floor. For "
        "regression, check skew before choosing squared-error losses.",
        "Unequal class frequencies change what accuracy and default thresholds "
        "can tell you. Associations with features remain descriptive, not causal.",
    ),
    "validation.drift": (
        "dataset-drift",
        "Train and test partitions were compared column-wise; some columns met "
        "configured shift thresholds (or none did).",
        "If the split populations differ systematically, holdout scores stop "
        "estimating future performance and start measuring the split artifact.",
        "Check split construction, time order, and group leakage before changing "
        "or reweighting data.",
        "Statistical shift does not identify its cause. Multiple-testing "
        "adjustment is not applied to per-column screens.",
    ),
    "multivariate.clusters": (
        "correlation",
        "Numeric features formed clusters of pairwise linear correlation above "
        "the configured threshold.",
        "A cluster is a redundancy map: members often carry overlapping signal "
        "and should be reviewed as a group, not one-by-one in isolation.",
        "Pick representatives, engineer composites, or reduce dimensions, then "
        "re-check VIF and validation metrics.",
        "Clusters are built by linking pairs above threshold (union-find). "
        "A–C may share a cluster via B even when A and C are weakly correlated.",
    ),
    "multivariate.pca": (
        "dimensionality-reduction",
        "PCA was fit on complete-case standardized numeric features and reported "
        "explained-variance ratios.",
        "Variance explained is not predictive utility. Components fit on the full "
        "frame can leak structure into training if reused carelessly.",
        "If you reduce dimensions for modeling, fit PCA inside training folds "
        "and choose component count with validation.",
        "PCA here is a diagnostic on the analysis complete-case matrix with "
        "standardized columns. It is not a fitted Session transform until you "
        "call reduce_dimensions on train.",
    ),
}


def note_for_finding(finding: Mapping[str, Any]) -> dict[str, str]:
    """Build a structured assumption note for one finding.

    Parameters
    ----------
    finding:
        Finding dict from :meth:`~buildml.eda.report.EDAReport.to_dict`
        (``key``, ``detail``, ``evidence``, …).

    Returns
    -------
    dict
        ``slug``, ``means``, ``matters``, ``next``, ``technical``, and
        ``evidence`` (concrete report evidence line when available).
    """
    key = str(finding.get("key") or "finding")
    template = _NOTE_TEMPLATES.get(key)
    if template is None:
        # Prefix fallback for future keys (e.g. quality.*) without skipping them.
        family = key.split(".", 1)[0] if "." in key else key
        for candidate, value in _NOTE_TEMPLATES.items():
            if candidate.startswith(f"{family}."):
                template = value
                break
    if template is None:
        limitations = _limitations(finding)
        slug = key.replace(".", "-")
        means = (
            str(finding.get("detail") or finding.get("title") or "A measured result.")
        )
        matters = (
            "Automated findings restate analyzer measurements; domain meaning "
            "still has to be confirmed."
        )
        next_check = "Trace the evidence source and confirm the column roles before acting."
        technical = (
            " ".join(limitations[:2])
            if limitations
            else "This finding restates a measured analyzer result without additional causal claims."
        )
    else:
        slug, means, matters, next_check, technical = template
        # Prefer target-distribution framing for regression targets.
        if key == "target.summary":
            detail = str(finding.get("detail") or "").lower()
            if "regression" in detail:
                slug = "target-distribution"
                means = (
                    "The declared continuous target was summarised (location, spread, skew) "
                    "on the analysed rows."
                )
                matters = (
                    "Skew and heavy tails decide whether squared-error metrics and "
                    "untransformed targets are appropriate."
                )
                next_check = (
                    "Compare mean vs median and skew; consider a transform or robust "
                    "metric before model search."
                )
                technical = (
                    "Regression-target associations in EDA are descriptive. "
                    "They do not establish causality or feature importance under validation."
                )

    evidence_line = _evidence_line(finding)
    return {
        "key": key,
        "slug": slug,
        "theme": theme_for_key(key),
        "means": means,
        "matters": matters,
        "next": next_check,
        "technical": technical,
        "evidence": evidence_line,
    }


def theme_for_key(key: str) -> str:
    """Return the scannable theme label for a finding key family."""
    family = key.split(".", 1)[0] if "." in key else key
    return _THEME_BY_PREFIX.get(family, "Other")


def _limitations(finding: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for item in finding.get("evidence") or []:
        if not isinstance(item, Mapping):
            continue
        out.extend(str(part) for part in (item.get("limitations") or []))
    return out


def _evidence_line(finding: Mapping[str, Any]) -> str:
    """Compact, dataset-specific evidence sentence for the footnote."""
    bits: list[str] = []
    detail = finding.get("detail")
    if detail:
        bits.append(str(detail).rstrip("."))
    columns = [str(col) for col in (finding.get("affected_columns") or []) if col]
    if columns:
        shown = ", ".join(columns[:8])
        more = len(columns) - min(len(columns), 8)
        suffix = f" (+{more} more)" if more > 0 else ""
        bits.append(f"Affected columns: {shown}{suffix}")
    evidence = finding.get("evidence") or []
    if evidence and isinstance(evidence[0], Mapping):
        source = evidence[0].get("source") or evidence[0].get("key")
        if source:
            bits.append(f"Source: {source}")
        limitations = [str(part) for part in (evidence[0].get("limitations") or [])]
        if limitations:
            bits.append(limitations[0].rstrip("."))
    return ". ".join(bits) + ("." if bits else "")


def unique_notes(
    findings: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return ordered unique notes (by slug) for the findings register.

    Each note includes ``theme`` (family group) and the strongest
    ``severity`` observed among findings that share the slug.
    """
    ordered: list[dict[str, str]] = []
    by_slug: dict[str, dict[str, str]] = {}
    for finding in findings:
        note = note_for_finding(finding)
        slug = note["slug"]
        severity = str(finding.get("severity") or "info").lower()
        existing = by_slug.get(slug)
        if existing is None:
            note["severity"] = severity
            by_slug[slug] = note
            ordered.append(note)
            continue
        prior = str(existing.get("severity") or "info").lower()
        if _SEV_RANK.get(severity, 9) < _SEV_RANK.get(prior, 9):
            existing["severity"] = severity
    for index, note in enumerate(ordered, start=1):
        note["n"] = str(index)
    return ordered
