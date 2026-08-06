# ruff: noqa: E501
"""Adaptive teaching depth for Command Cockpit readiness-sheet sidebar.

Ledger groups, assumption footnotes, and register rows carry beginner→advanced
pedagogy, evidence from *this* report, calculations when relevant, and
copy-paste Session examples. Nothing here is persisted — teaching is pure
payload enrichment for the live App / offline bundle.
"""

from __future__ import annotations

from typing import Any

from buildml.dashboard.adapt import (
    build_adapt_context,
    fmt_n,
    list_names,
    plural,
)

# Plain-language glossary: what each ledger group is and why it sits on the sheet.
LEDGER_GROUP_GLOSSARY: dict[str, dict[str, str]] = {
    "frame": {
        "means": (
            "Shape of the analysis frame — rows analysed vs loaded, column count, "
            "eligible features, missing cells, engine, and sampling disclosure."
        ),
        "why_on_sheet": (
            "Every later metric is relative to this frame. If sampling or engine "
            "limits apply, the ledger records them so audits do not invent full-frame coverage."
        ),
    },
    "roles": {
        "means": (
            "Column role counts (feature / target / id / ignore) plus finding "
            "severity tallies for this pass."
        ),
        "why_on_sheet": (
            "Roles decide which columns enter modeling screens; severity tallies "
            "show how noisy the triage board is before you open individual findings."
        ),
    },
    "missing": {
        "means": "Per-column missing rates for columns with any nulls, plus a complete-column remainder.",
        "why_on_sheet": (
            "Missingness drives imputation risk and leakage. The ledger keeps the "
            "measured rates so you can re-check after the next extract."
        ),
    },
    "quality-flags": {
        "means": (
            "Aggregate quality counters: duplicates, constants, near-constants, "
            "high-cardinality, identifier-like, mixed-type suspects, and rows with any missing."
        ),
        "why_on_sheet": (
            "These flags are the machine-checkable quality contract. Counts here "
            "must match the Quality board before you trust a modeling path."
        ),
    },
    "mi": {
        "means": "Mutual information of each eligible feature against the declared target.",
        "why_on_sheet": (
            "Ranks association strength with the label under the MI estimator used "
            "this pass — descriptive triage, not causal proof."
        ),
    },
    "pearson": {
        "means": "Strongest absolute Pearson correlations among numeric feature pairs.",
        "why_on_sheet": (
            "Surfaces linear co-movement that can inflate variance or create "
            "redundant predictors before you fit."
        ),
    },
    "spearman": {
        "means": "Strongest absolute Spearman rank correlations among numeric pairs.",
        "why_on_sheet": (
            "Catches monotonic (not only linear) association the Pearson ledger may miss."
        ),
    },
    "cramers": {
        "means": "Cramér's V association scores for categorical feature pairs.",
        "why_on_sheet": (
            "Categorical redundancy does not show up in Pearson. This group is the "
            "audit trail for those associations."
        ),
    },
    "kendall": {
        "means": "Leading Kendall τ pairs among ordinal/numeric rankings.",
        "why_on_sheet": (
            "A rank-concordance screen that is more robust to outliers than Pearson."
        ),
    },
    "vif": {
        "means": "Variance inflation factors on the complete-case numeric matrix.",
        "why_on_sheet": (
            "Quantifies multicollinearity risk for linear-style models; complete-case "
            "row count is part of the audit trail."
        ),
    },
    "clusters": {
        "means": "Correlation clusters of features that move together.",
        "why_on_sheet": (
            "Groups that behave as one signal — useful when deciding what to drop or combine."
        ),
    },
    "pca": {
        "means": "PCA explained-variance ratios and cumulative coverage on complete cases.",
        "why_on_sheet": (
            "Shows how much numeric variance collapses into a few components under "
            "this frame's complete-case subset."
        ),
    },
    "screens": {
        "means": (
            "Target declaration, task type, class balance or regression moments, "
            "association/screen counts, anomaly tallies, and drift availability."
        ),
        "why_on_sheet": (
            "One place to verify the modeling contract (target + task) and the "
            "headline screen counts before diving into domain boards."
        ),
    },
    "outlier-rates": {
        "means": "Per-column IQR outlier rates for numeric features that exceeded zero.",
        "why_on_sheet": (
            "Documents which columns carry heavy tail mass under the IQR rule used this pass."
        ),
    },
    "univariate": {
        "means": (
            "Univariate profile counts, non-normal flags, and leading absolute skew values."
        ),
        "why_on_sheet": (
            "Records distributional screens that inform transforms and model family choice."
        ),
    },
    "drift": {
        "means": "Train/test drift statistics for columns measured this pass.",
        "why_on_sheet": (
            "If a split exists, this is the audit trail of distribution shift before "
            "you trust holdout scores."
        ),
    },
    "exclusions": {
        "means": (
            "Columns withheld from feature analysis with the heuristic or role reason "
            "that excluded them."
        ),
        "why_on_sheet": (
            "Silent exclusions change the eligible set. The ledger names each column "
            "and reason so audits cannot assume every column was scored."
        ),
    },
    "skipped": {
        "means": "Analyzers that were skipped or not applicable for this frame.",
        "why_on_sheet": (
            "Absence of a figure is not silence — skipped analyzers are listed so "
            "the next extract does not inherit a false sense of coverage."
        ),
    },
}


def _names(items: list[Any], limit: int = 3) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("column") or item.get("feature") or item.get("k") or "")
        else:
            name = str(item)
        if name and name not in out:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def _quote_list(names: list[str]) -> str:
    if not names:
        return '["your_column"]'
    return "[" + ", ".join(f'"{n}"' for n in names) + "]"


def _ingest_header(ctx: dict[str, Any]) -> str:
    rows = fmt_n(ctx.get("n_rows") or ctx.get("analysis_rows") or 0)
    cols = ctx.get("n_columns") or 0
    engine = ctx.get("engine") or "pandas"
    return (
        f"# Adaptive to this session: {rows} rows, {cols} columns, engine={engine}\n"
        "from buildml import Session\n"
        "# Assume `frame` is your DataFrame (or pass a path to Session.ingest).\n"
    )


def _feature_sample(ctx: dict[str, Any], limit: int = 3) -> list[str]:
    eligible = ctx.get("eligible_features") or []
    names = _names(list(eligible) if not isinstance(eligible, list) else eligible, limit)
    if names:
        return names
    cols = ctx.get("columns") or []
    return _names(list(cols), limit) or ["feature_a"]


def _levels(beginner: str, intermediate: str, advanced: str) -> dict[str, str]:
    return {
        "beginner": beginner,
        "intermediate": intermediate,
        "advanced": advanced,
    }


def _example(
    *,
    summary: str,
    code: str,
    change_these: list[str],
    flexible: list[str],
    reading: str,
) -> dict[str, Any]:
    return {
        "summary": summary,
        "code": code.rstrip() + "\n",
        "change_these": change_these,
        "flexible": flexible,
        "reading": reading,
    }


def _calc(
    *,
    label: str,
    formula: str,
    inputs: dict[str, Any],
    result: str,
    reading: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "formula": formula,
        "inputs": inputs,
        "result": result,
        "reading": reading,
    }


def _glossary_for(key: str) -> dict[str, str]:
    return dict(LEDGER_GROUP_GLOSSARY.get(key) or {})


def ledger_purpose_copy() -> dict[str, str]:
    """Section 03 purpose blurb for the readiness sheet."""
    return {
        "title": "Ledger — every computed number",
        "purpose": (
            "The ledger is the audit trail of computed EDA numbers for this frame. "
            "Each group is a family of measured values (counts, rates, associations, "
            "exclusions). Jump chips scroll within this sheet — they are not domain boards."
        ),
        "how_to_use": (
            "Scan groups that match your risk, open a group to learn what the numbers mean, "
            "then drill a metric when you need the Session call that produced it."
        ),
    }


def assumptions_purpose_copy() -> dict[str, str]:
    """Section 02 purpose blurb for the readiness sheet."""
    return {
        "title": "What each finding assumes",
        "purpose": (
            "Every automated finding rests on assumptions — what the number means, "
            "why it matters before modeling, and what to check next. This section "
            "collects those footnotes so the register never floats free of its caveats."
        ),
        "how_to_use": (
            "Filter by theme, open a footnote for Means / Matters, then use the sidebar "
            "for beginner→advanced depth and a Session example bound to this report."
        ),
    }


def _ledger_calc(key: str, items: list[dict[str, str]], ctx: dict[str, Any]) -> dict[str, Any] | None:
    if key == "frame":
        by_k = {it.get("k"): it.get("v") for it in items}
        return _calc(
            label="Frame coverage",
            formula="rows analysed / rows in frame",
            inputs={
                "rows_analysed": by_k.get("rows analysed", "—"),
                "rows_in_frame": by_k.get("rows in frame", "—"),
                "columns": by_k.get("columns", "—"),
            },
            result=str(by_k.get("sampling") or "—"),
            reading=(
                "If sampling is not “none disclosed”, later rates describe the analysed "
                "subset, not necessarily every loaded row."
            ),
        )
    if key == "missing" and items:
        top = items[0]
        return _calc(
            label="Highest missing rate (this pass)",
            formula="missing(column) / n_rows_analysed",
            inputs={"column": top.get("k") or "—", "rate": top.get("v") or "—"},
            result=f"{top.get('k')} → {top.get('v')}",
            reading="Treat rates near 1.0 as near-empty columns for modeling eligibility.",
        )
    if key == "mi" and items:
        top = items[0]
        return _calc(
            label="Leading mutual information vs target",
            formula="MI(feature; target) under the session estimator",
            inputs={
                "feature": top.get("k") or "—",
                "mi": top.get("v") or "—",
                "target": ctx.get("target_column") or "undeclared",
            },
            result=f"{top.get('k')} → {top.get('v')}",
            reading="Higher MI means stronger dependence under this estimator — not causation.",
        )
    if key == "quality-flags":
        by_k = {it.get("k"): it.get("v") for it in items}
        return _calc(
            label="Quality flag roll-up",
            formula="count(flagged columns) by quality heuristic",
            inputs={
                "constant_columns": by_k.get("constant columns", "0"),
                "identifier_like": by_k.get("identifier-like columns", "0"),
                "duplicate_rows": by_k.get("duplicate rows", "0"),
            },
            result=(
                f"constants={by_k.get('constant columns', '0')}, "
                f"id-like={by_k.get('identifier-like columns', '0')}"
            ),
            reading="Non-zero constants and id-like columns usually leave the eligible feature set.",
        )
    if key == "exclusions" and items:
        return _calc(
            label="Feature analysis exclusions",
            formula="column ∉ eligible_features because reason(column)",
            inputs={
                "excluded_shown": str(len(items)),
                "example_column": items[0].get("k") or "—",
                "example_reason": items[0].get("v") or "—",
            },
            result=f"{len(items)} exclusion {plural(len(items), 'row')} listed",
            reading=(
                "Excluded columns do not contribute to MI/VIF/association screens. "
                "Re-role or clean them before expecting them in eligible features."
            ),
        )
    if key == "screens":
        by_k = {it.get("k"): it.get("v") for it in items}
        return _calc(
            label="Target contract",
            formula="target column + task type from roles / declaration",
            inputs={
                "target_column": by_k.get("target column") or ctx.get("target_column") or "—",
                "task": by_k.get("task") or ctx.get("task") or "—",
            },
            result=f"{by_k.get('target column', '—')} · {by_k.get('task', '—')}",
            reading="If target is undeclared, MI-vs-target and class-balance screens cannot close.",
        )
    return None


def _ledger_example(key: str, items: list[dict[str, str]], ctx: dict[str, Any]) -> dict[str, Any]:
    header = _ingest_header(ctx)
    feats = _feature_sample(ctx, 3)
    target = ctx.get("target_column")
    roles = {f: "feature" for f in feats}
    if target:
        roles[str(target)] = "target"
    role_literal = ",\n    ".join(f'"{k}": "{v}"' for k, v in roles.items())

    if key == "exclusions":
        cols = [it.get("k") for it in items[:3] if it.get("k")]
        sample = cols[0] if cols else "excluded_column"
        return _example(
            summary=(
                f"This frame excluded {len(items)} column(s) from feature analysis "
                f"(e.g. {sample}). Re-role or clean before expecting them in screens."
            ),
            code=(
                f"{header}"
                "session = Session.ingest(frame)\n"
                f"session = session.set_roles({{\n    {role_literal},\n}})\n"
                f'# Inspect why a column was excluded — example: "{sample}"\n'
                "report = session.eda(include_plots=False, show=False)\n"
                "reasons = (report.to_dict().get('overview') or {}).get('feature_exclusion_reasons') or {}\n"
                f'print(reasons.get("{sample}", "not listed — check roles / constants"))\n'
            ),
            change_these=[
                f'Replace roles with your real columns (sample features: {_quote_list(feats)}).',
                f'Swap "{sample}" for the exclusion you are investigating.',
            ],
            flexible=[
                "Drop include_plots=True only when you need figures.",
                "Add .split(...) before eda() when drift screens should run.",
            ],
            reading=(
                "feature_exclusion_reasons is the machine-readable twin of this ledger group. "
                "BuildML does not persist your sidebar reading."
            ),
        )

    if key in {"mi", "pearson", "spearman", "cramers", "vif"}:
        return _example(
            summary=(
                f"Rebuild the readiness sheet and inspect the “{key}” ledger family "
                "after declaring roles (and a target when required)."
            ),
            code=(
                f"{header}"
                "session = (\n"
                "    Session.ingest(frame)\n"
                f"    .set_roles({{\n    {role_literal},\n}})\n"
                "    .split(test_size=0.25, random_state=0)\n"
                ")\n"
                "report = session.eda(include_plots=False, show=False)\n"
                "# App: open Command cockpit → section 03 → this ledger group.\n"
                "# Or Static: session.eda(...).to_html('eda.html')\n"
                "print(session.last_eda.findings[:3] if session.last_eda else [])\n"
            ),
            change_these=[
                "Point Session.ingest at your frame or path.",
                f"Set roles for your columns; target sample: {target or 'declare a target'}.",
            ],
            flexible=[
                "Stratify classification splits when class balance allows.",
                "Use eda_app() for the interactive cockpit instead of to_html.",
            ],
            reading="Association ledgers describe co-occurrence under this frame — not causal effect.",
        )

    if key == "drift":
        return _example(
            summary="Drift ledger rows appear only when a train/test split exists on the Session.",
            code=(
                f"{header}"
                "session = (\n"
                "    Session.ingest(frame)\n"
                f"    .set_roles({{\n    {role_literal},\n}})\n"
                "    .split(test_size=0.25, random_state=0)\n"
                ")\n"
                "report = session.eda(include_plots=False, show=False).to_dict()\n"
                "drift = report.get('drift') or {}\n"
                "print('available', drift.get('available'), 'flags', drift.get('flagged_columns'))\n"
            ),
            change_these=[
                "Keep .split(...) if you need train/test drift screens.",
                "Adjust test_size to your protocol.",
            ],
            flexible=["Omit split only when a single-frame descriptive pass is intentional."],
            reading="No split ⇒ drift group omitted (empty theater), not a silent all-clear.",
        )

    if key == "skipped":
        return _example(
            summary="Skipped analyzers are listed so missing figures are explained, not hidden.",
            code=(
                f"{header}"
                "session = Session.ingest(frame).set_roles({\n"
                f"    {role_literal},\n"
                "})\n"
                "report = session.eda(include_plots=False, show=False).to_dict()\n"
                "# Cockpit section 08 + ledger “Skipped / not applicable analyzers”\n"
                "print((report.get('overview') or {}).get('skipped_analyzers') or 'none recorded')\n"
            ),
            change_these=["Declare the target/roles that unlock N/A analyzers you care about."],
            flexible=["Some analyzers stay N/A for unsupervised frames — that is expected."],
            reading="A skipped analyzer is an explicit coverage gap, not a green light.",
        )

    # Default frame / quality / screens / univariate / etc.
    return _example(
        summary=(
            f"Reproduce this cockpit ledger group (“{key}”) from a Session EDA pass "
            "bound to your frame."
        ),
        code=(
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({{\n    {role_literal},\n}})\n"
            ")\n"
            "handle = session.eda_app(open_browser=True)  # Command cockpit\n"
            "# Or: report = session.eda(include_plots=False, show=False)\n"
            "print(handle.url)\n"
        ),
        change_these=[
            "Replace `frame` with your DataFrame or ingest path.",
            f"Update set_roles to your schema (features sample: {_quote_list(feats)}).",
        ],
        flexible=[
            "Use eda(...).to_html(...) for a Static twin of the ledger.",
            "Pass port=... to eda_app when 8765 is busy.",
        ],
        reading=(
            "The cockpit sheet and Static EDA share ledger builders; numbers should match "
            "for the same report dict."
        ),
    )


def build_ledger_group_teaching(
    *,
    key: str,
    title: str,
    items: list[dict[str, str]],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Full pedagogical payload for one ledger group (sidebar)."""
    gloss = _glossary_for(key)
    means = gloss.get("means") or f"Computed metrics grouped as “{title}” for this EDA pass."
    why = gloss.get("why_on_sheet") or (
        "Kept on the readiness sheet so every computed number has an auditable home."
    )
    sample_keys = [str(it.get("k")) for it in items[:5] if it.get("k")]
    sample_line = ", ".join(sample_keys) if sample_keys else "no rows in this group"
    n = len(items)
    target = ctx.get("target_column")
    evidence = (
        f"This report lists {n} {plural(n, 'metric')} under “{title}”. "
        f"Sample labels: {sample_line}."
    )
    if target:
        evidence += f" Declared target for this session: {target}."
    if ctx.get("analysis_rows") and ctx.get("n_rows"):
        evidence += (
            f" Frame coverage: {fmt_n(ctx.get('analysis_rows'))} / "
            f"{fmt_n(ctx.get('n_rows'))} rows analysed."
        )

    beginner = (
        f"“{title}” is a ledger folder: {means} Read the labels on the left and the "
        "measured values on the right — nothing here is a model score."
    )
    intermediate = (
        f"Why this folder is on the readiness sheet: {why} Compare these numbers to the "
        "matching domain board (Quality, Features, Relationships, Target) before you change roles."
    )
    advanced = (
        "Ledger groups are produced by shared sheet_coverage builders used by both the "
        "App cockpit and Static EDA. Empty theater is omitted: if a group is absent, its "
        "analyzer produced no values for this frame (or was skipped — see the skipped group)."
    )

    next_checks = [
        f"Confirm the {n} listed values match your expectation for this extract.",
        "Open the related domain board if you need full tables behind a metric.",
        "Re-run session.eda after role or cleaning changes and re-check this group.",
    ]
    if key == "exclusions":
        next_checks.insert(
            0,
            "For each excluded column, decide: re-role, clean, or accept exclusion.",
        )
    if key == "missing":
        next_checks.insert(0, "Name an imputation or drop policy for high-missing columns.")

    return {
        "kind": "ledger_group",
        "key": key,
        "title": title,
        "means": means,
        "why_on_sheet": why,
        "why_it_matters": why,
        "beginner": beginner,
        "evidence": evidence,
        "levels": _levels(beginner, intermediate, advanced),
        "calculation": _ledger_calc(key, items, ctx),
        "worked_example": _ledger_example(key, items, ctx),
        "next_checks": next_checks,
        "item_count": n,
        "sample_items": items[:12],
        "session_note": (
            "Sidebar readings are ephemeral in this browser tab. BuildML never saves "
            "cockpit judgments to the Session, history, or disk."
        ),
    }


def build_ledger_metric_teaching(
    *,
    group_key: str,
    group_title: str,
    metric_key: str,
    metric_value: str,
    ctx: dict[str, Any],
    group_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Teaching payload focused on one ledger row inside a group."""
    base = build_ledger_group_teaching(
        key=group_key,
        title=group_title,
        items=list(group_items or [{"k": metric_key, "v": metric_value}]),
        ctx=ctx,
    )
    base["kind"] = "ledger_metric"
    base["metric_key"] = metric_key
    base["metric_value"] = metric_value
    base["beginner"] = (
        f"Metric “{metric_key}” = {metric_value} inside ledger group “{group_title}”. "
        f"{base['means']}"
    )
    base["evidence"] = (
        f"From this report’s “{group_title}” group: {metric_key} → {metric_value}. "
        + (f"Target={ctx.get('target_column')}." if ctx.get("target_column") else "")
    )
    return base


def build_assumption_teaching(note: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Teaching payload for one section-02 assumption footnote."""
    slug = str(note.get("slug") or note.get("concept_key") or "assumption")
    means = str(note.get("means") or "")
    matters = str(note.get("matters") or "")
    nxt = str(note.get("next") or "")
    technical = str(note.get("technical") or "")
    evidence = str(note.get("evidence") or "")
    theme = str(note.get("theme") or "general")
    feats = _feature_sample(ctx, 2)
    target = ctx.get("target_column")
    header = _ingest_header(ctx)
    roles = {f: "feature" for f in feats}
    if target:
        roles[str(target)] = "target"
    role_literal = ",\n    ".join(f'"{k}": "{v}"' for k, v in roles.items())

    beginner = means or f"This footnote explains what finding theme “{theme}” assumes."
    intermediate = matters or "The assumption bounds how far you can push the finding into a modeling decision."
    advanced = technical or (
        "Technical caveats live with the finding evidence keys; verify both the prose "
        "and the underlying analyzer table."
    )

    return {
        "kind": "assumption",
        "key": slug,
        "title": slug,
        "theme": theme,
        "means": means,
        "why_it_matters": matters,
        "beginner": beginner,
        "evidence": evidence
        or (
            "Footnote bound to this session"
            + (f" (target={target})" if target else "")
            + f"; theme={theme}."
        ),
        "levels": _levels(beginner, intermediate, advanced),
        "calculation": None,
        "worked_example": _example(
            summary=f"Revisit assumption “{slug}” after refreshing EDA on your frame.",
            code=(
                f"{header}"
                "session = Session.ingest(frame).set_roles({\n"
                f"    {role_literal},\n"
                "})\n"
                "report = session.eda(include_plots=False, show=False)\n"
                f'# Open cockpit section 02 → footnote “{slug}”\n'
                "print([f.get('key') for f in (report.findings or [])[:5]])\n"
            ),
            change_these=[
                "Bind roles to your schema.",
                f'Look up concept "{slug}" in the Academy if you need the formal note.',
            ],
            flexible=["Use session.learn(...) when the Academy key matches this slug."],
            reading=nxt or "Check-next lines on the card are the operational follow-up.",
        ),
        "next_checks": [nxt] if nxt else ["Re-read the linked finding after the next ingest."],
        "session_note": (
            "Assumption footnotes are derived from findings for this report only. "
            "Nothing you mark in the UI is written back."
        ),
    }


def build_finding_teaching(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Teaching payload for a findings-register row click."""
    key = str(row.get("key") or "")
    title = str(row.get("title") or key)
    detail = str(row.get("detail") or "")
    evidence = str(row.get("evidence") or "")
    sev = str(row.get("severity") or "info")
    cols = list(row.get("affected_columns") or [])
    col_line = list_names(cols, 4) or "no column list on finding"
    feats = _feature_sample(ctx, 2)
    target = ctx.get("target_column")
    header = _ingest_header(ctx)
    roles = {f: "feature" for f in feats}
    if target:
        roles[str(target)] = "target"
    role_literal = ",\n    ".join(f'"{k}": "{v}"' for k, v in roles.items())

    beginner = (
        f"Finding {key} ({sev}): {title}. In plain terms — {detail or 'see evidence line'}."
    )
    intermediate = (
        f"Affected columns / evidence: {col_line}. Evidence pointer: {evidence or 'report'}."
    )
    advanced = (
        "Findings are triage signals with severity, not automatic drops. Confirm the "
        "analyzer table and assumption footnote before changing production logic."
    )

    return {
        "kind": "finding",
        "key": key,
        "title": title,
        "severity": sev,
        "means": detail,
        "why_it_matters": (
            f"Severity {sev} on the readiness register — resolve blockers before modeling bake-offs."
        ),
        "beginner": beginner,
        "evidence": evidence or detail,
        "levels": _levels(beginner, intermediate, advanced),
        "calculation": None,
        "worked_example": _example(
            summary=f"Locate finding {key} after an EDA pass on your frame.",
            code=(
                f"{header}"
                "session = Session.ingest(frame).set_roles({\n"
                f"    {role_literal},\n"
                "})\n"
                "report = session.eda(include_plots=False, show=False)\n"
                f"hit = [f for f in report.findings if f.get('key') == '{key}']\n"
                "print(hit[0] if hit else 'finding not raised on this frame')\n"
            ),
            change_these=[
                "Use your roles / target.",
                "Swap the finding key if you are investigating a different register row.",
            ],
            flexible=["Open the Academy concept chip on the register when one is linked."],
            reading="Recommendations in section 04 name Session operations; they do not auto-run.",
        ),
        "next_checks": [
            "Read the matching assumption footnote in section 02.",
            "Follow a section 04 recommendation only after you accept the caveat.",
        ],
        "affected_columns": cols[:12],
        "session_note": (
            "Register clicks open teaching only. BuildML never persists gate or cockpit marks."
        ),
    }


def enrich_ledger_group(
    group: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Attach glossary + teaching to a sheet ledger group."""
    key = str(group.get("key") or "group")
    title = str(group.get("title") or "Group")
    items: list[dict[str, str]] = []
    for col in group.get("cols") or []:
        for it in col.get("items") or []:
            items.append({"k": str(it.get("k")), "v": str(it.get("v"))})
    if not items:
        for it in group.get("items") or []:
            if isinstance(it, dict):
                items.append({"k": str(it.get("k")), "v": str(it.get("v"))})
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                items.append({"k": str(it[0]), "v": str(it[1])})
    gloss = _glossary_for(key)
    enriched = dict(group)
    enriched["means"] = gloss.get("means") or ""
    enriched["why_on_sheet"] = gloss.get("why_on_sheet") or ""
    enriched["teaching"] = build_ledger_group_teaching(
        key=key, title=title, items=items, ctx=ctx
    )
    return enriched


def enrich_assumption_note(note: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(note)
    enriched["teaching"] = build_assumption_teaching(note, ctx)
    return enriched


def enrich_register_row(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["teaching"] = build_finding_teaching(row, ctx)
    return enriched


def enrich_cockpit_sheet(sheet: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Attach adaptive teaching + purpose blurbs to a cockpit sheet payload."""
    ctx = sheet.get("adapt") or build_adapt_context(report)
    out = dict(sheet)
    out["assumptions_purpose"] = assumptions_purpose_copy()
    out["ledger_purpose"] = ledger_purpose_copy()
    out["ledger_glossary"] = {
        key: dict(meta) for key, meta in LEDGER_GROUP_GLOSSARY.items()
    }
    out["assumptions"] = [
        enrich_assumption_note(note, ctx) for note in (sheet.get("assumptions") or [])
    ]
    out["ledger"] = [
        enrich_ledger_group(group, ctx) for group in (sheet.get("ledger") or [])
    ]
    out["register"] = [
        enrich_register_row(row, ctx) for row in (sheet.get("register") or [])
    ]
    return out
