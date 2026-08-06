"""Cockpit readiness-sheet payload shaped like the Industry redesign prototypes.

Maps a live Session EDA report into the numbered-spine structures used by
``redesigning_eda/Current EDA design overview/EDA Sheet - Cockpit.dc.html``:
KPI strip, findings register, deep assumption footnotes, full ledger groups,
recommended sequence, domain briefs, figures meta, methods/limitations, and
skipped/degraded rows.

Analytic depth is shared with Static EDA via :mod:`buildml.eda.sheet_coverage`
so App and Static stay aligned. Gate human marks are never part of this
payload — they stay UI-only in the browser.

Copy is dataset-adaptive via :mod:`buildml.dashboard.adapt` — never bound to a
demo/churn schema.
"""

from __future__ import annotations

from typing import Any

from buildml._version import __version__
from buildml.dashboard.adapt import (
    build_adapt_context,
    fmt_n as _fmt_n,
    fmt_pct as _fmt_pct,
    list_names,
    plural,
    what_to_change,
)
from buildml.dashboard.charts import charts_for_cockpit_report
from buildml.dashboard.cockpit_teaching import enrich_cockpit_sheet
from buildml.dashboard.gates import FINDING_CONCEPT_SLUG, resolve_concept_key
from buildml.eda.assumption_notes import unique_notes
from buildml.eda.sheet_coverage import (
    build_degraded_rows,
    build_domain_briefs,
    build_ledger_groups,
    build_methods_catalog,
)
from buildml.explain.concepts import CONCEPT_NOTES


def _sev_bucket(severity: str) -> str:
    s = str(severity or "info").lower()
    if s in {"critical", "crit", "high"}:
        return "blocking"
    if s in {"medium", "med"}:
        return "med"
    if s == "low":
        return "low"
    return "info"


def _sev_label(severity: str) -> str:
    s = str(severity or "info").lower()
    mapping = {
        "critical": "crit",
        "crit": "crit",
        "high": "high",
        "medium": "med",
        "med": "med",
        "low": "low",
        "info": "info",
    }
    return mapping.get(s, s)


def _evidence_full(finding: dict[str, Any]) -> str:
    """Richer evidence line for the register (columns + source + first caveat)."""
    bits: list[str] = []
    cols = [str(c) for c in (finding.get("affected_columns") or []) if c]
    if cols:
        shown = ", ".join(cols[:6])
        more = len(cols) - min(len(cols), 6)
        bits.append(f"{shown}{'…' if more else ''}")
    evidence = finding.get("evidence") or []
    if evidence and isinstance(evidence[0], dict):
        source = evidence[0].get("source") or evidence[0].get("key")
        if source:
            bits.append(str(source))
        limitations = evidence[0].get("limitations") or []
        if limitations:
            bits.append(str(limitations[0]))
    if not bits:
        return "report"
    return " · ".join(bits)


def _chunk(items: list[dict[str, str]], per: int = 14) -> list[dict[str, Any]]:
    """Split ledger items into scannable columns; large groups get more columns."""
    if not items:
        return [{"key": "c0", "items": []}]
    if len(items) <= per:
        return [{"key": "c0", "items": items}]
    # Prefer up to 3 columns for dense groups so the sheet scales.
    n_cols = 3 if len(items) > per * 2 else 2
    size = max(1, (len(items) + n_cols - 1) // n_cols)
    cols: list[dict[str, Any]] = []
    for index in range(n_cols):
        start = index * size
        part = items[start : start + size]
        if part:
            cols.append({"key": f"c{index}", "items": part})
    return cols or [{"key": "c0", "items": items}]


def _kv_items(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"k": k, "v": v} for k, v in pairs]


def build_findings_register(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Findings as register rows: Sev / Key / Detail / Evidence / Concept."""
    rows: list[dict[str, Any]] = []
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    findings = sorted(
        report.get("findings") or [],
        key=lambda item: rank.get(str(item.get("severity", "info")).lower(), 9),
    )
    for item in findings:
        key = str(item.get("key") or "")
        slug = FINDING_CONCEPT_SLUG.get(key)
        concept_key = resolve_concept_key(slug) if slug else None
        if concept_key is None and slug and slug in CONCEPT_NOTES:
            concept_key = slug
        sev = str(item.get("severity") or "info").lower()
        bucket = _sev_bucket(sev)
        caveats: list[str] = []
        for ev in item.get("evidence") or []:
            if isinstance(ev, dict):
                caveats.extend(str(c) for c in (ev.get("limitations") or [])[:2])
        rows.append(
            {
                "key": key,
                "anchor": f"f-{key.replace('.', '-')}",
                "severity": sev,
                "sev_label": _sev_label(sev),
                "is_blocking": bucket == "blocking",
                "is_med": bucket == "med",
                "is_low": bucket == "low",
                "is_info": bucket == "info",
                "title": item.get("title") or key,
                "detail": item.get("detail") or item.get("title") or "",
                "evidence": _evidence_full(item),
                "concept": concept_key or slug or "—",
                "concept_key": concept_key,
                "affected_columns": list(item.get("affected_columns") or []),
                "caveats": caveats[:3],
                "section": item.get("section") or key.split(".", 1)[0],
            }
        )
    return rows


def build_assumptions(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Section 02 — deep adaptive footnotes (means / matters / next / technical)."""
    notes = unique_notes(report.get("findings") or [])
    out: list[dict[str, Any]] = []
    for note in notes:
        slug = note.get("slug") or note.get("key") or ""
        concept_key = resolve_concept_key(str(slug)) or (
            str(slug) if str(slug) in CONCEPT_NOTES else None
        )
        out.append(
            {
                "n": note.get("n") or "",
                "slug": concept_key or slug,
                "concept_key": concept_key,
                "theme": note.get("theme") or "",
                "severity": note.get("severity") or "info",
                "means": note.get("means") or "",
                "matters": note.get("matters") or "",
                "next": note.get("next") or "",
                "technical": note.get("technical") or "",
                "evidence": note.get("evidence") or "",
                # Compact one-liner kept for older renderers / offline consumers.
                "text": note.get("means") or note.get("matters") or "",
            }
        )
    return out


def build_ledger(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Section 03 — every computed number, grouped like the redesign ledger."""
    groups = build_ledger_groups(report, report.get("findings") or [])
    adapt = build_adapt_context(report)
    skipped = adapt.get("skipped_details") or []
    out: list[dict[str, Any]] = []
    for group in groups:
        items = _kv_items([(str(k), str(v)) for k, v in (group.get("items") or [])])
        per = 16 if len(items) > 24 else 14
        out.append(
            {
                "key": group.get("key") or "group",
                "title": group.get("title") or "Group",
                "item_count": len(items),
                "cols": _chunk(items, per),
            }
        )
    if skipped:
        skip_items = _kv_items(
            [
                (
                    str(item).split(":", 1)[0],
                    str(item).split(":", 1)[-1].strip(),
                )
                for item in skipped
            ]
        )
        out.append(
            {
                "key": "skipped",
                "title": "Skipped / not applicable analyzers",
                "item_count": len(skip_items),
                "cols": _chunk(skip_items, 99),
            }
        )
    return out


def build_recommendation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Section 04 — recommended sequence table."""
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(report.get("recommendation_details") or [], start=1):
        action = item.get("action") or {}
        operation = action.get("operation") if isinstance(action, dict) else None
        call = f"session.{operation}(...)" if operation else (item.get("api") or "—")
        basis = item.get("based_on") or item.get("finding_key") or item.get("basis")
        if isinstance(basis, list):
            basis = ", ".join(str(b) for b in basis[:4])
        caveats = item.get("caveats") or []
        if isinstance(caveats, str):
            caveats = [caveats]
        rows.append(
            {
                "n": str(index),
                "when": str(item.get("priority") or "next").upper(),
                "title": item.get("title") or "",
                "call": call,
                "basis": str(basis or "—"),
                "rationale": item.get("rationale") or "",
                "caveats": [str(c) for c in caveats[:3]],
            }
        )
    return rows


def build_kpi_strip(report: dict[str, Any]) -> dict[str, Any]:
    """Header KPI strip: Readiness / Scope / Completeness / Runtime."""
    adapt = build_adapt_context(report)
    findings = report.get("findings") or []
    blocking = [
        item
        for item in findings
        if str(item.get("severity", "")).lower() in {"high", "critical", "crit"}
    ]
    med = [
        item for item in findings if str(item.get("severity", "")).lower() in {"medium", "med"}
    ]
    if blocking:
        readiness = "Blocked"
        readiness_note = f"{len(blocking)} blocking {plural(len(blocking), 'finding')}"
    elif med:
        readiness = "Caution"
        readiness_note = f"{len(med)} {plural(len(med), 'finding')} to resolve"
    else:
        readiness = "Clear"
        readiness_note = "no blocking findings raised"

    n_rows = adapt.get("n_rows")
    analysis_rows = adapt.get("analysis_rows") or n_rows
    n_cols = adapt.get("n_columns")
    missing_cells = adapt.get("missing_cells")
    engine = adapt.get("engine") or "pandas"
    mode = adapt.get("mode") or "eager"
    target_note = ""
    if adapt.get("target_column"):
        target_note = f" · {adapt['target_column']} ({adapt.get('task') or 'task undeclared'})"
    return {
        "readiness": readiness,
        "readiness_note": readiness_note,
        "scope": f"{_fmt_n(analysis_rows)} / {_fmt_n(n_rows)}",
        "scope_note": f"rows analysed · {_fmt_n(n_cols)} columns{target_note}",
        "completeness": _fmt_pct(adapt.get("completeness")),
        "completeness_note": f"{_fmt_n(missing_cells)} missing cells",
        "runtime": str(engine),
        "runtime_note": f"{mode} · BuildML {__version__}",
        "version": __version__,
        "session_label": str(adapt.get("session_label") or "session"),
        "engine": str(engine),
    }


def build_spine_meta(
    *,
    register_n: int,
    assumptions_n: int,
    ledger_n: int,
    ledger_items: int,
    sequence_n: int,
    domain_n: int,
    figure_n: int,
    methods_n: int,
    methods_ran: int,
    degraded_n: int,
    adapt: dict[str, Any],
) -> dict[str, str]:
    """Scannable counts for numbered-spine section heads."""
    skipped_n = len(adapt.get("skipped_analyzers") or [])
    return {
        "register": f"{register_n} {plural(register_n, 'finding')}",
        "assumptions": (
            f"{assumptions_n} {plural(assumptions_n, 'footnote')}"
            if assumptions_n
            else "none for this session"
        ),
        "ledger": (
            f"{ledger_n} {plural(ledger_n, 'group')} · {ledger_items} numbers"
            + (f" · {skipped_n} skipped" if skipped_n else "")
        ),
        "sequence": (
            f"{sequence_n} {plural(sequence_n, 'step')}" if sequence_n else "none produced"
        ),
        "domains": (
            f"{domain_n} board {plural(domain_n, 'brief')}" if domain_n else "no boards with data"
        ),
        "figures": (
            f"{figure_n} {plural(figure_n, 'figure')} with data"
            if figure_n
            else "no non-empty figures"
        ),
        "methods": f"{methods_ran} ran · {methods_n} families",
        "degraded": (
            f"{degraded_n} {plural(degraded_n, 'gap')}" if degraded_n else "none recorded"
        ),
    }


def build_cockpit_sheet(report: dict[str, Any]) -> dict[str, Any]:
    """Full cockpit sheet structures for the live App / offline bundle."""
    adapt = build_adapt_context(report)
    register = build_findings_register(report)
    assumptions = build_assumptions(report)
    ledger = build_ledger(report)
    sequence = build_recommendation_rows(report)
    domain_briefs = build_domain_briefs(report)
    methods = build_methods_catalog(report)
    degraded = build_degraded_rows(report)
    chart_ids = charts_for_cockpit_report(report)
    methods_ran = sum(1 for card in methods if card.get("status") == "ran")
    ledger_items = sum(int(group.get("item_count") or 0) for group in ledger)
    sheet = {
        "kpis": build_kpi_strip(report),
        "register": register,
        "assumptions": assumptions,
        "ledger": ledger,
        "sequence": sequence,
        "domain_briefs": domain_briefs,
        "methods": methods,
        "degraded": degraded,
        "chart_ids": chart_ids,
        "narrative": list(report.get("narrative") or []),
        "spine_meta": build_spine_meta(
            register_n=len(register),
            assumptions_n=len(assumptions),
            ledger_n=len(ledger),
            ledger_items=ledger_items,
            sequence_n=len(sequence),
            domain_n=len(domain_briefs),
            figure_n=len(chart_ids),
            methods_n=len(methods),
            methods_ran=methods_ran,
            degraded_n=len(degraded),
            adapt=adapt,
        ),
        "adapt": adapt,
        "session_sentence": adapt.get("session_sentence") or "",
        "what_to_change": what_to_change(report),
        "focus_columns": {
            "eligible": list_names(adapt.get("eligible_features") or [], 8),
            "constants": list_names(adapt.get("constant_columns") or [], 6),
            "id_like": list_names(adapt.get("id_like_columns") or [], 6),
            "target": adapt.get("target_column"),
        },
        "coverage": {
            "register": len(register),
            "assumptions": len(assumptions),
            "ledger_groups": len(ledger),
            "ledger_items": ledger_items,
            "sequence": len(sequence),
            "domain_briefs": len(domain_briefs),
            "figures": len(chart_ids),
            "methods": len(methods),
            "methods_ran": methods_ran,
            "degraded": len(degraded),
        },
    }
    return enrich_cockpit_sheet(sheet, report)
