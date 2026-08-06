# ruff: noqa: E501
"""BUILDML STATIC EDA: Industry readiness sheet HTML export.

Exports one offline HTML file that mirrors the Industry design-system spine:
KPI strip, findings register, assumption notes, ledger, recommended Session API
sequence, and figures. It deliberately omits Readiness Gates pages, Concept
Academy navigation, and recorded human gate-status UX.

Teaching content appears only as short inline assumption footnotes derived from
finding evidence, not as a searchable curriculum product.

See Also
--------
buildml.eda.assumption_notes : Structured footnotes for findings.
buildml.eda.cockpit_style : Inlined Industry tokens and shell behaviour.
buildml.reporting.html : Shared escape / asset helpers.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from buildml._version import __version__
from buildml.eda.assumption_notes import note_for_finding, unique_notes
from buildml.eda.cockpit_style import COCKPIT_CSS, COCKPIT_JS
from buildml.eda.sheet_coverage import (
    build_degraded_rows,
    build_ledger_groups,
    build_methods_catalog,
    fmt_int as _fmt_int,
    fmt_metric as _fmt,
    fmt_pct as _pct,
)
from buildml.reporting.html import encode_asset, escape, element_id

DEFAULT_MAX_HTML_BYTES = 12 * 1024 * 1024
DEFAULT_HEADING = "BUILDML STATIC EDA"
_LEGACY_DEFAULT_TITLES = frozenset(
    {
        "BuildML EDA Report",
        "Static EDA — readiness sheet",
        DEFAULT_HEADING,
    }
)

_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

_CALL_BY_REC_KEY = {
    "next.impute": 'session.impute(strategy="median")',
    "next.drop_constants": 'session.set_roles({"<constant>": "ignore"})',
    "next.exclude_ids": 'session.set_roles({"<id>": "id"})',
    "next.deduplicate": 'session.explain("ingest", moment="after")',
    "next.high_cardinality": 'session.explain("encode", moment="before")',
    "next.collinearity": 'session.explain("reduce_dimensions")',
    "next.correlated_pairs": 'session.explain("reduce_dimensions")',
    "next.outliers": 'session.handle_outliers(action="detect")',
    "next.drift": 'session.explain("split", moment="after")',
    "next.validate": 'session.split(test_size=0.2, stratify=True)',
}


def export_eda_html(
    report_dict: dict[str, Any],
    path: str | Path,
    *,
    title: str = DEFAULT_HEADING,
    figures: Mapping[str, Any] | None = None,
    include_raw_appendix: bool = True,
    max_figures: int = 36,
    max_html_bytes: int = DEFAULT_MAX_HTML_BYTES,
) -> Path:
    """Write the BUILDML STATIC EDA report as one self-contained HTML file.

    Assembles the Industry blueprint spine, inlines every figure, and
    writes a portable document. No CDN, no sidecar images, no server.

    Parameters
    ----------
    report_dict:
        The report as plain data, from
        :meth:`~buildml.eda.report.EDAReport.to_dict`. Missing sections are
        skipped rather than erroring, so a partial pass still exports.
    path:
        Where to write. Parent directories are created.
    title:
        Document title (also used in the page heading context).
    figures:
        Rendered figures to embed. Error entries are disclosed in the degraded
        section rather than skipped silently.
    include_raw_appendix:
        Append the full analyzer output as formatted JSON when size allows.
    max_figures:
        Ceiling on embedded matplotlib / path figures.
    max_html_bytes:
        Size budget, 12 MiB by default. The raw appendix is dropped first when
        the budget is exceeded.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If ``max_figures`` is negative, ``max_html_bytes`` is not positive, or
        the final document still exceeds the budget.
    OSError
        If the file cannot be written.
    """
    if max_figures < 0:
        raise ValueError("max_figures must be non-negative")
    if max_html_bytes < 1:
        raise ValueError("max_html_bytes must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    document = _render_cockpit(
        report_dict,
        title=title,
        figures=figures or {},
        include_raw_appendix=include_raw_appendix,
        max_figures=max_figures,
    )
    encoded_size = len(document.encode("utf-8"))
    if encoded_size > max_html_bytes and include_raw_appendix:
        document = _render_cockpit(
            report_dict,
            title=title,
            figures=figures or {},
            include_raw_appendix=False,
            max_figures=max_figures,
            appendix_omitted=(encoded_size, max_html_bytes),
        )
    final_size = len(document.encode("utf-8"))
    if final_size > max_html_bytes:
        raise ValueError(
            f"Report size {final_size:,} bytes exceeds max_html_bytes={max_html_bytes:,}; "
            "reduce sample_rows, max_columns, or max_figures."
        )
    destination.write_text(document, encoding="utf-8")
    return destination


def _render_cockpit(
    report: dict[str, Any],
    *,
    title: str,
    figures: Mapping[str, Any],
    include_raw_appendix: bool,
    max_figures: int,
    appendix_omitted: tuple[int, int] | None = None,
) -> str:
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    findings = list(report.get("findings") or [])
    recommendations = list(report.get("recommendation_details") or [])
    findings_sorted = sorted(
        findings,
        key=lambda item: (
            _SEV_RANK.get(str(item.get("severity", "info")).lower(), 9),
            str(item.get("key", "")),
        ),
    )

    readiness = _readiness(findings_sorted)
    scope = _scope_kpis(overview)
    completeness = _completeness_kpis(quality)
    runtime = _runtime_kpis(overview)
    assumption_rows = _assumption_rows(findings_sorted)
    ledger_groups = _ledger_groups(report, findings_sorted)
    figure_block, skipped_figures = _figure_assets(report, figures, max_figures=max_figures)
    chart_blocks = _industry_charts(report)
    warnings = [str(item) for item in (report.get("warnings") or [])]

    kicker = (
        f"BuildML {__version__} · Exploratory data analysis · "
        f"{escape(overview.get('mode') or 'session')}"
    )
    doc_title = DEFAULT_HEADING if (not title or title in _LEGACY_DEFAULT_TITLES) else title
    heading = escape(doc_title)

    spines = [
        _spine(
            "01",
            "findings-register",
            "Findings register",
            _findings_table(findings_sorted, assumption_rows),
            search_extra="findings severity evidence notes",
        ),
        _spine(
            "02",
            "assumptions",
            "What each finding assumes",
            _assumptions_block(assumption_rows),
            search_extra="assumptions themes means matters technical evidence",
        ),
        _spine(
            "03",
            "ledger",
            "Ledger — every computed number",
            _ledger_block(ledger_groups),
            search_extra="ledger metrics frame quality vif pca drift",
        ),
        _spine(
            "04",
            "recommended-sequence",
            "Recommended sequence",
            _recommendations_table(recommendations, report),
            search_extra="recommendations session calls priority",
        ),
        _spine(
            "05",
            "figures",
            "Figures",
            chart_blocks + figure_block + _skipped_figures_details(skipped_figures),
            search_extra="figures charts plots gallery",
        ),
        _spine(
            "06",
            "methods",
            "Methods and limitations",
            _methods_body(report),
            search_extra="methods limitations analyzers caveats ran skipped",
        ),
        _spine(
            "07",
            "degraded",
            "Skipped and degraded analyses",
            _degraded_body(report, figures, skipped_figures),
            search_extra="skipped degraded unavailable",
        ),
    ]
    if appendix_omitted is not None:
        original, budget = appendix_omitted
        spines.append(
            _spine(
                "08",
                "appendix",
                "Raw appendix omitted by output budget",
                (
                    f'<p class="om-note">The raw appendix was omitted because the first render was '
                    f"{escape(f'{original:,}')} bytes against a {escape(f'{budget:,}')}-byte output budget. "
                    "Use EDAReport.to_dict() or raise max_html_bytes for an intentional larger export.</p>"
                ),
            )
        )
    elif include_raw_appendix:
        spines.append(
            _spine(
                "08",
                "appendix",
                "Raw technical appendix",
                (
                    "<details><summary>Open raw technical payload</summary>"
                    f"{_json_block(report)}</details>"
                ),
            )
        )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '  <meta name="generator" content="BuildML">\n'
        f"  <title>{escape(doc_title)}</title>\n"
        f"  <style>{COCKPIT_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '  <a class="bml-skip-link" href="#main-content">Skip to report content</a>\n'
        '  <div class="om-shell">\n'
        f'    <header class="om-header" role="banner">{_header_inner(kicker, heading)}</header>\n'
        f"    {_kpi_strip(readiness, scope, completeness, runtime)}\n"
        f"    {_disclosure(warnings)}\n"
        f"    {_tools_row()}\n"
        '    <main id="main-content" tabindex="-1">\n'
        + "".join(spines)
        + "\n    </main>\n"
        '    <footer class="om-footer" role="contentinfo">'
        "<p>Generated locally by BuildML. This file does not load network assets. "
        "Recommendations name Session operations; they do not execute them.</p></footer>\n"
        f"    <script>{COCKPIT_JS}</script>\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def _header_inner(kicker: str, heading: str) -> str:
    return (
        '<div class="om-header__title">'
        f'<div class="om-kicker">{kicker}</div>'
        f"<h1>{heading}</h1>"
        "</div>"
        '<div class="om-tools" aria-label="Export actions">'
        '<button type="button" class="btn btn-primary blueprint" id="bml-offline-html" '
        'data-filename="buildml-static-eda.html" '
        'aria-label="Download Offline HTML snapshot" '
        'title="Download this Static EDA file (already offline)">'
        '<i class="corner tl"></i><i class="corner tr"></i>'
        '<i class="corner bl"></i><i class="corner br"></i>'
        "Offline HTML</button>"
        "</div>"
    )


def _tools_row() -> str:
    return (
        '<div class="om-tools-row" role="search" aria-label="Search report sections">'
        '<div class="om-search-field">'
        '<label for="bml-section-search">Search report</label>'
        '<input id="bml-section-search" class="bml-section-search" type="search" '
        'placeholder="Type to filter sections by title or keywords" autocomplete="off" '
        'aria-describedby="bml-section-search-hint" aria-controls="main-content">'
        '<p id="bml-section-search-hint" class="bml-search-hint">'
        "Filters whole sections. Escape clears. Nested panels keep their own filters."
        "</p>"
        "</div>"
        '<p id="bml-section-search-status" class="bml-search-status" aria-live="polite"></p>'
        "</div>"
    )


def _kpi_strip(
    readiness: dict[str, str],
    scope: dict[str, str],
    completeness: dict[str, str],
    runtime: dict[str, str],
) -> str:
    cells = (
        ("Readiness", readiness["value"], readiness["note"], False),
        ("Scope", scope["value"], scope["note"], True),
        ("Completeness", completeness["value"], completeness["note"], True),
        ("Runtime", runtime["value"], runtime["note"], True),
    )
    parts = [
        '<i class="corner tl"></i><i class="corner tr"></i>',
        '<i class="corner bl"></i><i class="corner br"></i>',
    ]
    for label, value, note, mono in cells:
        value_class = "om-kpi__value om-mono" if mono else "om-kpi__value"
        parts.append(
            '<div class="om-kpi__cell">'
            f'<div class="om-kpi__label">{escape(label)}</div>'
            f'<div class="{value_class}">{escape(value)}</div>'
            f'<div class="om-kpi__note">{escape(note)}</div>'
            "</div>"
        )
    return f'<section class="blueprint om-kpi" aria-label="Readiness summary">{"".join(parts)}</section>'


def _disclosure(warnings: Sequence[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<div>{escape(item)}</div>" for item in warnings)
    return f'<div class="om-disclosure" aria-label="Analysis disclosures">{items}</div>'


def _spine(
    number: str,
    section_id: str,
    title: str,
    body: str,
    *,
    search_extra: str = "",
) -> str:
    search = " ".join(part for part in (title, search_extra) if part).strip()
    return (
        f'<div class="om-spine" data-spine-section data-search="{escape(search)}">'
        f'<div class="om-spine__n om-mono">{escape(number)}</div>'
        f'<section id="{escape(section_id)}">'
        f'<h4 class="om-section-title">{escape(title)}</h4>'
        f"{body}"
        "</section></div>"
    )


def _readiness(findings: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for finding in findings:
        sev = str(finding.get("severity", "info")).lower()
        counts[sev] = counts.get(sev, 0) + 1
    blocking = counts["critical"] + counts["high"]
    caution = counts["medium"]
    if blocking:
        label = "Blocked"
        note = f"{blocking} blocking finding" + ("s" if blocking != 1 else "")
    elif caution:
        label = "Caution"
        note = f"{caution} medium finding" + ("s" if caution != 1 else "")
    else:
        label = "Clear"
        note = "No blocking or medium findings"
    return {"value": label, "note": note}


def _scope_kpis(overview: Mapping[str, Any]) -> dict[str, str]:
    rows = overview.get("analysis_rows", overview.get("n_rows"))
    total = overview.get("n_rows")
    columns = overview.get("n_columns")
    value = f"{_fmt_int(rows)} / {_fmt_int(total)}"
    note = f"rows analysed · {_fmt_int(columns)} columns"
    return {"value": value, "note": note}


def _completeness_kpis(quality: Mapping[str, Any]) -> dict[str, str]:
    score = quality.get("completeness_score")
    missing = int(quality.get("missing_cell_count") or 0)
    if isinstance(score, (int, float)):
        value = f"{float(score) * 100:.3f}%"
    else:
        value = "not available"
    note = f"{missing:,} missing cells"
    return {"value": value, "note": note}


def _runtime_kpis(overview: Mapping[str, Any]) -> dict[str, str]:
    engine = str(overview.get("engine") or overview.get("mode") or "pandas")
    native = overview.get("has_native")
    lazy = overview.get("has_lazy_native", overview.get("lazy"))
    bits = []
    if native is not None:
        bits.append(f"has_native = {str(bool(native)).lower()}")
    if lazy is not None:
        bits.append(f"lazy = {str(bool(lazy)).lower()}")
    if not bits:
        memory = overview.get("memory_bytes_approx")
        if memory is not None:
            bits.append(f"approx memory = {_fmt(memory)} B")
        else:
            bits.append(f"mode = {overview.get('mode', 'session')}")
    return {"value": engine, "note": " · ".join(bits)}


def _sev_tag(severity: object) -> str:
    sev = str(severity or "info").lower()
    label = {"critical": "crit", "high": "high", "medium": "med", "low": "low", "info": "info"}.get(
        sev, sev
    )
    if sev in {"critical", "high"}:
        return f'<span class="tag tag-blocking">{escape(label)}</span>'
    if sev == "medium":
        return f'<span class="tag tag-accent">{escape(label)}</span>'
    if sev == "low":
        return f'<span class="tag tag-outline">{escape(label)}</span>'
    return f'<span class="tag tag-neutral">{escape(label)}</span>'


def _assumption_rows(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return unique_notes(findings)


def _findings_table(
    findings: Sequence[Mapping[str, Any]],
    assumption_rows: Sequence[Mapping[str, str]],
) -> str:
    slug_note = {row["slug"]: row["n"] for row in assumption_rows}
    key_note = {}
    for finding in findings:
        key = str(finding.get("key", ""))
        note = note_for_finding(finding)
        key_note[key] = slug_note.get(note["slug"], "")

    if not findings:
        return '<p class="bml-empty">No structured findings were produced for this pass.</p>'

    rows_html = []
    for finding in findings:
        key = str(finding.get("key", "finding"))
        evidence = finding.get("evidence") or []
        cols = [str(c) for c in (finding.get("affected_columns") or []) if c]
        evidence_bits: list[str] = []
        if evidence:
            first = evidence[0]
            source = str(first.get("source") or first.get("key") or "evidence")
            evidence_bits.append(source)
        elif cols:
            evidence_bits.append("affected_columns")
        chips = ""
        if cols:
            shown = cols[:8]
            chip_html = "".join(
                f'<span class="bml-col-chip om-mono">{escape(c)}</span>' for c in shown
            )
            if len(cols) > 8:
                chip_html += (
                    f'<span class="bml-col-chip om-mono text-muted">+{len(cols) - 8}</span>'
                )
            chips = f'<div class="bml-col-chips">{chip_html}</div>'
        evidence_label = " · ".join(evidence_bits) if evidence_bits else ""
        note = key_note.get(key, "")
        anchor = element_id(key, prefix="finding")
        rows_html.append(
            f'<tr id="{escape(anchor)}">'
            f'<td class="bml-col-sev">{_sev_tag(finding.get("severity"))}</td>'
            f'<td class="om-mono bml-cell-wrap" style="font-size:12px">{escape(key)}</td>'
            f'<td class="bml-cell-wrap bml-cell-wrap--prose">'
            f"{escape(finding.get('detail') or finding.get('title'))}</td>"
            f'<td class="bml-cell-wrap bml-cell-wrap--evidence">'
            f"{chips}"
            f'<div class="om-mono text-muted" style="font-size:11px">'
            f"{escape(evidence_label)}</div></td>"
            f'<td class="om-mono bml-cell-wrap" style="font-size:11px">{escape(note)}</td>'
            "</tr>"
        )
    table = (
        '<div class="bml-table-wrap">'
        '<div class="bml-table-tools">'
        '<label class="text-muted">Filter rows '
        '<input class="bml-table-search" type="search" placeholder="Filter rows" autocomplete="off">'
        "</label>"
        '<label class="text-muted">Sort by '
        '<select class="bml-table-sort">'
        '<option value="">Original order</option>'
        '<option value="0">Severity</option>'
        '<option value="1">Key</option>'
        "</select></label>"
        "</div>"
        '<table class="table bml-data-table bml-table--fit bml-table--register">'
        "<thead><tr>"
        '<th scope="col" class="bml-col-sev">Sev</th>'
        '<th scope="col" class="bml-col-key">Key</th>'
        '<th scope="col" class="bml-col-detail">Detail</th>'
        '<th scope="col" class="bml-col-evidence">Evidence</th>'
        '<th scope="col" class="bml-col-note">Note</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    return table


def _assumptions_block(assumption_rows: Sequence[Mapping[str, str]]) -> str:
    if not assumption_rows:
        return '<p class="bml-empty">No assumption notes were attached to findings.</p>'

    theme_order: list[str] = []
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in assumption_rows:
        theme = str(row.get("theme") or "Other")
        if theme not in grouped:
            theme_order.append(theme)
        grouped[theme].append(row)

    jump = "".join(
        f'<a class="om-jump__chip" href="#assumption-theme-{escape(element_id(theme))}">'
        f"{escape(theme)} · {len(grouped[theme])}</a>"
        for theme in theme_order
    )
    open_first = len(assumption_rows) <= 8
    groups_html: list[str] = []
    for theme_index, theme in enumerate(theme_order):
        rows = grouped[theme]
        cards = "".join(_assumption_card(row, expand=open_first and theme_index == 0) for row in rows)
        open_attr = " open" if (open_first or theme_index == 0) else ""
        theme_id = element_id(theme)
        groups_html.append(
            f'<details class="om-group" data-assumption-group data-theme="{escape(theme)}"{open_attr}>'
            f'<summary class="om-group__summary" id="assumption-theme-{escape(theme_id)}">'
            f'<span class="om-group__title">{escape(theme)}</span>'
            f'<span class="om-group__meta om-mono">{len(rows)} note{"s" if len(rows) != 1 else ""}</span>'
            f"</summary>"
            f'<div class="om-assumptions">{cards}</div>'
            f"</details>"
        )

    return (
        '<div class="om-panel" data-panel="assumptions">'
        '<div class="om-panel__tools">'
        '<label class="om-panel__filter">Filter notes '
        '<input class="om-panel-search" type="search" data-filter-target="assumptions" '
        'placeholder="Filter by theme, slug, or wording" autocomplete="off">'
        "</label>"
        f'<p class="om-panel__count om-mono" data-filter-count="assumptions">'
        f"{len(assumption_rows)} notes · {len(theme_order)} themes</p>"
        f'<nav class="om-jump" aria-label="Jump to assumption theme">{jump}</nav>'
        "</div>"
        f'<div class="om-assumption-groups">{"".join(groups_html)}</div>'
        "</div>"
    )


def _assumption_card(row: Mapping[str, str], *, expand: bool = False) -> str:
    evidence = row.get("evidence") or ""
    means = str(row.get("means") or "")
    summary_line = means if len(means) <= 140 else means[:137].rstrip() + "…"
    open_attr = " open" if expand else ""
    search = " ".join(
        str(row.get(key) or "")
        for key in ("theme", "slug", "key", "means", "matters", "next", "technical", "evidence", "severity")
    )
    return (
        f'<details class="om-assumption-card" data-assumption-card '
        f'data-search="{escape(search)}"{open_attr}>'
        f'<summary class="om-assumption-card__summary">'
        f'<span class="om-assumption-card__top">'
        f'<span class="om-assumption-card__id om-mono">'
        f"{escape(row.get('n', ''))} · {escape(row.get('slug', ''))}</span>"
        f"{_sev_tag(row.get('severity'))}"
        "</span>"
        f'<span class="om-assumption-card__blurb">{escape(summary_line)}</span>'
        f'<span class="om-mono text-muted om-assumption-card__key">{escape(row.get("key", ""))}</span>'
        "</summary>"
        '<div class="om-assumption-card__body">'
        f'<p><span class="om-assumption__label">What this means</span> {escape(means)}</p>'
        f'<p><span class="om-assumption__label">Why it matters</span> {escape(row.get("matters", ""))}</p>'
        f'<p><span class="om-assumption__label">What to check next</span> {escape(row.get("next", ""))}</p>'
        f'<p class="om-assumption__tech"><span class="om-assumption__label">Technical note</span> '
        f"{escape(row.get('technical', ''))}</p>"
        + (
            f'<p class="om-assumption__evidence"><span class="om-assumption__label">'
            f"Evidence in this report</span> {escape(evidence)}</p>"
            if evidence
            else ""
        )
        + "</div></details>"
    )


def _ledger_groups(
    report: dict[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Delegate to shared coverage so Static EDA matches the App cockpit ledger."""
    return build_ledger_groups(report, findings)


def _ledger_block(groups: Sequence[Mapping[str, Any]]) -> str:
    if not groups:
        return '<p class="bml-empty">No ledger metrics were recorded for this pass.</p>'

    total_items = sum(len(group.get("items") or []) for group in groups)
    jump = "".join(
        f'<a class="om-jump__chip" href="#ledger-group-{escape(element_id(str(group["title"])))}">'
        f'{escape(group["title"])} · {len(group.get("items") or [])}</a>'
        for group in groups
        if group.get("items")
    )
    open_budget = 3
    blocks: list[str] = []
    for index, group in enumerate(groups):
        items = list(group.get("items") or [])
        if not items:
            continue
        rows = "".join(
            f'<div class="om-led" data-ledger-row data-search="{escape(f"{key} {value}")}" '
            f'title="{escape(f"{key} — {value}")}">'
            f'<span class="om-led__key om-mono" title="{escape(str(key))}">{escape(key)}</span>'
            f'<span class="om-led__val om-mono" title="{escape(str(value))}">{escape(value)}</span></div>'
            for key, value in items
        )
        open_attr = " open" if index < open_budget else ""
        group_id = element_id(str(group["title"]))
        blocks.append(
            f'<details class="om-ledger-group" data-ledger-group '
            f'data-search="{escape(group["title"])}"{open_attr}>'
            f'<summary class="om-group__summary" id="ledger-group-{escape(group_id)}">'
            f'<span class="om-group__title">{escape(group["title"])}</span>'
            f'<span class="om-group__meta om-mono">{len(items)} metric{"s" if len(items) != 1 else ""}</span>'
            f"</summary>"
            f'<div class="om-ledger__rows">{rows}</div>'
            f"</details>"
        )

    return (
        '<div class="om-panel" data-panel="ledger">'
        '<div class="om-panel__tools">'
        '<label class="om-panel__filter">Filter ledger '
        '<input class="om-panel-search" type="search" data-filter-target="ledger" '
        'placeholder="Filter metrics by name or value" autocomplete="off">'
        "</label>"
        f'<p class="om-panel__count om-mono" data-filter-count="ledger">'
        f"{total_items} metrics · {len(blocks)} groups</p>"
        f'<nav class="om-jump" aria-label="Jump to ledger group">{jump}</nav>'
        "</div>"
        f'<div class="om-ledger">{"".join(blocks)}</div>'
        "</div>"
    )


def _session_call(recommendation: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    key = str(recommendation.get("key", ""))
    action = recommendation.get("action") or {}
    quality = report.get("quality") or {}
    outliers = report.get("outliers") or {}
    if key == "next.drop_constants":
        cols = list(quality.get("constant_columns") or [])[:2]
        if cols:
            mapping = ", ".join(f'"{col}": "ignore"' for col in cols)
            return f"session.set_roles({{{mapping}}})"
    if key == "next.exclude_ids":
        cols = list(quality.get("id_like_columns") or [])[:2]
        if cols:
            mapping = ", ".join(f'"{col}": "id"' for col in cols)
            return f"session.set_roles({{{mapping}}})"
    if key == "next.high_cardinality":
        cols = list(quality.get("high_cardinality_columns") or [])[:1]
        if cols:
            return (
                f'session.explain("encode", moment="before")  # review {cols[0]!s}'
            )
    if key == "next.outliers":
        hot = [
            column
            for column, stats in (outliers.get("per_column") or {}).items()
            if float(stats.get("iqr_outlier_rate") or 0) >= 0.05
        ][:1]
        if hot:
            return f'session.handle_outliers(action="detect", columns={json.dumps(hot)})'
    if isinstance(action, Mapping) and action.get("operation"):
        op = action["operation"]
        params = action.get("parameters") or {}
        if params:
            args = ", ".join(f"{name}={json.dumps(value)}" for name, value in params.items())
            return f"session.{op}({args})"
        return f"session.{op}()"
    return _CALL_BY_REC_KEY.get(key, f"session.explain({json.dumps(key)})")


def _recommendations_table(
    recommendations: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> str:
    if not recommendations:
        return '<p class="bml-empty">No recommendations were generated.</p>'
    rows = []
    for index, recommendation in enumerate(recommendations, start=1):
        based = recommendation.get("based_on") or []
        basis = ", ".join(str(item) for item in based) if based else "—"
        priority = str(recommendation.get("priority") or "next")
        if hasattr(priority, "value"):
            priority = str(priority)
        rows.append(
            "<tr>"
            f'<td class="om-mono" style="font-size:12px">{index}</td>'
            f'<td class="om-mono bml-cell-wrap" style="font-size:11px">{escape(priority)}</td>'
            f'<td class="bml-cell-wrap bml-cell-wrap--prose">'
            f"{escape(recommendation.get('title'))}</td>"
            f'<td class="om-mono bml-cell-wrap" style="font-size:11px">'
            f"{escape(_session_call(recommendation, report))}</td>"
            f'<td class="om-mono bml-cell-wrap" style="font-size:11px">{escape(basis)}</td>'
            "</tr>"
        )
    return (
        '<div class="bml-table-wrap">'
        '<table class="table bml-table--fit">'
        "<thead><tr>"
        '<th scope="col" style="width:34px">#</th>'
        '<th scope="col" style="width:12%">Priority</th>'
        '<th scope="col" style="width:36%">Action</th>'
        '<th scope="col" style="width:28%">Call</th>'
        '<th scope="col" style="width:16%">Based on</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="om-note">Recommendations name Session operations; they do not execute them. '
        "Full-dataset descriptive EDA after a split summarises observed rows and is not "
        "train-fitted transform evidence.</p>"
    )


def _industry_charts(report: Mapping[str, Any]) -> str:
    """Build measurement charts only when the underlying analyzer tables exist."""
    quality = report.get("quality") or {}
    bivariate = report.get("bivariate") or {}
    multivariate = report.get("multivariate") or {}
    target = report.get("target") or {}
    outliers = report.get("outliers") or {}
    drift = report.get("drift") or {}
    missing_rates = quality.get("missing_rate_by_column") or {}
    parts: list[str] = []
    fig_n = 0

    def _next_caption(body: str) -> str:
        nonlocal fig_n
        caption = f"Fig. 5.{fig_n} — {body}"
        fig_n += 1
        return caption

    if missing_rates:
        items = sorted(
            ((str(col), float(rate)) for col, rate in missing_rates.items() if float(rate or 0) > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )[:16]
        if items:
            parts.append(
                _bar_figure(
                    _next_caption("missing rate by column"),
                    items,
                    display=lambda value: _pct(value),
                    note="Observed missingness rates on the analysed frame.",
                )
            )

    mi = bivariate.get("mutual_information_vs_target") or {}
    if mi:
        items = sorted(
            ((str(name), float(value)) for name, value in mi.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )[:16]
        target_name = target.get("column") or "target"
        parts.append(
            _bar_figure(
                _next_caption(f"mutual information vs {target_name}"),
                items,
                display=lambda value: _fmt(value),
                note="MI does not establish direction or causality.",
            )
        )

    vif_rows = multivariate.get("vif") or []
    if vif_rows:
        items = [
            (str(row.get("column") or row.get("feature")), float(row.get("vif") or 0))
            for row in vif_rows
        ]
        items.sort(key=lambda pair: pair[1], reverse=True)
        parts.append(
            _bar_figure(
                _next_caption("variance inflation, eligible numeric features"),
                items[:16],
                display=lambda value: f"{value:.3f}",
                threshold=5.0,
                note="VIF is sensitive to the included feature set.",
            )
        )

    summary = target.get("summary") or {}
    class_counts = summary.get("class_counts") or {}
    if class_counts and target.get("column"):
        items = sorted(
            ((str(label), float(count)) for label, count in class_counts.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )[:16]
        total = sum(value for _, value in items) or 1.0
        parts.append(
            _bar_figure(
                _next_caption(f"class balance of {target.get('column')}"),
                items,
                display=lambda value: f"{_fmt_int(value)} · {_pct(value / total)}",
                note=(
                    f"A constant majority-class predictor scores "
                    f"{_pct(max(value for _, value in items) / total)} accuracy on this frame."
                ),
            )
        )
    elif summary.get("type") == "regression_target" and target.get("column"):
        # Continuous targets do not have class bars; surface shape stats as a compact board.
        stat_items = [
            (key, float(summary[key]))
            for key in ("mean", "std", "skew")
            if isinstance(summary.get(key), (int, float))
        ]
        if stat_items:
            parts.append(
                _bar_figure(
                    _next_caption(f"distribution summary of {target.get('column')}"),
                    [(key, abs(value)) for key, value in stat_items],
                    display=lambda value: _fmt(value),
                    note="Continuous target — class balance does not apply. Magnitudes are |stat| for display.",
                )
            )

    pairs = bivariate.get("top_abs_pearson_pairs") or []
    strong = [
        pair
        for pair in pairs
        if abs(float(pair.get("corr") or 0)) >= 0.5
        and str(pair.get("a")) != str(pair.get("b"))
    ][:16]
    if strong:
        items = [
            (f"{pair.get('a')}↔{pair.get('b')}", abs(float(pair.get("corr") or 0)))
            for pair in strong
        ]
        parts.append(
            _bar_figure(
                _next_caption("top |Pearson| feature pairs"),
                items,
                display=lambda value: f"{value:.3f}",
                threshold=0.8,
                note="Absolute Pearson on eligible numeric pairs; association is not causation.",
            )
        )

    outlier_cols = outliers.get("per_column") or {}
    if outlier_cols:
        items = sorted(
            (
                (str(column), float(stats.get("iqr_outlier_rate") or 0))
                for column, stats in outlier_cols.items()
                if float(stats.get("iqr_outlier_rate") or 0) > 0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:16]
        if items:
            parts.append(
                _bar_figure(
                    _next_caption("IQR outlier rate by column"),
                    items,
                    display=lambda value: _pct(value),
                    threshold=0.05,
                    note="Screening rates from IQR fences; skewed columns naturally flag heavy tails.",
                )
            )

    drift_rows = [
        *list(drift.get("numeric_drift") or []),
        *list(drift.get("categorical_drift") or []),
    ]
    if drift.get("available") and drift_rows:
        items = []
        for row in drift_rows:
            score = row.get("ks_stat", row.get("js_divergence"))
            if score is None:
                continue
            items.append((str(row.get("column")), float(score)))
        items.sort(key=lambda pair: pair[1], reverse=True)
        if items:
            flagged = len(drift.get("flagged_columns") or [])
            parts.append(
                _bar_figure(
                    _next_caption("train/test drift scores"),
                    items[:16],
                    display=lambda value: _fmt(value),
                    note=(
                        f"{flagged} column(s) met configured drift thresholds. "
                        "Check split construction before changing data."
                    ),
                )
            )

    variance = (multivariate.get("pca") or {}).get("explained_variance_ratio") or []
    if variance:
        items = [(f"PC{index}", float(value)) for index, value in enumerate(variance, start=1)]
        parts.append(
            _bar_figure(
                _next_caption("PCA explained variance ratio"),
                items,
                display=lambda value: _pct(value),
                note="Diagnostic PCA on complete-case standardized numerics; not a fitted Session transform.",
            )
        )

    multi = outliers.get("multivariate") or {}
    if multi:
        flagged = multi.get("anomaly_count", multi.get("flagged_row_count"))
        scored = multi.get("n_rows_scored", multi.get("scored_row_count"))
        rate = multi.get("anomaly_rate")
        if flagged is not None and scored:
            inlier = max(float(scored) - float(flagged), 0.0)
            parts.append(
                _bar_figure(
                    _next_caption("multivariate anomaly screen counts"),
                    [("inlier", inlier), ("flagged", float(flagged))],
                    display=lambda value: _fmt_int(value),
                    note=(
                        f"Rate={_pct(rate) if isinstance(rate, (int, float)) else 'n/a'}. "
                        "Screening signals, not confirmed errors."
                    ),
                )
            )

    if not parts:
        return (
            '<p class="bml-empty">No analyzer measurement charts were available for this frame. '
            "Tables above still list whatever sections were computed; "
            "pass include_plots=True when Matplotlib figures are needed.</p>"
        )
    return "".join(parts)


def _bar_figure(
    caption: str,
    items: Sequence[tuple[str, float]],
    *,
    display,
    note: str,
    threshold: float | None = None,
) -> str:
    if not items:
        return ""
    peak = max(value for _, value in items) or 1.0
    if threshold is not None:
        peak = max(peak, threshold)
    rows = []
    for name, value in items:
        width = max(0.4, (value / peak) * 100)
        hot = threshold is not None and value >= threshold
        cls = "om-bar-fill is-hot" if hot else "om-bar-fill"
        label = str(name)
        value_text = str(display(value))
        # Single-line value lane: strip internal newlines so a bad formatter
        # cannot break 1:1 bar alignment.
        value_text = " ".join(value_text.split())
        rows.append(
            '<div class="om-bar-row">'
            f'<div class="om-mono om-cap" title="{escape(label)}">{escape(label)}</div>'
            f'<div class="om-bar-track"><div class="{cls}" style="width:{width:.2f}%"></div></div>'
            f'<div class="om-bar-val" title="{escape(value_text)}">{escape(value_text)}</div>'
            "</div>"
        )
    return (
        f'<figure class="blueprint om-figure">'
        f'<i class="corner tl"></i><i class="corner tr"></i>'
        f'<i class="corner bl"></i><i class="corner br"></i>'
        f"<figcaption>{escape(caption)}</figcaption>"
        f"{''.join(rows)}"
        f'<p class="text-muted" style="margin:var(--space-3) 0 0;font-size:11px">{escape(note)}</p>'
        "</figure>"
    )


def _figure_assets(
    report: Mapping[str, Any],
    figures: Mapping[str, Any],
    *,
    max_figures: int,
) -> tuple[str, list[str]]:
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

    if not assets:
        gallery = (
            '<p class="om-note">No matplotlib figures were embedded. '
            "Measurement charts above are built from analyzer tables when present; "
            "pass include_plots=True for rendered plot images.</p>"
        )
        return gallery, skipped

    cards = []
    for index, (name, uri) in enumerate(assets, start=1):
        cards.append(
            f'<figure class="blueprint om-figure">'
            f'<i class="corner tl"></i><i class="corner tr"></i>'
            f'<i class="corner bl"></i><i class="corner br"></i>'
            f"<figcaption>Fig. M.{index} — {escape(name)}</figcaption>"
            f'<img src="{uri}" alt="{escape(name)}">'
            "</figure>"
        )
    return f'<div class="om-gallery">{"".join(cards)}</div>', skipped


def _skipped_figures_details(skipped: Sequence[str]) -> str:
    if not skipped:
        return ""
    items = "".join(f"<li>{escape(item)}</li>" for item in skipped)
    return f"<details><summary>Skipped plots</summary><ul>{items}</ul></details>"


def _methods_body(report: Mapping[str, Any]) -> str:
    cards = _methods_catalog(report)
    status_counts = {"ran": 0, "skipped": 0, "not_applicable": 0}
    for card in cards:
        status_counts[str(card["status"])] = status_counts.get(str(card["status"]), 0) + 1

    grid = "".join(_method_card(card) for card in cards)
    return (
        '<div class="om-methods" data-panel="methods">'
        '<div class="om-methods__legend" aria-label="Analyzer status counts">'
        f'<span class="tag tag-accent">Ran · {status_counts.get("ran", 0)}</span>'
        f'<span class="tag tag-outline">Skipped · {status_counts.get("skipped", 0)}</span>'
        f'<span class="tag tag-neutral">Not applicable · {status_counts.get("not_applicable", 0)}</span>'
        "</div>"
        f'<div class="om-methods__grid">{grid}</div>'
        '<aside class="om-callout" role="note">'
        '<div class="om-callout__label">Caveats</div>'
        "<ul>"
        "<li>Associations in this report describe co-occurrence; they do not establish causality.</li>"
        "<li>Library versions and low-level estimator defaults are not recorded in this export.</li>"
        "<li>Empty analyzer sections are omitted from figures and ledger groups rather than filled with placeholders.</li>"
        "</ul>"
        "</aside>"
        "</div>"
    )


def _methods_catalog(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """Delegate to shared coverage so Static EDA matches the App cockpit."""
    return build_methods_catalog(report)


def _method_card(card: Mapping[str, str]) -> str:
    status = str(card.get("status") or "skipped")
    label = {
        "ran": "Ran",
        "skipped": "Skipped",
        "not_applicable": "Not applicable",
    }.get(status, status.replace("_", " ").title())
    tag_class = {
        "ran": "tag-accent",
        "skipped": "tag-outline",
        "not_applicable": "tag-neutral",
    }.get(status, "tag-neutral")
    why = str(card.get("why") or "")
    why_html = (
        f'<p class="om-method__why"><span class="om-assumption__label">Why</span> {escape(why)}</p>'
        if why
        else ""
    )
    search = " ".join(
        str(card.get(key) or "")
        for key in ("family", "status", "summary", "detail", "why", label)
    )
    return (
        f'<article class="blueprint om-method" data-method-card data-status="{escape(status)}" '
        f'data-search="{escape(search)}">'
        '<i class="corner tl"></i><i class="corner tr"></i>'
        '<i class="corner bl"></i><i class="corner br"></i>'
        '<header class="om-method__head">'
        f'<h5 class="om-method__family">{escape(card.get("family", ""))}</h5>'
        f'<span class="tag {tag_class}">{escape(label)}</span>'
        "</header>"
        f'<p class="om-method__summary">{escape(card.get("summary", ""))}</p>'
        f"{why_html}"
        '<details class="om-method__detail">'
        "<summary>Technical detail</summary>"
        f'<p class="om-mono">{escape(card.get("detail", ""))}</p>'
        "</details>"
        "</article>"
    )


def _degraded_body(
    report: Mapping[str, Any],
    figures: Mapping[str, Any],
    skipped_figures: Sequence[str],
) -> str:
    rows: list[tuple[str, str]] = [
        (row["analysis"], row["reason"]) for row in build_degraded_rows(report)
    ]
    for name, value in figures.items():
        if isinstance(value, Mapping):
            rows.append((f"plot: {name}", str(value.get("error", "not rendered"))))
    for item in skipped_figures:
        if ": " in item:
            name, reason = item.split(": ", 1)
            rows.append((f"plot: {name}", reason))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in rows:
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    if not unique:
        return '<p class="bml-empty">No degraded or skipped analyses were recorded.</p>'
    table_rows = "".join(
        "<tr>"
        f'<td class="om-mono" style="font-size:12px">{escape(analysis)}</td>'
        f"<td>{escape(reason)}</td>"
        "</tr>"
        for analysis, reason in unique
    )
    return (
        '<table class="table" style="margin-top:var(--space-2)">'
        '<thead><tr><th scope="col">Analysis</th><th scope="col">Reason</th></tr></thead>'
        f"<tbody>{table_rows}</tbody></table>"
    )


def _json_block(value: Any) -> str:
    return f'<pre class="bml-json">{escape(json.dumps(value, indent=2, default=str))}</pre>'
