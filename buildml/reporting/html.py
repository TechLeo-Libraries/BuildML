# ruff: noqa: E501
"""Dependency-free HTML components and an offline BuildML report shell."""

from __future__ import annotations

import base64
import html
import mimetypes
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"[^a-z0-9_-]+")
DEFAULT_MAX_TABLE_ROWS = 500
DEFAULT_MAX_TABLE_COLUMNS = 50


def escape(value: object, *, quote: bool = True) -> str:
    """Escape untrusted text for HTML text or attribute contexts."""
    return html.escape("" if value is None else str(value), quote=quote)


def element_id(value: object, *, prefix: str = "section") -> str:
    """Return a stable, conservative HTML id."""
    identifier = _ID_PATTERN.sub("-", str(value).strip().lower()).strip("-_")
    return identifier or prefix


def encode_asset(
    source: str | Path | bytes,
    *,
    media_type: str | None = None,
) -> str:
    """Encode local bytes as a data URI for a network-free report."""
    if isinstance(source, bytes):
        payload = source
        resolved_type = media_type or "application/octet-stream"
    else:
        path = Path(source)
        payload = path.read_bytes()
        guessed, _ = mimetypes.guess_type(path.name)
        resolved_type = media_type or guessed or "application/octet-stream"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{resolved_type};base64,{encoded}"


def render_badge(label: object, *, tone: str = "neutral") -> str:
    """Render a compact status label."""
    allowed_tone = tone if tone in {"neutral", "info", "good", "warn", "danger"} else "neutral"
    return f'<span class="bml-badge bml-badge--{allowed_tone}">{escape(label)}</span>'


def severity_tone(severity: object) -> str:
    """Map editorial severity labels onto badge/card tone tokens."""
    return {
        "critical": "danger",
        "high": "danger",
        "medium": "warn",
        "low": "info",
        "info": "info",
        "good": "good",
    }.get(str(severity).lower(), "neutral")


def render_reading_frame(
    *,
    examined: object,
    observed: object,
    why: object,
    limits: object,
    next_step: object,
) -> str:
    """Render the shared five-part reading frame used by offline reports."""
    return (
        '<dl class="bml-reading-frame">'
        f"<div><dt>What was examined</dt><dd>{escape(examined)}</dd></div>"
        f"<div><dt>Observed result</dt><dd>{escape(observed)}</dd></div>"
        f"<div><dt>Why it matters</dt><dd>{escape(why)}</dd></div>"
        f"<div><dt>Limits</dt><dd>{escape(limits)}</dd></div>"
        f"<div><dt>What next</dt><dd>{escape(next_step)}</dd></div>"
        "</dl>"
    )


def render_card(
    title: object,
    body: object,
    *,
    heading_level: int = 3,
    tone: str = "neutral",
) -> str:
    """Render an escaped text card."""
    level = min(6, max(2, heading_level))
    allowed_tone = tone if tone in {"neutral", "info", "good", "warn", "danger"} else "neutral"
    return (
        f'<article class="bml-card bml-card--{allowed_tone}">'
        f"<h{level}>{escape(title)}</h{level}>"
        f"<p>{escape(body)}</p>"
        "</article>"
    )


def render_list(items: Iterable[object], *, ordered: bool = False) -> str:
    """Render an escaped ordered or unordered list."""
    tag = "ol" if ordered else "ul"
    content = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<{tag}>{content}</{tag}>"


def render_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    caption: str | None = None,
    empty_message: str = "No rows to display.",
    max_rows: int | None = DEFAULT_MAX_TABLE_ROWS,
    max_columns: int | None = DEFAULT_MAX_TABLE_COLUMNS,
) -> str:
    """Render a bounded accessible table with escaped headings and cells."""
    if not rows:
        return f'<p class="bml-empty">{escape(empty_message)}</p>'
    selected = list(columns or _column_order(rows))
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive or None")
    if max_columns is not None and max_columns < 1:
        raise ValueError("max_columns must be positive or None")
    omitted_rows = max(0, len(rows) - max_rows) if max_rows is not None else 0
    omitted_columns = (
        max(0, len(selected) - max_columns) if max_columns is not None else 0
    )
    visible_rows = rows[:max_rows] if max_rows is not None else rows
    visible_columns = selected[:max_columns] if max_columns is not None else selected
    caption_html = f"<caption>{escape(caption)}</caption>" if caption else ""
    heading_html = "".join(
        f'<th scope="col">{escape(column)}</th>' for column in visible_columns
    )
    body_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(_format_cell(row.get(column)))}</td>"
            for column in visible_columns
        )
        + "</tr>"
        for row in visible_rows
    )
    budget_note = ""
    if omitted_rows or omitted_columns:
        details = []
        if omitted_rows:
            details.append(f"{omitted_rows:,} additional rows")
        if omitted_columns:
            details.append(f"{omitted_columns:,} additional columns")
        budget_note = (
            '<p class="bml-budget-note" role="note">Display budget applied: '
            + escape(" and ".join(details))
            + " omitted from this table.</p>"
        )
    return budget_note + (
        '<div class="bml-table-wrap" tabindex="0" role="region" '
        f'aria-label="{escape(caption or "Data table")}">'
        f'<table class="bml-data-table">{caption_html}<thead><tr>{heading_html}</tr></thead>'
        f"<tbody>{body_html}</tbody></table></div>"
    )


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One navigable report section; ``body_html`` must be trusted HTML."""

    key: str
    title: str
    body_html: str
    summary: str | None = None


def render_navigation(sections: Sequence[ReportSection]) -> str:
    """Render landmark navigation linked to report sections."""
    links = "".join(
        f'<li><a href="#{escape(element_id(section.key))}">{escape(section.title)}</a></li>'
        for section in sections
    )
    return (
        '<nav class="bml-nav" aria-label="Report sections">'
        '<p class="bml-nav__title">Contents</p>'
        f"<ul>{links}</ul></nav>"
    )


def render_report(
    title: str,
    sections: Sequence[ReportSection],
    *,
    subtitle: str | None = None,
    metadata: Mapping[str, object] | None = None,
    lang: str = "en",
) -> str:
    """Return a complete, self-contained HTML report document."""
    if not sections:
        raise ValueError("A report requires at least one section")
    seen: set[str] = set()
    for section in sections:
        identifier = element_id(section.key)
        if identifier in seen:
            raise ValueError(f"Duplicate report section id: {identifier}")
        seen.add(identifier)

    subtitle_html = f'<p class="bml-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    metadata_html = _render_metadata(metadata or {})
    section_html = "".join(_render_section(section) for section in sections)
    navigation = render_navigation(sections)
    safe_title = escape(title)
    return f"""<!doctype html>
<html lang="{escape(lang)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="BuildML">
  <title>{safe_title}</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
  <a class="bml-skip-link" href="#main-content">Skip to report content</a>
  <header class="bml-header" role="banner">
    <div>
      <p class="bml-kicker">BuildML report</p>
      <h1>{safe_title}</h1>
      {subtitle_html}
      {metadata_html}
    </div>
    <button class="bml-theme" type="button" aria-pressed="false" aria-controls="main-content">Use dark theme</button>
  </header>
  <div class="bml-report-tools" role="search" aria-label="Search report sections">
    <label for="bml-section-search">Search report</label>
    <input id="bml-section-search" class="bml-section-search" type="search"
      placeholder="Filter sections by title or content" autocomplete="off">
    <p id="bml-section-search-status" class="bml-search-status" aria-live="polite"></p>
  </div>
  <div class="bml-layout">
    {navigation}
    <main id="main-content" tabindex="-1">{section_html}</main>
  </div>
  <footer role="contentinfo"><p>Generated locally by BuildML. This file does not load network assets.</p></footer>
  <script>{REPORT_JS}</script>
</body>
</html>
"""


def write_report(
    path: str | Path,
    title: str,
    sections: Sequence[ReportSection],
    **kwargs: Any,
) -> Path:
    """Render and write a self-contained report as UTF-8."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(title, sections, **kwargs), encoding="utf-8")
    return destination


def _column_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            text = str(key)
            if text not in columns:
                columns.append(text)
    return columns


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    return str(value)


def _render_metadata(metadata: Mapping[str, object]) -> str:
    if not metadata:
        return ""
    items = "".join(
        f"<div><dt>{escape(key)}</dt><dd>{escape(value)}</dd></div>"
        for key, value in metadata.items()
    )
    return f'<dl class="bml-metadata">{items}</dl>'


def _render_section(section: ReportSection) -> str:
    summary = f'<p class="bml-section-summary">{escape(section.summary)}</p>' if section.summary else ""
    return (
        f'<section id="{escape(element_id(section.key))}" class="bml-section">'
        f"<h2>{escape(section.title)}</h2>{summary}{section.body_html}</section>"
    )


REPORT_CSS = """
:root {
  color-scheme: light;
  --bml-bg: #eef2f0;
  --bml-surface: #ffffff;
  --bml-ink: #14201c;
  --bml-muted: #51615c;
  --bml-line: #c9d5d0;
  --bml-line-strong: #a8b8b2;
  --bml-accent: #0b6e4f;
  --bml-accent-ink: #064e38;
  --bml-accent-soft: #dff3ea;
  --bml-focus: #b45309;
  --bml-good: #176b3a;
  --bml-warn: #8a5600;
  --bml-danger: #a62b2b;
  --bml-info: #1d4e89;
  --bml-shadow: 0 10px 28px rgba(20, 32, 28, 0.08);
  --bml-radius: 14px;
  --bml-sans: "Iowan Old Style", "Palatino Linotype", Georgia, "Segoe UI", serif;
  --bml-body: "Segoe UI Variable", "Segoe UI", "Avenir Next", "Helvetica Neue", sans-serif;
  --bml-mono: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-family: var(--bml-body);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; overflow-x: clip; }
body {
  margin: 0; color: var(--bml-ink); line-height: 1.6;
  background:
    radial-gradient(900px 420px at 0% -10%, color-mix(in srgb, var(--bml-accent) 14%, transparent), transparent 55%),
    radial-gradient(700px 360px at 100% 0%, color-mix(in srgb, var(--bml-info) 10%, transparent), transparent 50%),
    var(--bml-bg);
}
a { color: var(--bml-accent-ink); text-underline-offset: 2px; }
a:focus-visible, button:focus-visible, [tabindex="0"]:focus-visible {
  outline: 3px solid var(--bml-focus); outline-offset: 3px;
}
.bml-skip-link {
  position: absolute; left: 1rem; top: -5rem; z-index: 10; padding: .65rem 1rem;
  color: white; background: var(--bml-focus); border-radius: 8px;
}
.bml-skip-link:focus { top: 1rem; }
.bml-header {
  display: flex; justify-content: space-between; gap: 2rem; align-items: flex-start;
  padding: 2.5rem max(1.25rem, calc((100vw - 1180px) / 2)) 2rem;
  color: #f4fffa;
  background:
    linear-gradient(135deg, #064b45 0%, #0b6e4f 55%, #0f766e 100%);
  border-bottom: 1px solid color-mix(in srgb, #fff 18%, transparent);
}
.bml-header h1 {
  margin: .2rem 0; font-family: var(--bml-sans);
  font-size: clamp(2rem, 4.5vw, 3rem); line-height: 1.08; letter-spacing: -0.02em;
}
.bml-kicker {
  margin: 0; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  font-size: .78rem; opacity: .92;
}
.bml-subtitle { margin: .7rem 0 0; max-width: 68ch; color: #d7f0e8; font-size: 1.02rem; }
.bml-theme {
  padding: .6rem .9rem; color: inherit; background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.35); border-radius: 999px; cursor: pointer;
}
.bml-theme:hover { background: rgba(255,255,255,.14); }
.bml-layout {
  display: grid; grid-template-columns: minmax(13rem, 16.5rem) minmax(0, 1fr); gap: 1.75rem;
  max-width: 1180px; margin: 0 auto; padding: 1.75rem 1.25rem 2.5rem;
}
.bml-report-tools {
  max-width: 1180px; margin: 1.1rem auto 0; padding: 0 1.25rem;
  display: grid; gap: .35rem;
}
.bml-report-tools label { display: block; font-weight: 700; font-size: .9rem; }
.bml-section-search {
  width: min(36rem, 100%); padding: .7rem .85rem; color: var(--bml-ink);
  background: var(--bml-surface); border: 1px solid var(--bml-line);
  border-radius: 999px; box-shadow: var(--bml-shadow);
}
.bml-search-status { min-height: 1.4em; margin: .25rem 0 0; color: var(--bml-muted); font-size: .9rem; }
.bml-nav {
  align-self: start; position: sticky; top: 1rem; padding: 1.1rem 1rem;
  background: var(--bml-surface); border: 1px solid var(--bml-line);
  border-radius: var(--bml-radius); box-shadow: var(--bml-shadow);
}
.bml-nav__title { margin: 0 0 .55rem; font-weight: 700; letter-spacing: .02em; }
.bml-nav ul { margin: 0; padding-left: 1.05rem; }
.bml-nav li { margin: .45rem 0; }
.bml-nav a { text-decoration: none; }
.bml-nav a:hover { text-decoration: underline; }
main { min-width: 0; display: grid; gap: 1.1rem; }
.bml-section {
  margin: 0; padding: 1.35rem 1.4rem; background: var(--bml-surface);
  border: 1px solid var(--bml-line); border-radius: var(--bml-radius);
  box-shadow: var(--bml-shadow);
}
.bml-section h2 {
  margin: 0 0 .55rem; color: var(--bml-accent-ink);
  font-family: var(--bml-sans); font-size: 1.45rem; letter-spacing: -0.02em;
}
.bml-section h3 {
  margin: 1.25rem 0 .5rem; font-size: 1.05rem; color: var(--bml-ink);
}
.bml-section-summary { color: var(--bml-muted); font-size: 1.02rem; margin: 0 0 .85rem; }
.bml-metadata {
  display: flex; flex-wrap: wrap; gap: .55rem .85rem; margin: 1.1rem 0 0;
}
.bml-metadata div {
  display: grid; gap: .15rem; min-width: 8rem;
  padding: .55rem .75rem; border-radius: 10px;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
}
.bml-metadata dt { font-weight: 700; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .9; }
.bml-metadata dd { margin: 0; font-family: var(--bml-mono); font-size: .95rem; }
.bml-table-wrap {
  overflow-x: auto; border: 1px solid var(--bml-line); border-radius: 12px;
  background: color-mix(in srgb, var(--bml-surface) 92%, var(--bml-bg));
}
.bml-table-tools {
  display: flex; flex-wrap: wrap; gap: .55rem; align-items: center; margin: .75rem 0 .35rem;
}
.bml-table-search {
  min-width: min(24rem, 100%); padding: .55rem .7rem; color: var(--bml-ink);
  background: var(--bml-surface); border: 1px solid var(--bml-line); border-radius: 10px;
}
table { width: 100%; border-collapse: collapse; margin: 0; font-size: .92rem; }
caption { padding: .65rem .75rem; font-weight: 700; text-align: left; }
th, td {
  padding: .65rem .75rem; border-bottom: 1px solid var(--bml-line);
  text-align: left; vertical-align: top;
}
th { background: var(--bml-accent-soft); position: sticky; top: 0; }
th button {
  width: 100%; padding: 0; color: inherit; font: inherit; font-weight: 700;
  text-align: left; background: transparent; border: 0; cursor: pointer;
}
tbody tr:hover td { background: color-mix(in srgb, var(--bml-accent) 6%, transparent); }
.bml-card {
  padding: .95rem 1.05rem; margin: .75rem 0; border: 1px solid var(--bml-line);
  border-left: .35rem solid var(--bml-muted); border-radius: 12px;
  background: color-mix(in srgb, var(--bml-surface) 88%, var(--bml-bg));
}
.bml-card h3, .bml-card h4 { margin: 0 0 .3rem; font-family: var(--bml-sans); }
.bml-card p { margin: 0; color: var(--bml-muted); }
.bml-card--info { border-left-color: var(--bml-info); }
.bml-card--good { border-left-color: var(--bml-good); }
.bml-card--warn { border-left-color: var(--bml-warn); }
.bml-card--danger { border-left-color: var(--bml-danger); }
.bml-badge {
  display: inline-flex; align-items: center; gap: .25rem;
  padding: .15rem .55rem; border-radius: 999px; font-size: .78rem; font-weight: 700;
  border: 1px solid currentColor;
}
.bml-badge--info { color: var(--bml-info); background: color-mix(in srgb, var(--bml-info) 12%, transparent); }
.bml-badge--good { color: var(--bml-good); background: color-mix(in srgb, var(--bml-good) 12%, transparent); }
.bml-badge--warn { color: var(--bml-warn); background: color-mix(in srgb, var(--bml-warn) 12%, transparent); }
.bml-badge--danger { color: var(--bml-danger); background: color-mix(in srgb, var(--bml-danger) 12%, transparent); }
.bml-badge--neutral { color: var(--bml-muted); }
.bml-empty { color: var(--bml-muted); font-style: italic; }
.bml-budget-note { color: var(--bml-muted); font-size: .9rem; }
.bml-reading-frame {
  display: grid; gap: .55rem; padding: 1rem 1.1rem; margin: .85rem 0 1.1rem;
  background: linear-gradient(180deg, var(--bml-accent-soft), color-mix(in srgb, var(--bml-accent-soft) 40%, transparent));
  border: 1px solid color-mix(in srgb, var(--bml-accent) 25%, var(--bml-line));
  border-radius: 12px;
}
.bml-reading-frame div {
  display: grid; grid-template-columns: minmax(8.5rem, 11rem) 1fr; gap: .75rem;
}
.bml-reading-frame dt { font-weight: 700; color: var(--bml-accent-ink); }
.bml-reading-frame dd { margin: 0; }
details {
  margin: .9rem 0; padding: .75rem .9rem; border: 1px dashed var(--bml-line-strong);
  border-radius: 12px; background: color-mix(in srgb, var(--bml-bg) 55%, var(--bml-surface));
}
summary { cursor: pointer; font-weight: 700; color: var(--bml-accent-ink); }
.bml-gallery {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 1rem;
}
.bml-figure {
  margin: 0; padding: .65rem; border: 1px solid var(--bml-line); border-radius: 12px;
  background: color-mix(in srgb, var(--bml-surface) 92%, var(--bml-bg));
}
.bml-figure img { display: block; width: 100%; height: auto; border-radius: 8px; }
.bml-figure__expand { padding: 0; border: 0; background: transparent; cursor: zoom-in; width: 100%; }
.bml-figure figcaption { padding-top: .55rem; color: var(--bml-muted); font-size: .9rem; }
.bml-figure-dialog {
  width: min(96vw, 90rem); max-height: 94vh; padding: .65rem;
  border: 1px solid var(--bml-line); border-radius: 14px; background: var(--bml-surface);
}
.bml-figure-dialog img { display: block; max-width: 100%; max-height: 86vh; margin: auto; }
.bml-figure-dialog::backdrop { background: rgb(0 0 0 / 75%); }
.bml-json {
  max-height: 42rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere;
  font-family: var(--bml-mono); font-size: .85rem; padding: .85rem;
  border-radius: 12px; border: 1px solid var(--bml-line); background: color-mix(in srgb, var(--bml-bg) 70%, var(--bml-surface));
}
.bml-finding {
  margin: 1rem 0; padding: .9rem 1.05rem; border-left: .35rem solid var(--bml-accent);
  border-radius: 0 12px 12px 0;
  background: color-mix(in srgb, var(--bml-accent-soft) 45%, transparent);
}
.bml-finding.severity-high, .bml-finding.severity-critical {
  border-left-color: var(--bml-danger);
  background: color-mix(in srgb, var(--bml-danger) 10%, transparent);
}
.bml-finding.severity-medium {
  border-left-color: var(--bml-warn);
  background: color-mix(in srgb, var(--bml-warn) 10%, transparent);
}
footer {
  max-width: 1180px; margin: 0 auto; padding: 0 1.25rem 2.5rem;
  color: var(--bml-muted); font-size: .92rem;
}
body.bml-dark {
  color-scheme: dark;
  --bml-bg: #101714; --bml-surface: #1a2420; --bml-ink: #edf4f1;
  --bml-muted: #b3c2bd; --bml-line: #3a4a44; --bml-line-strong: #567068;
  --bml-accent: #3ddc97; --bml-accent-ink: #d7ffe9; --bml-accent-soft: #163528;
  --bml-info: #7db4f0; --bml-shadow: 0 12px 32px rgba(0,0,0,.35);
}
@media (max-width: 860px) {
  .bml-header { display: block; padding: 1.75rem 1rem; }
  .bml-theme { margin-top: 1rem; }
  .bml-layout { grid-template-columns: 1fr; padding: 1.25rem 1rem 2rem; }
  .bml-nav { position: static; }
  .bml-reading-frame div { grid-template-columns: 1fr; gap: .15rem; }
  .bml-metadata div { min-width: calc(50% - .5rem); }
}
@media print {
  .bml-skip-link, .bml-theme, .bml-nav, .bml-report-tools, .bml-table-tools { display: none; }
  .bml-layout { display: block; max-width: none; padding: 0; }
  .bml-section { break-inside: avoid; border-color: #777; box-shadow: none; }
  body { background: white; }
}
"""


REPORT_JS = """
(() => {
  const button = document.querySelector(".bml-theme");
  if (button) {
    button.addEventListener("click", () => {
      const enabled = document.body.classList.toggle("bml-dark");
      button.setAttribute("aria-pressed", String(enabled));
      button.textContent = enabled ? "Use light theme" : "Use dark theme";
    });
  }

  const sectionSearch = document.querySelector(".bml-section-search");
  const sectionStatus = document.querySelector(".bml-search-status");
  const sections = Array.from(document.querySelectorAll(".bml-section"));
  if (sectionSearch) {
    sectionSearch.addEventListener("input", () => {
      const query = sectionSearch.value.toLocaleLowerCase().trim();
      let visible = 0;
      sections.forEach((section) => {
        const matches = !query || section.textContent.toLocaleLowerCase().includes(query);
        section.hidden = !matches;
        if (matches) visible += 1;
      });
      if (sectionStatus) {
        sectionStatus.textContent = query
          ? `${visible} of ${sections.length} sections shown`
          : "";
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
        event.preventDefault();
        sectionSearch.focus();
      }
    });
  }

  document.querySelectorAll(".bml-data-table").forEach((table, tableIndex) => {
    const wrapper = table.closest(".bml-table-wrap");
    if (!wrapper) return;
    const tools = document.createElement("div");
    tools.className = "bml-table-tools";
    const label = document.createElement("label");
    label.setAttribute("for", `bml-table-search-${tableIndex}`);
    label.textContent = "Filter rows";
    const search = document.createElement("input");
    search.id = `bml-table-search-${tableIndex}`;
    search.className = "bml-table-search";
    search.type = "search";
    search.placeholder = "Search this table";
    search.addEventListener("input", () => {
      const query = search.value.toLocaleLowerCase();
      table.querySelectorAll("tbody tr").forEach((row) => {
        row.hidden = !row.textContent.toLocaleLowerCase().includes(query);
      });
    });
    tools.append(label, search);
    wrapper.before(tools);

    table.querySelectorAll("thead th").forEach((heading, columnIndex) => {
      const text = heading.textContent;
      const sort = document.createElement("button");
      sort.type = "button";
      sort.textContent = `${text} ↕`;
      sort.setAttribute("aria-label", `Sort by ${text}`);
      let ascending = true;
      sort.addEventListener("click", () => {
        const rows = Array.from(table.querySelectorAll("tbody tr"));
        rows.sort((left, right) => {
          const a = left.children[columnIndex]?.textContent.trim() || "";
          const b = right.children[columnIndex]?.textContent.trim() || "";
          const an = Number(a), bn = Number(b);
          const compared = Number.isNaN(an) || Number.isNaN(bn)
            ? a.localeCompare(b, undefined, {numeric: true})
            : an - bn;
          return ascending ? compared : -compared;
        });
        rows.forEach((row) => table.tBodies[0].append(row));
        sort.textContent = `${text} ${ascending ? "↑" : "↓"}`;
        ascending = !ascending;
      });
      heading.replaceChildren(sort);
    });
  });

  const dialog = document.createElement("dialog");
  dialog.className = "bml-figure-dialog";
  const dialogImage = document.createElement("img");
  dialog.append(dialogImage);
  document.body.append(dialog);
  document.querySelectorAll(".bml-figure__expand").forEach((expand) => {
    expand.addEventListener("click", () => {
      const image = expand.querySelector("img");
      if (!image) return;
      dialogImage.src = image.src;
      dialogImage.alt = image.alt;
      dialog.showModal();
    });
  });
  dialog.addEventListener("click", () => dialog.close());
})();
"""

