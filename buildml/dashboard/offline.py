# ruff: noqa: E501
"""Offline Teaching Studio HTML snapshot (same product surface as eda_app)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dashboard.charts import build_chart_catalog, charts_for_domain
from buildml.dashboard.domains import DOMAINS
from buildml.dashboard.exports import list_csv_sections
from buildml.dashboard.serialize import flagged_column_names, json_safe
from buildml.dashboard.teaching import build_teaching_studios
from buildml.explain.concepts import CONCEPT_NOTES, get_concept


def export_studio_html(
    report: dict[str, Any],
    path: str | Path,
    *,
    title: str = "BuildML EDA Studio",
    session_meta: dict[str, Any] | None = None,
) -> Path:
    """Save the whole interactive studio as one HTML file that needs no server.

    The dashboard, frozen. Same boards, same charts, same teaching panels, same
    interactivity — hovering, zooming, switching boards, toggling theme — with
    the entire application inlined into a single file. No Python, no server, no
    network.

    Everything goes in: the Plotly library, the stylesheet, the application
    script, both light and dark chart catalogues, the teaching studios, and the
    concept notes. Both themes are embedded because the toggle has to work
    offline, which means the charts exist twice.

    This is what you send to someone who needs to explore the analysis rather
    than read a summary of it, and who is not going to install anything.

    Parameters
    ----------
    report:
        The report as a dict, from
        :meth:`~buildml.eda.report.EDAReport.to_dict`.
    path:
        Where to write. Parent directories are created.
    title:
        Header and window title.
    session_meta:
        Extra Session facts for the cockpit board.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    MissingExtraError
        If Plotly is not installed. Install with
        ``pip install 'buildml[dashboard]'``.
    OSError
        If the file cannot be written.

    Notes
    -----
    **These files are large.** Plotly alone is several megabytes, and the chart
    catalogues are embedded twice for the theme toggle. Ten to twenty megabytes
    is typical, which is fine for a shared drive and often too large for email.

    **Everything in the report is in the file.** Column names, distributions,
    example values, findings. Treat it with the same care as the data.

    **No live data.** It is a snapshot; regenerate it when the analysis changes.

    Examples
    --------
    ::

        report = session.eda()
        export_studio_html(
            report.to_dict(),
            "artifacts/studio.html",
            title="Churn dataset · 2026-08",
        )

    See Also
    --------
    buildml.dashboard.launch.launch_eda_app : The live version.
    buildml.eda.html_report.export_eda_html : Smaller, static, no Plotly.
    """
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise MissingExtraError("dashboard", "Teaching Studio offline HTML export") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_offline_bundle(report, title=title, session_meta=session_meta or {})
    html = render_offline_html(bundle)
    destination.write_text(html, encoding="utf-8")
    return destination


def build_offline_bundle(
    report: dict[str, Any],
    *,
    title: str = "BuildML EDA Studio",
    session_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gather everything the offline app needs into one serialisable payload.

    The data half of the export, separated from the HTML half. It collects the
    boards, the chart catalogues for both themes, the teaching studios, the
    concept notes, the findings with their severity counts, and the export
    manifest.

    Splitting build from render is what makes this testable. The bundle can be
    inspected and asserted against without generating a ten-megabyte document,
    and a caller who wants to feed the same data into a different shell can.

    Parameters
    ----------
    report:
        The report as a dict.
    title:
        Title carried into the bundle.
    session_meta:
        Extra Session facts for the cockpit.

    Returns
    -------
    dict
        JSON-safe throughout — every value passed through
        :func:`~buildml.dashboard.serialize.json_safe`, since analyzer output is
        full of NumPy scalars that would otherwise fail at serialisation time,
        after the whole payload has been assembled.

    Raises
    ------
    MissingExtraError
        If Plotly is not installed; the chart catalogues need it.

    Notes
    -----
    **Both theme catalogues are built**, so this does roughly twice the chart
    work of a single-theme render.

    See Also
    --------
    render_offline_html : Turning this into the document.
    """
    meta = session_meta or {}
    overview = report.get("overview") or {}
    studios = build_teaching_studios(report)
    findings = report.get("findings") or []
    severity_counts: dict[str, int] = {}
    for item in findings:
        key = str(item.get("severity", "info")).lower()
        severity_counts[key] = severity_counts.get(key, 0) + 1

    cockpit = {
        "overview": overview,
        "quality": {
            "completeness_score": (report.get("quality") or {}).get("completeness_score"),
            "missing_cell_count": (report.get("quality") or {}).get("missing_cell_count"),
            "constant_columns": (report.get("quality") or {}).get("constant_columns"),
            "id_like_columns": (report.get("quality") or {}).get("id_like_columns"),
        },
        "findings": findings,
        "recommendations": report.get("recommendation_details") or [],
        "narrative": report.get("narrative") or [],
        "warnings": report.get("warnings") or [],
        "severity_counts": severity_counts,
        "readiness": _readiness(report),
        "teaching": studios["cockpit"],
        "chart_ids": charts_for_domain("cockpit"),
    }

    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        if domain.key == "academy":
            domains["academy"] = {
                "domain": {
                    "key": domain.key,
                    "title": domain.title,
                    "short": domain.short,
                    "icon": domain.icon,
                    "concept_keys": list(domain.concept_keys),
                    "csv_sections": list(domain.csv_sections),
                },
                "concepts": _concept_index(),
                "teaching": None,
            }
            continue
        domains[domain.key] = {
            "domain": {
                "key": domain.key,
                "title": domain.title,
                "short": domain.short,
                "icon": domain.icon,
                "concept_keys": list(domain.concept_keys),
                "csv_sections": list(domain.csv_sections),
            },
            "sections": {key: report.get(key) for key in domain.report_keys},
            "teaching": studios.get(domain.key),
            "findings": [
                item
                for item in findings
                if _finding_matches_domain(item, domain.key)
            ],
            "chart_ids": charts_for_domain(domain.key),
        }

    concepts = _concept_index()
    concept_details: dict[str, Any] = {}
    for item in concepts:
        key = item["key"]
        try:
            note = get_concept(key)
        except KeyError:
            continue
        related = []
        for related_key in note.related_concepts:
            try:
                related_note = get_concept(related_key)
            except KeyError:
                continue
            related.append(
                {
                    "key": related_note.key,
                    "title": related_note.title,
                    "summary": related_note.summary,
                }
            )
        linked_domains = [
            domain_key
            for domain_key, studio in studios.items()
            if key in (studio.get("concepts") or [])
        ]
        concept_details[key] = {
            "concept": note.to_dict(),
            "related": related,
            "linked_domains": linked_domains,
        }

    return json_safe(
        {
            "title": title,
            "app_name": "BuildML EDA Studio",
            "offline": True,
            "meta": {
                "title": title,
                "session": meta,
                "overview": {
                    "n_rows": overview.get("n_rows"),
                    "n_columns": overview.get("n_columns"),
                    "analysis_rows": overview.get("analysis_rows"),
                    "engine": overview.get("engine"),
                    "mode": overview.get("mode"),
                    "has_native": overview.get("has_native"),
                    "has_lazy_native": overview.get("has_lazy_native"),
                    "engine_disclosures": overview.get("engine_disclosures"),
                    "eligible_feature_columns": overview.get("eligible_feature_columns"),
                    "warnings": report.get("warnings") or [],
                },
                "domains": [
                    {
                        "key": domain.key,
                        "title": domain.title,
                        "short": domain.short,
                        "icon": domain.icon,
                    }
                    for domain in DOMAINS
                ],
                "csv_sections": list_csv_sections(report),
            },
            "cockpit": cockpit,
            "domains": domains,
            "concepts": concepts,
            "concept_details": concept_details,
            "charts_light": build_chart_catalog(report, theme="light"),
            "charts_dark": build_chart_catalog(report, theme="dark"),
            "report": report,
        }
    )


def render_offline_html(bundle: dict[str, Any]) -> str:
    """Inline the application's own assets around a bundle to make one document.

    Reads the same CSS and JavaScript the live dashboard serves and embeds them
    with the data. Using the identical assets is what guarantees the offline
    file behaves like the served app — a separate offline template would drift,
    and the drift would only show up in the artifact you handed to someone else.

    One adjustment is needed. The application script imports its icon module by
    relative path, which requires a server to resolve; that import is rewritten
    to an import map entry pointing at the icons inlined in the same document.

    Parameters
    ----------
    bundle:
        The payload from :func:`build_offline_bundle`.

    Returns
    -------
    str
        A complete HTML document with the stylesheet, the scripts, Plotly, and
        the bundle all inlined.

    Raises
    ------
    OSError
        If the static assets cannot be read from the package.

    Notes
    -----
    **The whole document is assembled in memory** before being returned, so peak
    memory is roughly the file size — tens of megabytes.

    **Plotly is embedded in full.** It is the largest single component and the
    reason these files are what they are.

    See Also
    --------
    export_studio_html : Build and write in one call.
    """
    root = Path(__file__).resolve().parent
    tokens = (root / "static" / "css" / "tokens.css").read_text(encoding="utf-8")
    app_css = (root / "static" / "css" / "app.css").read_text(encoding="utf-8")
    icons_js = (root / "static" / "js" / "icons.js").read_text(encoding="utf-8")
    app_js = (root / "static" / "js" / "app.js").read_text(encoding="utf-8")
    # Offline build: rewrite relative icon import to same-document module.
    app_js = app_js.replace(
        'from "./icons.js"',
        'from "#buildml-icons"',
    )
    plotly_js = _read_plotly_min()
    payload = json.dumps(bundle, ensure_ascii=False)
    title = bundle.get("title") or "BuildML EDA Studio"
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_escape(title)} · Offline Teaching Studio</title>
  <style>{tokens}\n{app_css}</style>
  <script>{plotly_js}</script>
  <!-- Inline modules via blob URLs — no network or local server required. -->
  <script type="module">
    const iconsSource = {json.dumps(icons_js)};
    const appSource = {json.dumps(app_js)};
    const iconsUrl = URL.createObjectURL(new Blob([iconsSource], {{ type: "text/javascript" }}));
    const rewritten = appSource.replaceAll("#buildml-icons", iconsUrl);
    const appUrl = URL.createObjectURL(new Blob([rewritten], {{ type: "text/javascript" }}));
    window.__BUILDML_OFFLINE__ = {payload};
    await import(appUrl);
  </script>
</head>
<body>
  <a class="skip-link" href="#main">Skip to main content</a>
  <div class="app-shell" id="app">
    <aside class="sidebar" aria-label="Primary">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="8" fill="var(--accent)"/>
            <path d="M8 20.5L13.2 11l4.1 7.2L20.8 13 24 20.5" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div>
          <div class="brand-name">BuildML EDA Studio</div>
          <div class="brand-sub">Teaching Studio · Offline</div>
        </div>
      </div>
      <nav class="nav" id="domain-nav" aria-label="Analysis domains"></nav>
      <div class="sidebar-foot">
        <button type="button" class="ghost-btn" id="theme-toggle" aria-pressed="false" title="Toggle theme">
          <span data-icon="moon" aria-hidden="true"></span>
          <span>Theme</span>
        </button>
      </div>
    </aside>
    <div class="main-column">
      <header class="topbar">
        <div class="topbar-left">
          <button type="button" class="icon-btn mobile-nav" id="nav-toggle" aria-label="Open navigation">
            <span data-icon="menu"></span>
          </button>
          <div>
            <h1 class="page-title" id="page-title">Command cockpit</h1>
            <p class="page-sub" id="page-sub">Offline snapshot of the Teaching Studio</p>
          </div>
        </div>
        <div class="topbar-actions">
          <label class="search-field" aria-label="Search findings and concepts">
            <span data-icon="search" aria-hidden="true"></span>
            <input id="global-search" type="search" placeholder="Search findings, domains, concepts" autocomplete="off" />
          </label>
          <span class="btn btn-secondary" id="pdf-download" title="Open the live Teaching Studio to download PDF/CSV exports.">
            Offline snapshot
          </span>
        </div>
      </header>
      <div class="filter-bar" id="filter-bar" hidden>
        <label>
          Severity
          <select id="severity-filter">
            <option value="all">All</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
            <option value="info">Info</option>
          </select>
        </label>
        <label>
          Section filter
          <input id="section-filter" type="search" placeholder="Filter tables and cards" />
        </label>
      </div>
      <main id="main" tabindex="-1"></main>
    </div>
  </div>
  <div id="drawer-backdrop" class="drawer-backdrop" hidden></div>
  <aside id="concept-drawer" class="concept-drawer" aria-hidden="true">
    <div class="drawer-head">
      <h2 id="drawer-title">Concept</h2>
      <button type="button" class="icon-btn" id="drawer-close" aria-label="Close concept">×</button>
    </div>
    <div id="drawer-body" class="drawer-body"></div>
  </aside>
  <dialog id="figure-modal">
    <form method="dialog">
      <div class="drawer-head">
        <h2 id="modal-title">Figure</h2>
        <button type="submit" class="icon-btn" aria-label="Close figure">×</button>
      </div>
      <div id="modal-figure" class="chart-host"></div>
    </form>
  </dialog>
  <div id="toast" class="toast" hidden></div>
</body>
</html>
"""


def _read_plotly_min() -> str:
    try:
        import plotly
    except ImportError as exc:
        raise MissingExtraError("dashboard", "Teaching Studio offline HTML export") from exc
    package_dir = Path(plotly.__file__).resolve().parent
    matches = list(package_dir.rglob("plotly.min.js"))
    if not matches:
        raise MissingExtraError("dashboard", "Teaching Studio offline HTML export (plotly.min.js)")
    return matches[0].read_text(encoding="utf-8", errors="ignore")


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _concept_index() -> list[dict[str, Any]]:
    return [
        {
            "key": note.key,
            "title": note.title,
            "summary": note.summary,
            "details": list(note.details),
            "related_concepts": list(note.related_concepts),
        }
        for note in sorted(CONCEPT_NOTES.values(), key=lambda item: item.title.lower())
    ]


def _readiness(report: dict[str, Any]) -> dict[str, Any]:
    findings = report.get("findings") or []
    blocking = [
        item for item in findings if str(item.get("severity", "")).lower() in {"high", "critical"}
    ]
    quality = report.get("quality") or {}
    drift = report.get("drift") or {}
    return {
        "status": "blocked" if blocking else "review",
        "blocking_findings": len(blocking),
        "completeness_score": quality.get("completeness_score"),
        "drift_available": bool(drift.get("available")),
        "drift_flags": len(flagged_column_names(drift.get("flagged_columns"))),
        "sampling_disclosed": bool(report.get("warnings")),
        "next_actions": [
            {
                "title": item.get("title"),
                "priority": item.get("priority"),
                "rationale": item.get("rationale"),
                "api": ((item.get("action") or {}).get("operation")),
            }
            for item in (report.get("recommendation_details") or [])[:8]
        ],
    }


def _finding_matches_domain(finding: dict[str, Any], domain_key: str) -> bool:
    key = str(finding.get("key", "")).lower()
    prefixes = {
        "quality": ("quality.",),
        "features": ("univariate.", "feature."),
        "relationships": ("bivariate.", "mi.", "correlation."),
        "multivariate": ("multivariate.", "vif.", "pca."),
        "target": ("target.", "drift."),
        "outliers": ("outliers.", "outlier."),
        "visuals": (),
        "cockpit": (
            "eda.",
            "quality.",
            "target.",
            "drift.",
            "outliers.",
            "outlier.",
            "bivariate.",
            "multivariate.",
            "univariate.",
        ),
    }
    accepted = prefixes.get(domain_key)
    if accepted is None:
        return False
    if not accepted:
        return True
    return any(key.startswith(prefix) for prefix in accepted)

