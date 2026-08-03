"""FastAPI application factory for the local EDA Teaching Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dashboard.charts import build_chart_catalog, charts_for_domain
from buildml.dashboard.domains import DOMAIN_BY_KEY, DOMAINS
from buildml.dashboard.exports import export_csv, export_pdf, list_csv_sections
from buildml.dashboard.offline import export_studio_html
from buildml.dashboard.serialize import flagged_column_names, json_safe
from buildml.dashboard.state import get_state
from buildml.dashboard.teaching import build_teaching_studios
from buildml.explain.concepts import CONCEPT_NOTES, get_concept

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:  # pragma: no cover - exercised when dashboard extra missing
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Query = None  # type: ignore[assignment]
    Request = Any  # type: ignore[misc, assignment]
    HTMLResponse = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    StaticFiles = None  # type: ignore[assignment]
    Jinja2Templates = None  # type: ignore[assignment]


def create_app() -> Any:
    """Build the ASGI app, wiring routes to whatever report is installed.

    A factory rather than a module-level app, because routes read from
    process-local state that has to be installed first. Building the app at
    import time would create routes with nothing behind them.

    The app serves the studio page, JSON endpoints for boards and chart
    catalogues, the teaching content, and the CSV, PDF, and offline-HTML export
    routes. Static assets and the vendored Plotly are mounted from the package.

    Returns
    -------
    Any
        A FastAPI application, ready for uvicorn. Typed loosely so this module
        imports cleanly when FastAPI is absent.

    Raises
    ------
    MissingExtraError
        If FastAPI, Jinja2, or the static-files support is unavailable. Install
        with ``pip install 'buildml[dashboard]'``.

    Notes
    -----
    **The state must be installed first.** Every route calls
    :func:`~buildml.dashboard.state.get_state`, which raises without it. Use
    :func:`~buildml.dashboard.launch.launch_eda_app`, which does this in the
    right order.

    **No authentication, and the API docs are disabled.** This is meant for
    ``127.0.0.1`` and nothing else; anyone who can reach the port can read the
    whole report.

    **One report per process.** The state is a single global slot.

    See Also
    --------
    buildml.dashboard.launch.launch_eda_app : The supported way to start this.
    """
    if FastAPI is None or Jinja2Templates is None or StaticFiles is None:
        raise MissingExtraError("dashboard", "EDA Teaching Studio app")

    root = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(root / "templates"))
    app = FastAPI(title="BuildML EDA Studio", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(root / "static")), name="static")
    plotly_dir = _plotly_static_dir()
    if plotly_dir is not None:
        app.mount("/vendor/plotly", StaticFiles(directory=str(plotly_dir)), name="plotly")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        state = get_state()
        context = {
            "request": request,
            "title": state.title,
            "app_name": "BuildML EDA Studio",
        }
        try:
            return templates.TemplateResponse(request, "index.html", context)
        except TypeError:
            return templates.TemplateResponse("index.html", context)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        state = get_state()
        return {"ok": True, "title": state.title, "product": "buildml-eda-studio"}

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        state = get_state()
        overview = state.report_dict.get("overview") or {}
        return json_safe(
            {
                "title": state.title,
                "session": state.session_meta,
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
                    "warnings": state.report_dict.get("warnings") or [],
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
                "csv_sections": list_csv_sections(state.report_dict),
            }
        )

    @app.get("/api/cockpit")
    def cockpit() -> dict[str, Any]:
        state = get_state()
        report = state.report_dict
        findings = report.get("findings") or []
        severity_counts: dict[str, int] = {}
        for item in findings:
            key = str(item.get("severity", "info")).lower()
            severity_counts[key] = severity_counts.get(key, 0) + 1
        studios = build_teaching_studios(report)
        return json_safe(
            {
                "overview": report.get("overview") or {},
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
        )

    @app.get("/api/domains/{domain_key}")
    def domain_board(domain_key: str) -> dict[str, Any]:
        if domain_key == "academy":
            academy = DOMAIN_BY_KEY["academy"]
            return {
                "domain": {
                    "key": academy.key,
                    "title": academy.title,
                    "short": academy.short,
                    "icon": academy.icon,
                    "concept_keys": list(academy.concept_keys),
                    "csv_sections": list(academy.csv_sections),
                },
                "concepts": _concept_index(),
                "teaching": None,
            }
        domain = DOMAIN_BY_KEY.get(domain_key)
        if domain is None:
            raise HTTPException(status_code=404, detail=f"Unknown domain: {domain_key}")
        state = get_state()
        report = state.report_dict
        studios = build_teaching_studios(report)
        return json_safe(
            {
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
                    for item in report.get("findings") or []
                    if _finding_matches_domain(item, domain.key)
                ],
                "chart_ids": charts_for_domain(domain.key),
            }
        )

    @app.get("/api/charts")
    def charts(theme: str = Query(default="light")) -> dict[str, Any]:
        state = get_state()
        resolved = "dark" if str(theme).lower() == "dark" else "light"
        return build_chart_catalog(state.report_dict, theme=resolved)

    @app.get("/api/charts/{chart_id}")
    def chart(chart_id: str, theme: str = Query(default="light")) -> dict[str, Any]:
        state = get_state()
        resolved = "dark" if str(theme).lower() == "dark" else "light"
        catalog = build_chart_catalog(state.report_dict, theme=resolved)
        if chart_id not in catalog:
            raise HTTPException(status_code=404, detail=f"Unknown chart: {chart_id}")
        return {"id": chart_id, "figure": catalog[chart_id]}

    @app.get("/api/concepts")
    def concepts(q: str | None = Query(default=None)) -> dict[str, Any]:
        items = _concept_index()
        if q:
            needle = q.strip().lower()
            items = [
                item
                for item in items
                if needle in item["key"]
                or needle in item["title"].lower()
                or needle in item["summary"].lower()
                or any(needle in detail.lower() for detail in item["details"])
            ]
        return {"count": len(items), "concepts": items}

    @app.get("/api/concepts/{concept_key}")
    def concept_detail(concept_key: str) -> dict[str, Any]:
        try:
            note = get_concept(concept_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        related = []
        for key in note.related_concepts:
            try:
                related_note = get_concept(key)
            except KeyError:
                continue
            related.append(
                {
                    "key": related_note.key,
                    "title": related_note.title,
                    "summary": related_note.summary,
                }
            )
        state = get_state()
        studios = build_teaching_studios(state.report_dict)
        linked_domains = [
            key for key, studio in studios.items() if concept_key in (studio.get("concepts") or [])
        ]
        return {
            "concept": note.to_dict(),
            "related": related,
            "linked_domains": linked_domains,
        }

    @app.get("/api/search")
    def search(q: str = Query(min_length=1)) -> dict[str, Any]:
        state = get_state()
        needle = q.strip().lower()
        findings = [
            item
            for item in state.report_dict.get("findings") or []
            if needle in str(item.get("title", "")).lower()
            or needle in str(item.get("detail", "")).lower()
            or needle in str(item.get("key", "")).lower()
        ]
        concepts = [
            item
            for item in _concept_index()
            if needle in item["key"]
            or needle in item["title"].lower()
            or needle in item["summary"].lower()
        ]
        domains = [
            {"key": d.key, "title": d.title, "short": d.short}
            for d in DOMAINS
            if needle in d.key or needle in d.title.lower() or needle in d.short.lower()
        ]
        return {
            "query": q,
            "findings": findings[:30],
            "concepts": concepts[:30],
            "domains": domains,
        }

    @app.get("/api/export/csv/{section}")
    def csv_export(section: str) -> Response:
        state = get_state()
        try:
            filename, content = export_csv(state.report_dict, section)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/export/pdf")
    def pdf_export(view: str = Query(default="briefing")) -> Response:
        state = get_state()
        payload = export_pdf(state.report_dict, view=view, title=state.title)
        filename = f"buildml_eda_{view}.pdf"
        return Response(
            content=payload,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/export/html")
    def html_export() -> Response:
        """Download an offline Teaching Studio snapshot (same SPA surface)."""
        import tempfile

        state = get_state()
        with tempfile.TemporaryDirectory(prefix="buildml-studio-") as tmp:
            path = Path(tmp) / "buildml_eda_studio.html"
            export_studio_html(
                state.report_dict,
                path,
                title=state.title,
                session_meta=state.session_meta,
            )
            payload = path.read_bytes()
        return Response(
            content=payload,
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="buildml_eda_studio.html"'
            },
        )

    @app.get("/api/report")
    def raw_report() -> JSONResponse:
        state = get_state()
        # Exclude non-serializable figure objects; metadata only.
        return JSONResponse(state.report_dict)

    return app


def _plotly_static_dir() -> Path | None:
    try:
        import plotly
    except ImportError:
        return None
    package_dir = Path(plotly.__file__).resolve().parent
    for candidate in (
        package_dir / "package_data" / "plotly.min.js",
        package_dir / "package_data",
    ):
        if candidate.is_file():
            return candidate.parent
        if candidate.is_dir() and (candidate / "plotly.min.js").exists():
            return candidate
    # Newer plotly layouts may ship under different paths.
    matches = list(package_dir.rglob("plotly.min.js"))
    if matches:
        return matches[0].parent
    return None


def _concept_index() -> list[dict[str, Any]]:
    return [
        {
            "key": note.key,
            "title": note.title,
            "summary": note.summary,
            "details": list(note.details),
            "related_concepts": list(note.related_concepts),
            "definition": note.definition,
            "intuition": note.intuition,
            "formal_idea": note.formal_idea,
            "why_it_matters": list(note.why_it_matters),
            "how_buildml_uses": list(note.how_buildml_uses),
            "interpretation_rules": list(note.interpretation_rules),
            "assumptions": list(note.assumptions),
            "failure_modes": list(note.failure_modes),
            "anti_patterns": list(note.anti_patterns),
            "worked_example_pattern": list(note.worked_example_pattern),
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
