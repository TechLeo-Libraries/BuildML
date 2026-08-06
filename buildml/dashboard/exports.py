"""CSV and PDF exporters for EDA dashboard views."""

from __future__ import annotations

import csv
import io
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dashboard.serialize import flagged_column_names
from buildml.dashboard.teaching import build_teaching_studios
from buildml.explain.concepts import CONCEPT_NOTES

_CSV_BUILDERS = {
    "findings": lambda report: _rows_findings(report),
    "recommendations": lambda report: _rows_recommendations(report),
    "missing_rates": lambda report: _rows_missing(report),
    "quality_flags": lambda report: _rows_quality_flags(report),
    "univariate_numeric": lambda report: _rows_univariate_numeric(report),
    "univariate_categorical": lambda report: _rows_univariate_categorical(report),
    "correlations": lambda report: _rows_correlations(report),
    "spearman": lambda report: _rows_spearman(report),
    "cramers_v": lambda report: _rows_cramers(report),
    "mutual_information": lambda report: _rows_mi(report),
    "vif": lambda report: _rows_vif(report),
    "pca": lambda report: _rows_pca(report),
    "target_summary": lambda report: _rows_target(report),
    "drift": lambda report: _rows_drift(report),
    "outliers": lambda report: _rows_outliers(report),
    "adaptive_plan": lambda report: _rows_plan(report),
    "concepts": lambda report: _rows_concepts(report),
    "roles": lambda report: _rows_roles(report),
}


def list_csv_sections(report: dict[str, Any]) -> list[dict[str, str]]:
    """List the tables this report can actually produce, with their row counts.

    Builds each section to see whether it has rows, and reports only the ones
    that do. So the download menu offers a drift table when drift was analysed
    and does not when it was not: rather than offering everything and returning
    an empty file for half of it.

    Parameters
    ----------
    report:
        The report as a dict.

    Returns
    -------
    list of dict
        One entry per non-empty section, with ``key`` for the download call,
        ``label`` for display, and ``rows`` as a string count. Row counts are
        strings because this feeds straight into JSON responses and templates
        where everything is text anyway.

    Notes
    -----
    **Every section is built to answer this.** Cheap for tabular sections, and
    it does mean the work is repeated when the download actually happens.

    See Also
    --------
    export_csv : Downloading one.
    """
    available = []
    for key, builder in _CSV_BUILDERS.items():
        rows = builder(report)
        if rows:
            available.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "rows": str(len(rows)),
                }
            )
    return available


def export_csv(report: dict[str, Any], section: str) -> tuple[str, str]:
    """Render one section as CSV text, with a filename to offer it under.

    The escape hatch from the studio. Every table shown on a board can be pulled
    out as CSV, because at some point everyone wants the numbers in a
    spreadsheet, and a dashboard that traps its data is a dashboard people work
    around.

    Parameters
    ----------
    report:
        The report as a dict.
    section:
        The section key, from :func:`list_csv_sections`.

    Returns
    -------
    tuple
        ``(filename, csv_text)``. The filename is prefixed ``buildml_eda_`` so
        downloads stay identifiable in a downloads folder.

    Raises
    ------
    KeyError
        If the section is unknown, or known but empty for this report. Empty is
        an error rather than a zero-row file, because a zero-row CSV looks like
        a finding: "no problems found": when it actually means the analysis
        never ran.

    Notes
    -----
    **Columns come from the first row.** Sections build uniform rows, so this
    holds; a section whose later rows carried extra keys would lose them.

    **The text is returned, not written.** The caller decides where it goes.

    See Also
    --------
    list_csv_sections : What is available.
    """
    if section not in _CSV_BUILDERS:
        raise KeyError(f"Unknown CSV section: {section}")
    rows = _CSV_BUILDERS[section](report)
    if not rows:
        raise KeyError(f"CSV section '{section}' has no rows for this report.")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return f"buildml_eda_{section}.csv", buffer.getvalue()


def export_pdf(
    report: dict[str, Any],
    *,
    view: str = "briefing",
    title: str = "BuildML EDA Studio",
    include_charts: bool = True,
) -> bytes:
    """Produce a PDF briefing: cover, contents, findings, charts, teaching notes.

    For the audience that wants a document: a reviewer, a regulator, an
    attachment to a ticket. Same content as the studio, laid out for reading
    linearly and printing.

    Charts become static PNGs of the same Plotly figures, via Kaleido. When
    Kaleido is unavailable, each chart is replaced by a placeholder and the rest
    of the briefing still renders; a document missing its figures is worth more
    than no document. The interactive versions stay in the local app and the
    offline HTML.

    Parameters
    ----------
    report:
        The report as a dict.
    view:
        Which layout. ``'briefing'`` is the full document.
    title:
        Cover title. Set it to something identifying: these get filed.
    include_charts:
        Embed rasterised charts. Turning this off is much faster and produces a
        far smaller file, which is often what you want for a findings-only
        summary.

    Returns
    -------
    bytes
        The PDF, for writing to disk or returning from an endpoint.

    Raises
    ------
    MissingExtraError
        If ReportLab is not installed. Install with
        ``pip install 'buildml[dashboard]'``.

    Notes
    -----
    **Chart rendering is the slow part**: a headless browser per figure, so
    seconds to tens of seconds. ``include_charts=False`` is near-instant.

    **A PDF is a snapshot.** It cannot be filtered or drilled into; pair it with
    the offline HTML when the reader may want to explore.

    **Associations are not causation**, and the briefing says so where it
    reports them. Worth remembering when a PDF is circulated to people who did
    not run the analysis.

    See Also
    --------
    buildml.dashboard.offline.export_studio_html : Interactive and offline.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:
        raise MissingExtraError("dashboard", "EDA PDF export") from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="BuildML",
    )
    styles = _pdf_styles(getSampleStyleSheet, ParagraphStyle, colors, TA_CENTER, TA_LEFT)

    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    findings = report.get("findings") or []
    studios = build_teaching_studios(report)
    domain_key = view if view in studios else ("briefing" if view == "briefing" else "cockpit")
    if domain_key not in studios:
        domain_key = "cockpit"
    studio = studios[domain_key]
    high_critical = sum(
        1 for item in findings if str(item.get("severity", "")).lower() in {"high", "critical"}
    )
    evidence_view = "briefing" if view == "briefing" else domain_key

    story: list[Any] = []

    # --- Cover / meta ---
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph(escape_xml(title), styles["BuildMLCover"]))
    story.append(Paragraph("EDA Teaching Studio briefing", styles["BuildMLCoverSub"]))
    story.append(Spacer(1, 8 * mm))
    story.append(
        _styled_table(
            [
                ["Field", "Value"],
                ["Briefing view", str(view)],
                ["Dataset rows", str(overview.get("n_rows"))],
                ["Analysis rows", str(overview.get("analysis_rows"))],
                ["Columns", str(overview.get("n_columns"))],
                [
                    "Eligible features",
                    str(len(overview.get("eligible_feature_columns") or [])),
                ],
                ["Engine", str(overview.get("engine"))],
                ["Completeness", _fmt_pct(quality.get("completeness_score"))],
                ["Missing cells", str(quality.get("missing_cell_count"))],
                ["High / critical findings", str(high_critical)],
            ],
            colors,
            mm,
            styles,
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            "This PDF is a structured offline briefing: cover metadata, contents, "
            "findings, domain evidence, static Plotly chart stills (PNG via kaleido), "
            "Teaching Studio excerpts, and methods/limitations. Interactive hover and "
            "zoom remain in the live Teaching Studio or offline Studio HTML. "
            "Associations do not establish causality; severity ranks workflow impact.",
            styles["BuildMLSmall"],
        )
    )
    story.append(PageBreak())

    # --- Contents ---
    story.append(Paragraph("Contents", styles["BuildMLH1"]))
    toc_items = [
        "1. Session overview",
        "2. Key findings",
        "3. Domain evidence",
        "4. Chart stills",
        f"5. Teaching Studio · {studio.get('title', domain_key)}",
        "6. Quality flags",
        "7. Methods and limitations",
    ]
    for item in toc_items:
        story.append(Paragraph(escape_xml(item), styles["BuildMLTOC"]))
    story.append(PageBreak())

    # --- 1. Overview ---
    story.append(Paragraph("1. Session overview", styles["BuildMLH1"]))
    story.append(
        Paragraph(
            escape_xml(
                f"Analyzers used {overview.get('analysis_rows')} of "
                f"{overview.get('n_rows')} rows across {overview.get('n_columns')} columns "
                f"(engine={overview.get('engine')}). Completeness is "
                f"{_fmt_pct(quality.get('completeness_score'))}."
            ),
            styles["BuildMLBody"],
        )
    )
    story.append(
        _styled_table(
            [
                ["Metric", "Value"],
                ["Rows", str(overview.get("n_rows"))],
                ["Analysis rows", str(overview.get("analysis_rows"))],
                ["Columns", str(overview.get("n_columns"))],
                [
                    "Eligible features",
                    str(len(overview.get("eligible_feature_columns") or [])),
                ],
                ["Engine", str(overview.get("engine"))],
                ["Completeness", _fmt_pct(quality.get("completeness_score"))],
                ["Missing cells", str(quality.get("missing_cell_count"))],
                ["High / critical findings", str(high_critical)],
            ],
            colors,
            mm,
            styles,
        )
    )

    # --- 2. Findings ---
    story.append(Paragraph("2. Key findings", styles["BuildMLH1"]))
    story.append(
        Paragraph(
            "Severity reflects likely workflow impact, not visual emphasis. "
            "Verify evidence keys before changing the pipeline.",
            styles["BuildMLSmall"],
        )
    )
    if not findings:
        story.append(Paragraph("No findings recorded.", styles["BuildMLBody"]))
    else:
        finding_rows = [["Severity", "Key", "Finding", "Columns"]]
        for item in findings[:30]:
            finding_rows.append(
                [
                    Paragraph(escape_xml(item.get("severity", "info")), styles["BuildMLCell"]),
                    Paragraph(escape_xml(item.get("key", "")), styles["BuildMLCell"]),
                    Paragraph(
                        escape_xml(f"{item.get('title')}: {item.get('detail')}")[:360],
                        styles["BuildMLCell"],
                    ),
                    Paragraph(
                        escape_xml(", ".join((item.get("affected_columns") or [])[:6])),
                        styles["BuildMLCell"],
                    ),
                ]
            )
        story.append(
            _styled_table(
                finding_rows,
                colors,
                mm,
                styles,
                col_widths=[22 * mm, 30 * mm, 88 * mm, 34 * mm],
            )
        )

    # --- 3. Evidence ---
    story.append(Paragraph("3. Domain evidence", styles["BuildMLH1"]))
    evidence_tables = _evidence_tables_for_view(report, evidence_view)
    if not evidence_tables:
        story.append(Paragraph("No compact evidence tables for this view.", styles["BuildMLBody"]))
    for heading, rows in evidence_tables:
        block: list[Any] = [
            Paragraph(escape_xml(heading), styles["BuildMLH2"]),
            _styled_table(rows, colors, mm, styles),
        ]
        story.append(KeepTogether(block))
        story.append(Spacer(1, 3 * mm))

    # --- 4. Charts ---
    if include_charts:
        story.append(PageBreak())
        story.extend(
            _chart_story_blocks(
                report,
                view=view,
                domain_key=domain_key,
                styles=styles,
                mm=mm,
                section_title="4. Chart stills (Teaching Studio)",
            )
        )

    # --- 5. Teaching Studio excerpts ---
    story.append(PageBreak())
    story.append(
        Paragraph(
            f"5. Teaching Studio · {escape_xml(studio['title'])}",
            styles["BuildMLH1"],
        )
    )
    story.extend(_teaching_story_blocks(studio, styles, colors, mm, KeepTogether, Paragraph))

    # --- 6. Quality flags ---
    story.append(Paragraph("6. Quality flags", styles["BuildMLH1"]))
    story.append(
        _styled_table(
            [
                ["Flag", "Columns"],
                [
                    "Constants",
                    ", ".join(map(str, (quality.get("constant_columns") or [])[:12])) or "none",
                ],
                [
                    "Id-like",
                    ", ".join(map(str, (quality.get("id_like_columns") or [])[:12])) or "none",
                ],
                [
                    "High cardinality",
                    ", ".join(map(str, (quality.get("high_cardinality_columns") or [])[:12]))
                    or "none",
                ],
                [
                    "Quasi-constant",
                    ", ".join(map(str, (quality.get("quasi_constant_columns") or [])[:12]))
                    or "none",
                ],
            ],
            colors,
            mm,
            styles,
        )
    )

    # --- 7. Methods / limitations ---
    story.append(Paragraph("7. Methods and limitations", styles["BuildMLH1"]))
    for line in (
        "EDA screens describe the analysis frame (possibly sampled). Rare categories "
        "and tails may differ in the full dataset or holdout partitions.",
        "Association measures (Pearson, Spearman, mutual information, Cramér's V) do "
        "not establish causality or leakage by themselves.",
        "Heuristic cutoffs (for example VIF above 5, IQR 1.5×, normality α=0.05) are "
        "review flags, not automatic deletion or transform rules.",
        "Chart stills are static PNG snapshots of Studio Plotly figures. Install "
        "buildml[dashboard] (plotly, reportlab, kaleido) for embedded stills.",
        "Full-dataset EDA after a split is descriptive context; train-fitted "
        "preprocessing and model comparison still require partition discipline.",
    ):
        story.append(Paragraph(escape_xml(f"• {line}"), styles["BuildMLBody"]))

    doc.build(story)
    return buffer.getvalue()


def _pdf_styles(getSampleStyleSheet, ParagraphStyle, colors, TA_CENTER, TA_LEFT) -> Any:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BuildMLCover",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5980a6"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLCoverSub",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5B6775"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            spaceAfter=6,
            textColor=colors.HexColor("#5980a6"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#5980a6"),
            spaceBefore=4,
            spaceAfter=8,
            borderPadding=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#1C2430"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLH3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=colors.HexColor("#334155"),
            spaceBefore=6,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLSmall",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#5B6775"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLCaption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#5B6775"),
            spaceBefore=2,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BuildMLTOC",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            spaceBefore=2,
            spaceAfter=2,
            textColor=colors.HexColor("#1C2430"),
        )
    )
    return styles


def _teaching_story_blocks(
    studio: dict[str, Any],
    styles: Any,
    colors: Any,
    mm: Any,
    KeepTogether: Any,
    Paragraph: Any,
) -> list[Any]:
    blocks: list[Any] = []
    blocks.append(Paragraph("What is analyzed", styles["BuildMLH2"]))
    for para in str(studio.get("definition") or "").split("\n\n"):
        if para.strip():
            blocks.append(Paragraph(escape_xml(para.strip()), styles["BuildMLBody"]))
    blocks.append(Paragraph("Why it matters", styles["BuildMLH2"]))
    for para in str(studio.get("why") or "").split("\n\n"):
        if para.strip():
            blocks.append(Paragraph(escape_xml(para.strip()), styles["BuildMLBody"]))
    blocks.append(Paragraph("How BuildML computes it", styles["BuildMLH2"]))
    for para in str(studio.get("how") or "").split("\n\n"):
        if para.strip():
            blocks.append(Paragraph(escape_xml(para.strip()), styles["BuildMLBody"]))

    if studio.get("interpretation"):
        blocks.append(Paragraph("Interpretation rules", styles["BuildMLH2"]))
        for item in studio["interpretation"][:8]:
            blocks.append(Paragraph(escape_xml(f"• {item}"), styles["BuildMLBody"]))
    if studio.get("thresholds"):
        blocks.append(Paragraph("Thresholds and review cues", styles["BuildMLH2"]))
        for item in studio["thresholds"][:8]:
            blocks.append(Paragraph(escape_xml(f"• {item}"), styles["BuildMLBody"]))
    if studio.get("assumptions"):
        blocks.append(Paragraph("Assumptions", styles["BuildMLH2"]))
        for item in studio["assumptions"][:6]:
            blocks.append(Paragraph(escape_xml(f"• {item}"), styles["BuildMLBody"]))
    if studio.get("pitfalls"):
        blocks.append(Paragraph("Pitfalls and anti-patterns", styles["BuildMLH2"]))
        for item in studio["pitfalls"][:8]:
            blocks.append(Paragraph(escape_xml(f"• {item}"), styles["BuildMLBody"]))

    worked = studio.get("worked_example") or {}
    blocks.append(Paragraph("Worked example (this dataset)", styles["BuildMLH2"]))
    blocks.append(
        Paragraph(escape_xml(worked.get("summary", "")), styles["BuildMLBody"]),
    )
    blocks.append(
        Paragraph(escape_xml(f"Reading: {worked.get('reading', '')}"), styles["BuildMLBody"]),
    )
    values = worked.get("values") or {}
    if isinstance(values, dict) and values:
        value_rows = [["Field", "Value"]]
        for key, value in list(values.items())[:14]:
            value_rows.append(
                [
                    Paragraph(escape_xml(key), styles["BuildMLCell"]),
                    Paragraph(escape_xml(_shorten(value, 240)), styles["BuildMLCell"]),
                ]
            )
        blocks.append(
            _styled_table(value_rows, colors, mm, styles, col_widths=[45 * mm, 129 * mm])
        )

    blocks.append(Paragraph("Impact on modeling", styles["BuildMLH2"]))
    for para in str(studio.get("modeling_impact") or "").split("\n\n"):
        if para.strip():
            blocks.append(Paragraph(escape_xml(para.strip()), styles["BuildMLBody"]))

    if studio.get("practice_checklist"):
        blocks.append(Paragraph("Practice checklist", styles["BuildMLH2"]))
        for item in studio["practice_checklist"][:8]:
            blocks.append(Paragraph(escape_xml(f"☐ {item}"), styles["BuildMLBody"]))
    if studio.get("mastery_notes"):
        blocks.append(Paragraph("Mastery notes", styles["BuildMLH2"]))
        for item in studio["mastery_notes"][:6]:
            blocks.append(Paragraph(escape_xml(f"• {item}"), styles["BuildMLBody"]))

    next_action = studio.get("next_action") or {}
    blocks.append(
        KeepTogether(
            [
                Paragraph("Next action", styles["BuildMLH2"]),
                Paragraph(
                    escape_xml(
                        f"{next_action.get('label')} · API: {next_action.get('api')}"
                    ),
                    styles["BuildMLBody"],
                ),
            ]
        )
    )
    return blocks


def escape_xml(text: Any) -> str:
    """Escape a value for ReportLab's markup, which is XML-ish and unforgiving.

    ReportLab paragraphs accept a small markup dialect: ``<b>``, ``<i>``,
    ``<font>``: parsed as XML. So a column literally named ``a<b`` produces a
    parse error and takes the whole PDF with it, and a value containing ``&``
    does the same.

    Separate from :func:`buildml.reporting.html.escape` because the targets
    differ: this escapes for ReportLab's parser, that one for a browser.

    Parameters
    ----------
    text:
        Anything. Stringified unless ``None``.

    Returns
    -------
    str
        With ``&``, ``<``, ``>``, and ``"`` replaced by entities. ``None``
        becomes empty, since a missing value should read as absent rather than
        as the word "None".

    Notes
    -----
    **``&`` is replaced first**, deliberately. Doing it later would re-escape
    the ampersands introduced by the other replacements, turning ``<`` into
    ``&amp;lt;``.

    **Apostrophes are left alone**, which is safe in element content and in the
    double-quoted attributes ReportLab uses.

    Examples
    --------
    >>> escape_xml("a < b & c")
    'a &lt; b &amp; c'
    >>> escape_xml(None)
    ''
    """
    value = "" if text is None else str(text)
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _chart_story_blocks(
    report: dict[str, Any],
    *,
    view: str,
    domain_key: str,
    styles: Any,
    mm: Any,
    section_title: str = "4. Chart stills (Teaching Studio)",
) -> list[Any]:
    """Build ReportLab flowables for static Plotly chart stills."""
    from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer

    from buildml.dashboard.charts import (
        build_chart_figures,
        charts_for_domain,
        render_chart_png,
    )

    # Prefer explicit view mapping (includes briefing), else domain boards.
    chart_ids = charts_for_domain(view) or charts_for_domain(domain_key) or charts_for_domain(
        "briefing"
    )
    max_charts = 8 if view == "briefing" else 6
    chart_ids = chart_ids[:max_charts]

    caption_style = styles["BuildMLCaption"] if "BuildMLCaption" in styles.byName else styles[
        "BuildMLSmall"
    ]
    heading_style = styles["BuildMLH1"] if "BuildMLH1" in styles.byName else styles["BuildMLH2"]
    blocks: list[Any] = [
        Paragraph(escape_xml(section_title), heading_style),
        Paragraph(
            "Static PNG stills of the same Plotly Teaching Studio figures. "
            "Captions name the chart; values come from the analysis frame.",
            styles["BuildMLSmall"],
        ),
    ]
    try:
        figures = build_chart_figures(report, theme="light")
    except Exception as exc:  # pragma: no cover - optional stack failures
        blocks.append(
            Paragraph(
                escape_xml(f"Chart stills unavailable: {exc}"),
                styles["BuildMLSmall"],
            )
        )
        return blocks

    embedded = 0
    embedded_ids: list[str] = []
    for index, chart_id in enumerate(chart_ids, start=1):
        fig = figures.get(chart_id)
        if fig is None:
            continue
        png = render_chart_png(fig, width=920, scale=2.0)
        if not png:
            continue
        image = Image(io.BytesIO(png))
        max_w = 174 * mm
        max_h = 92 * mm
        image._restrictSize(max_w, max_h)  # noqa: SLF001 - ReportLab sizing helper
        try:
            title = str(getattr(fig.layout.title, "text", "") or chart_id)
        except Exception:
            title = chart_id
        caption = f"Figure {index}. {title} (chart id: {chart_id})."
        blocks.append(
            KeepTogether(
                [
                    Paragraph(escape_xml(title or chart_id), styles["BuildMLH3"]),
                    image,
                    Paragraph(escape_xml(caption), caption_style),
                    Spacer(1, 4),
                ]
            )
        )
        embedded += 1
        embedded_ids.append(chart_id)

    if embedded == 0:
        blocks.append(
            Paragraph(
                "Chart stills could not be rasterized. Install the dashboard extra with "
                "kaleido (pip install 'buildml[dashboard]') and retry, or open the "
                "Teaching Studio for interactive Plotly boards.",
                styles["BuildMLSmall"],
            )
        )
    else:
        blocks.append(
            Paragraph(
                escape_xml(
                    f"Embedded {embedded} static PNG still(s): {', '.join(embedded_ids)}."
                ),
                styles["BuildMLSmall"],
            )
        )
    return blocks

def _styled_table(
    rows: list[list[Any]],
    colors: Any,
    mm: Any,
    styles: Any,
    *,
    col_widths: list[Any] | None = None,
) -> Any:
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=col_widths or [50 * mm, 124 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F5EF")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D7DEE7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _evidence_tables_for_view(
    report: dict[str, Any], view: str
) -> list[tuple[str, list[list[Any]]]]:
    tables: list[tuple[str, list[list[Any]]]] = []
    if view in {"features", "cockpit", "briefing", "visuals"}:
        rows = _rows_univariate_numeric(report)[:8]
        if rows:
            tables.append(
                (
                    "Numeric feature highlights",
                    [["column", "mean", "median", "skew", "appears_non_normal"]]
                    + [
                        [
                            str(r.get("column")),
                            _fmt_num(r.get("mean")),
                            _fmt_num(r.get("median")),
                            _fmt_num(r.get("skew")),
                            str(r.get("appears_non_normal")),
                        ]
                        for r in rows
                    ],
                )
            )
        cat_rows = _rows_univariate_categorical(report)[:8]
        if cat_rows:
            tables.append(
                (
                    "Categorical feature highlights",
                    [["column", "nunique", "entropy_bits", "mode", "rare_level_rate"]]
                    + [
                        [
                            str(r.get("column")),
                            str(r.get("nunique")),
                            _fmt_num(r.get("entropy_bits")),
                            str(r.get("mode")),
                            _fmt_num(r.get("rare_level_rate")),
                        ]
                        for r in cat_rows
                    ],
                )
            )
    if view in {"outliers", "cockpit", "briefing", "visuals"}:
        rows = _rows_outliers(report)[:10]
        if rows:
            tables.append(
                (
                    "Outlier IQR screens",
                    [
                        [
                            "column",
                            "iqr_outlier_rate",
                            "iqr_outlier_count",
                            "zscore_abs_gt_3_rate",
                        ]
                    ]
                    + [
                        [
                            str(r.get("column")),
                            _fmt_num(r.get("iqr_outlier_rate")),
                            str(r.get("iqr_outlier_count")),
                            _fmt_num(r.get("zscore_abs_gt_3_rate")),
                        ]
                        for r in rows
                    ],
                )
            )
    if view in {"relationships", "cockpit", "briefing", "visuals"}:
        mi_rows = sorted(
            _rows_mi(report),
            key=lambda row: float(row.get("mutual_information") or 0.0),
            reverse=True,
        )[:8]
        if mi_rows:
            tables.append(
                (
                    "Top mutual information vs target",
                    [["feature", "mutual_information"]]
                    + [
                        [str(r.get("feature")), _fmt_num(r.get("mutual_information"))]
                        for r in mi_rows
                    ],
                )
            )
    if view in {"multivariate", "cockpit", "briefing"}:
        vif_rows = sorted(
            _rows_vif(report),
            key=lambda row: float(row.get("vif") or 0.0),
            reverse=True,
        )[:8]
        if vif_rows:
            tables.append(
                (
                    "Top VIF rows",
                    [["column", "vif"]]
                    + [[str(r.get("column")), _fmt_num(r.get("vif"))] for r in vif_rows],
                )
            )
    if view in {"target", "cockpit", "briefing"}:
        target_rows = _rows_target(report)[:12]
        if target_rows:
            tables.append(
                (
                    "Target summary",
                    [["key", "value"]]
                    + [[str(r.get("key")), str(r.get("value"))] for r in target_rows],
                )
            )
    if view in {"quality", "cockpit", "briefing"}:
        missing = sorted(
            _rows_missing(report),
            key=lambda row: float(row.get("missing_rate") or 0.0),
            reverse=True,
        )[:10]
        if missing:
            tables.append(
                (
                    "Top missing rates",
                    [["column", "missing_rate", "missing_count"]]
                    + [
                        [
                            str(r.get("column")),
                            _fmt_num(r.get("missing_rate")),
                            str(r.get("missing_count")),
                        ]
                        for r in missing
                    ],
                )
            )
    return tables


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.3%}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_num(value: Any) -> str:
    try:
        if value is None:
            return ""
        number = float(value)
        if abs(number) >= 1000 or (0 < abs(number) < 0.001):
            return f"{number:.4g}"
        return f"{number:.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _shorten(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rows_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report.get("findings") or []:
        rows.append(
            {
                "key": item.get("key"),
                "title": item.get("title"),
                "severity": item.get("severity"),
                "detail": item.get("detail"),
                "affected_columns": ",".join(item.get("affected_columns") or []),
            }
        )
    return rows


def _rows_recommendations(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report.get("recommendation_details") or []:
        action = item.get("action") or {}
        rows.append(
            {
                "key": item.get("key"),
                "title": item.get("title"),
                "priority": item.get("priority"),
                "rationale": item.get("rationale"),
                "based_on": ",".join(item.get("based_on") or []),
                "action_operation": action.get("operation"),
                "caveats": " | ".join(item.get("caveats") or []),
            }
        )
    return rows


def _rows_missing(report: dict[str, Any]) -> list[dict[str, Any]]:
    rates = (report.get("quality") or {}).get("missing_rate_by_column") or {}
    counts = (report.get("quality") or {}).get("missing_by_column") or {}
    return [
        {
            "column": column,
            "missing_rate": rates.get(column),
            "missing_count": counts.get(column),
        }
        for column in rates
    ]


def _rows_quality_flags(report: dict[str, Any]) -> list[dict[str, Any]]:
    quality = report.get("quality") or {}
    rows = []
    for kind in (
        "constant_columns",
        "quasi_constant_columns",
        "id_like_columns",
        "high_cardinality_columns",
    ):
        for column in quality.get(kind) or []:
            rows.append({"flag": kind, "column": column})
    return rows


def _univariate_per_column(report: dict[str, Any]) -> dict[str, Any]:
    uni = report.get("univariate") or {}
    per_column = uni.get("per_column")
    if isinstance(per_column, dict):
        return per_column
    return {}


def _rows_univariate_numeric(report: dict[str, Any]) -> list[dict[str, Any]]:
    per_column = _univariate_per_column(report)
    if per_column:
        rows = []
        for column, stats in per_column.items():
            if not isinstance(stats, dict):
                continue
            if str(stats.get("kind", "numeric")) != "numeric":
                continue
            flat = {
                key: value for key, value in stats.items() if not isinstance(value, (dict, list))
            }
            rows.append({"column": column, **flat})
        return rows

    numeric = (report.get("univariate") or {}).get("numeric") or {}
    if isinstance(numeric, list):
        return [dict(row) for row in numeric if isinstance(row, dict)]
    rows = []
    for column, stats in numeric.items():
        if isinstance(stats, dict):
            rows.append({"column": column, **stats})
    return rows


def _rows_univariate_categorical(report: dict[str, Any]) -> list[dict[str, Any]]:
    per_column = _univariate_per_column(report)
    if per_column:
        rows = []
        for column, stats in per_column.items():
            if not isinstance(stats, dict):
                continue
            if str(stats.get("kind")) != "categorical":
                continue
            flat = {
                key: value for key, value in stats.items() if not isinstance(value, (dict, list))
            }
            top = stats.get("top_values")
            if isinstance(top, dict) and top:
                mode_key = next(iter(top))
                flat.setdefault("mode", mode_key)
                flat["top_value"] = mode_key
                flat["top_count"] = top[mode_key]
            rows.append({"column": column, **flat})
        return rows

    categorical = (report.get("univariate") or {}).get("categorical") or {}
    if isinstance(categorical, list):
        return [dict(row) for row in categorical if isinstance(row, dict)]
    rows = []
    for column, stats in categorical.items():
        if isinstance(stats, dict):
            flat = {key: stats[key] for key in stats if not isinstance(stats[key], (dict, list))}
            rows.append({"column": column, **flat})
    return rows


def _matrix_pair_rows(
    matrix: Any,
    *,
    value_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(matrix, dict):
        return []
    rows: list[dict[str, Any]] = []
    if "columns" in matrix and "matrix" in matrix:
        columns = list(matrix["columns"])
        for i, row_name in enumerate(columns):
            for j, col_name in enumerate(columns):
                if j <= i:
                    continue
                try:
                    rows.append(
                        {
                            "feature_a": row_name,
                            "feature_b": col_name,
                            value_key: matrix["matrix"][i][j],
                        }
                    )
                except (IndexError, TypeError, KeyError):
                    continue
        return rows
    for a, row in matrix.items():
        if not isinstance(row, dict):
            continue
        for b, value in row.items():
            if str(a) >= str(b):
                continue
            rows.append({"feature_a": a, "feature_b": b, value_key: value})
    return rows


def _rows_correlations(report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = (report.get("bivariate") or {}).get("pearson") or (report.get("bivariate") or {}).get(
        "correlation_pearson"
    )
    return _matrix_pair_rows(matrix, value_key="pearson_r")


def _rows_spearman(report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = (report.get("bivariate") or {}).get("spearman") or (report.get("bivariate") or {}).get(
        "correlation_spearman"
    )
    return _matrix_pair_rows(matrix, value_key="spearman_rho")


def _rows_cramers(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (report.get("bivariate") or {}).get("categorical_pairs") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "feature_a": item.get("a"),
                "feature_b": item.get("b"),
                "cramers_v": item.get("cramers_v"),
            }
        )
    return rows


def _rows_mi(report: dict[str, Any]) -> list[dict[str, Any]]:
    mi = (report.get("bivariate") or {}).get("mutual_information_vs_target") or {}
    rows = []
    for column, value in mi.items():
        score = value.get("score") if isinstance(value, dict) else value
        rows.append({"feature": column, "mutual_information": score})
    return rows


def _rows_vif(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (report.get("multivariate") or {}).get("vif") or []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _rows_pca(report: dict[str, Any]) -> list[dict[str, Any]]:
    pca = (report.get("multivariate") or {}).get("pca") or {}
    ratios = pca.get("explained_variance_ratio") or []
    return [
        {"component": f"PC{index + 1}", "explained_variance_ratio": value}
        for index, value in enumerate(ratios)
    ]


def _rows_target(report: dict[str, Any]) -> list[dict[str, Any]]:
    target = report.get("target") or {}
    rows = [
        {"key": key, "value": value}
        for key, value in target.items()
        if not isinstance(value, (dict, list))
    ]
    summary = target.get("summary") if isinstance(target.get("summary"), dict) else {}
    balance = (
        target.get("class_balance")
        or target.get("value_counts")
        or (summary.get("class_counts") if summary else None)
        or {}
    )
    if isinstance(balance, dict):
        for key, value in balance.items():
            rows.append({"key": f"class:{key}", "value": value})
    return rows


def _rows_drift(report: dict[str, Any]) -> list[dict[str, Any]]:
    drift = report.get("drift") or {}
    rows: list[dict[str, Any]] = []
    flagged = set(flagged_column_names(drift.get("flagged_columns")))
    for item in drift.get("flagged_columns") or []:
        if isinstance(item, dict):
            rows.append(dict(item))
        else:
            rows.append({"column": item, "flagged": True})
    for key in ("numeric_drift", "categorical_drift"):
        for item in drift.get(key) or []:
            if isinstance(item, dict):
                rows.append(dict(item))
    scores = drift.get("scores") or drift.get("column_scores") or {}
    if isinstance(scores, dict):
        for column, score in scores.items():
            rows.append(
                {
                    "column": column,
                    "score": score,
                    "flagged": column in flagged,
                }
            )
    return rows


def _rows_outliers(report: dict[str, Any]) -> list[dict[str, Any]]:
    outliers = report.get("outliers") or {}
    per_column = outliers.get("per_column") or outliers.get("univariate") or {}
    rows: list[dict[str, Any]] = []
    if isinstance(per_column, dict):
        for column, value in per_column.items():
            if isinstance(value, dict):
                flat = {key: val for key, val in value.items() if not isinstance(val, (dict, list))}
                bounds = value.get("iqr_bounds")
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    flat["iqr_lower"] = bounds[0]
                    flat["iqr_upper"] = bounds[1]
                rows.append({"column": column, **flat})
            else:
                rows.append({"column": column, "value": value})
    multi = outliers.get("multivariate") or {}
    if isinstance(multi, dict) and multi:
        rows.append(
            {
                "column": "__multivariate__",
                "method": multi.get("method"),
                "anomaly_rate": multi.get("anomaly_rate"),
                "anomaly_count": multi.get("anomaly_count"),
                "n_rows_scored": multi.get("n_rows_scored"),
            }
        )
    return rows


def _rows_plan(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(report.get("adaptive_plan") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "index": index,
                "kind": item.get("kind"),
                "title": item.get("title"),
                "column": item.get("column"),
                "columns": ",".join(map(str, item.get("columns") or [])),
            }
        )
    return rows


def _rows_concepts(_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": note.key,
            "title": note.title,
            "summary": note.summary,
            "related": ",".join(note.related_concepts),
        }
        for note in CONCEPT_NOTES.values()
    ]


def _rows_roles(report: dict[str, Any]) -> list[dict[str, Any]]:
    overview = report.get("overview") or {}
    reasons = overview.get("feature_exclusion_reasons") or {}
    rows = []
    for column in overview.get("eligible_feature_columns") or []:
        rows.append({"column": column, "status": "eligible", "reasons": ""})
    for column, reason_list in reasons.items():
        rows.append(
            {
                "column": column,
                "status": "excluded",
                "reasons": " | ".join(map(str, reason_list)),
            }
        )
    return rows
