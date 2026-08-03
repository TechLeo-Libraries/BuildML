# ruff: noqa: E501
"""HTML components with no dependencies, and the document shell around them.

String concatenation, not a template engine. That sounds like a shortcut and is
a deliberate choice: a reporting layer that pulls in Jinja makes the dependency
mandatory for anyone who wants an HTML export, and the markup here is simple
enough that the templates would not earn their cost.

The choice does put the burden of escaping on this module rather than on a
framework, so :func:`escape` is applied at every point where a caller's value
reaches the output. A dataset column named ``<script>`` is a real thing that
happens, and a report that executes it is a report that cannot be shared.

Three ideas run through the components. Everything is escaped unless a field is
explicitly documented as trusted HTML. Everything is bounded, so a frame with a
hundred thousand rows produces a readable report rather than a browser that
stops responding. And everything carries the accessibility attributes that make
a report usable with a screen reader or a keyboard: captions, scopes, landmark
roles, skip links: because reports get read by people who did not run the
analysis.

See Also
--------
buildml.reporting : The public surface re-exported from here.
"""

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
    """Make any value safe to drop into markup, whatever it turns out to be.

    The single point where caller data becomes HTML. Applied everywhere in this
    module, on the assumption that any value reaching a report might contain
    markup: a column named ``<b>total</b>``, a category value with an
    apostrophe, an error message quoting user input.

    Non-strings are stringified rather than rejected, since report values are
    routinely numbers, enums, and ``None``. ``None`` becomes the empty string
    rather than the word "None", because a missing value should read as absent.

    Parameters
    ----------
    value:
        Anything. Converted with ``str`` unless it is ``None``.
    quote:
        Also escape quotes, which is required inside an attribute and harmless
        in body text. Defaults to on, so forgetting it cannot open a hole.

    Returns
    -------
    str
        The escaped text.

    Notes
    -----
    **This is safe for text and attributes, and not for anything else.** Values
    placed inside a ``<script>`` block, a ``<style>`` block, or a URL need their
    own encoding, and none of the components here put caller data in those
    positions.

    Examples
    --------
    >>> escape("<script>alert(1)</script>")
    '&lt;script&gt;alert(1)&lt;/script&gt;'
    >>> escape(None)
    ''
    >>> escape(0.5)
    '0.5'
    >>> escape('a "quoted" value')
    'a &quot;quoted&quot; value'
    >>> escape('a "quoted" value', quote=False)
    'a "quoted" value'
    """
    return html.escape("" if value is None else str(value), quote=quote)


def element_id(value: object, *, prefix: str = "section") -> str:
    """Turn a human label into an id that is safe in a URL fragment.

    Section keys come from analysis names: ``"Missing Values"``, ``"ROC / PR
    curves"``: and those cannot be used directly as ids, because a fragment
    link containing a space or a slash breaks navigation.

    Everything outside lowercase letters, digits, hyphens, and underscores
    collapses to a hyphen, and leading and trailing separators are trimmed. The
    mapping is deterministic, so the same key produces the same id every time :
    which is what lets a table of contents link to a section rendered
    separately.

    Parameters
    ----------
    value:
        The label to convert.
    prefix:
        The fallback when nothing survives the transformation, for a key that is
        entirely punctuation or whitespace. An empty id would produce a link to
        nowhere.

    Returns
    -------
    str
        A lowercase, hyphen-separated identifier.

    Notes
    -----
    **The mapping is not injective.** ``"ROC/PR"`` and ``"ROC PR"`` both become
    ``'roc-pr'``, so distinct sections can collide.
    :func:`render_report` checks for that and refuses rather than producing a
    document where one link reaches the wrong section.

    Examples
    --------
    >>> element_id("Missing Values")
    'missing-values'
    >>> element_id("ROC / PR curves")
    'roc-pr-curves'
    >>> element_id("   ")
    'section'
    >>> element_id("!!!", prefix="chart")
    'chart'
    """
    identifier = _ID_PATTERN.sub("-", str(value).strip().lower()).strip("-_")
    return identifier or prefix


def encode_asset(
    source: str | Path | bytes,
    *,
    media_type: str | None = None,
) -> str:
    """Inline an image or file into the document, so nothing is fetched later.

    This is what makes a report portable. A ``<img src="figure.png">`` is a
    promise that the file will still be beside the HTML when someone opens it,
    and reports get emailed, copied to shared drives, and attached to tickets :
    the promise does not survive. Base64 into a data URI, and the image is part
    of the document.

    Parameters
    ----------
    source:
        A path to read, or the bytes themselves when the content was generated
        in memory: a Matplotlib figure saved to a buffer, for instance.
    media_type:
        The MIME type. Guessed from the file extension when a path was given;
        required for bytes if the browser is to render rather than download
        them.

    Returns
    -------
    str
        A ``data:`` URI, usable directly as an ``src`` or ``href``.

    Raises
    ------
    FileNotFoundError
        If a path is given that does not exist.
    OSError
        If the file cannot be read.

    Notes
    -----
    **Base64 costs about a third more bytes than the original**, and the whole
    thing lands in a single HTML file. A report with fifty high-resolution
    figures becomes tens of megabytes, which browsers handle but email gateways
    often do not. Prefer fewer, smaller figures over many large ones.

    **An unguessable type falls back to ``application/octet-stream``**, which
    browsers download rather than display. If a figure appears as a download
    prompt instead of an image, pass ``media_type`` explicitly.

    Examples
    --------
    >>> uri = encode_asset(b"\\x89PNG...", media_type="image/png")
    >>> uri.startswith("data:image/png;base64,")
    True
    """
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
    """Render a small coloured label for a status or count.

    Used for the short signals that would be noise as full sentences: a
    severity, a pass or fail, a count of affected columns.

    Parameters
    ----------
    label:
        The text. Escaped.
    tone:
        One of ``'neutral'``, ``'info'``, ``'good'``, ``'warn'``, or
        ``'danger'``, which selects the colour.

    Returns
    -------
    str
        A ``<span>`` carrying the tone class.

    Notes
    -----
    **An unrecognised tone silently becomes ``'neutral'``.** The alternative is
    a report that fails to render over a typo in a colour name, which is a bad
    trade: but it does mean a misspelled tone shows up as a missing colour
    rather than an error.

    **Colour is never the only signal.** The label text carries the meaning, so
    the badge still reads correctly in greyscale, in print, and to a screen
    reader.

    Examples
    --------
    >>> render_badge("high", tone="danger")
    '<span class="bml-badge bml-badge--danger">high</span>'
    >>> render_badge("ok", tone="nonsense")
    '<span class="bml-badge bml-badge--neutral">ok</span>'

    See Also
    --------
    severity_tone : Choosing the tone from a severity label.
    """
    allowed_tone = tone if tone in {"neutral", "info", "good", "warn", "danger"} else "neutral"
    return f'<span class="bml-badge bml-badge--{allowed_tone}">{escape(label)}</span>'


def severity_tone(severity: object) -> str:
    """Translate a finding's severity into the tone that renders it.

    Findings across BuildML carry severities like ``'critical'`` or
    ``'medium'``; components take tones like ``'danger'`` or ``'warn'``. Mapping
    in one place keeps a critical finding the same colour in every report,
    rather than each caller picking its own.

    Critical and high both map to danger, because the visual distinction between
    two shades of alarming is not one a reader reliably picks up: the severity
    text carries that difference.

    Parameters
    ----------
    severity:
        The label. Lowercased before matching, so casing does not matter.

    Returns
    -------
    str
        A tone token for :func:`render_badge` or :func:`render_card`.

    Notes
    -----
    **Unknown severities become ``'neutral'``**, so a new severity introduced
    elsewhere renders uncoloured rather than breaking the report: and looks
    plain enough to notice.

    Examples
    --------
    >>> severity_tone("critical"), severity_tone("HIGH")
    ('danger', 'danger')
    >>> severity_tone("medium"), severity_tone("low")
    ('warn', 'info')
    >>> severity_tone("unheard-of")
    'neutral'

    See Also
    --------
    render_badge : The usual consumer.
    """
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
    """Render the five questions every BuildML result should answer.

    A number without a frame around it invites the reader to supply their own
    interpretation, which is where most misreadings start. The frame is a fixed
    structure: what was examined, what came out, why it matters, what it cannot
    tell you, and what to do next: and its value comes from being the same
    everywhere. A reader learns the shape once and then knows where to look in
    every report.

    The ``limits`` slot is the one that earns the structure. It is the part
    authors omit when writing prose freely, and the part a reader most needs in
    order not to over-claim.

    Parameters
    ----------
    examined:
        What the analysis looked at: which data, which partition, which
        columns.
    observed:
        What came out, stated as a result rather than an interpretation.
    why:
        Why the result matters for a decision.
    limits:
        What this cannot tell you. Assumptions, sample sizes, checks that were
        skipped.
    next_step:
        The concrete action the result suggests.

    Returns
    -------
    str
        A ``<dl>`` with the five terms. All values escaped.

    Notes
    -----
    **All five are required, deliberately.** An optional ``limits`` would be
    omitted precisely when it is most needed. If a slot genuinely has nothing to
    say, saying so is more useful than leaving it out.

    See Also
    --------
    render_card : For narrower observations that do not need the full frame.
    """
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
    """Render a titled block of text with a coloured edge for its tone.

    The workhorse for a single observation: a finding, a recommendation, a
    caveat. Both title and body are escaped, so this is the safe choice for
    anything containing values from the data.

    Parameters
    ----------
    title:
        The heading text.
    body:
        The paragraph text. Plain text only; markup would be escaped and shown
        literally.
    heading_level:
        Which heading tag to use, clamped to 2–6. Getting this right matters
        more than it looks: screen readers navigate by heading structure, and a
        card nested under a section heading should be one level deeper, not
        whatever looks right visually.
    tone:
        One of ``'neutral'``, ``'info'``, ``'good'``, ``'warn'``, ``'danger'``.

    Returns
    -------
    str
        An ``<article>`` with a heading and a paragraph.

    Notes
    -----
    **The level is clamped rather than validated**, so a caller computing
    ``heading_level=section_depth + 1`` cannot produce an ``<h7>`` or an
    ``<h1>`` that competes with the report title.

    **The body is one paragraph.** For lists use :func:`render_list`, and for a
    full result use :func:`render_reading_frame`.

    Examples
    --------
    >>> render_card("Class imbalance", "Positives are 3% of rows.", tone="warn")
    '<article class="bml-card bml-card--warn"><h3>Class imbalance</h3><p>Positives are 3% of rows.</p></article>'

    See Also
    --------
    severity_tone : Deriving the tone from a finding.
    """
    level = min(6, max(2, heading_level))
    allowed_tone = tone if tone in {"neutral", "info", "good", "warn", "danger"} else "neutral"
    return (
        f'<article class="bml-card bml-card--{allowed_tone}">'
        f"<h{level}>{escape(title)}</h{level}>"
        f"<p>{escape(body)}</p>"
        "</article>"
    )


def render_list(items: Iterable[object], *, ordered: bool = False) -> str:
    """Render items as a list, escaping each one.

    Use an ordered list when the sequence carries meaning: ranked features,
    steps to follow: and an unordered one otherwise. The distinction is not
    cosmetic: a screen reader announces an ordered list's positions, which tells
    the listener the order matters.

    Parameters
    ----------
    items:
        The entries. Each is stringified and escaped, so numbers, enums, and
        ``None`` are all acceptable.
    ordered:
        Render as ``<ol>`` rather than ``<ul>``.

    Returns
    -------
    str
        A ``<ul>`` or ``<ol>``. An empty iterable yields an empty list element,
        which renders as nothing.

    Notes
    -----
    **There is no length bound here.** A list built from an unbounded source :
    every distinct category, say: should be truncated by the caller, which
    knows what "the rest" means well enough to say so.

    Examples
    --------
    >>> render_list(["age", "income"])
    '<ul><li>age</li><li>income</li></ul>'
    >>> render_list(["first", "second"], ordered=True)
    '<ol><li>first</li><li>second</li></ol>'
    >>> render_list([])
    '<ul></ul>'
    """
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
    """Render rows as a table, bounded in both directions and stated when cut.

    The bounds are the point. Rendering a frame straight to HTML works fine on
    the developer's thousand-row sample and produces an unopenable file on the
    real data. Defaults of 500 rows and 50 columns keep a report readable, and
    when anything is dropped a note above the table says how much: a silently
    truncated table is worse than no table, because it looks complete.

    Cells are formatted before escaping, so floats get six significant figures
    rather than seventeen, collections become comma-separated text, and ``None``
    becomes empty.

    Parameters
    ----------
    rows:
        The data, as mappings. Missing keys render as empty cells, so rows need
        not be uniform.
    columns:
        Which columns to show, in order. Defaults to first-seen order across all
        rows, which keeps output stable for a given input.
    caption:
        A caption, rendered as ``<caption>`` and reused as the scroll region's
        accessible label. Worth supplying: it is what tells a screen-reader
        user what the table contains before they enter it.
    empty_message:
        Shown instead of the table when there are no rows. A sentence saying
        nothing was found is more informative than an empty grid.
    max_rows:
        Row budget, or ``None`` for no limit. Removing the limit on data of
        unknown size is how a report becomes unopenable.
    max_columns:
        Column budget, or ``None`` for no limit.

    Returns
    -------
    str
        A scrollable, labelled table, preceded by a truncation note when the
        budget was applied. Or the empty message when there are no rows.

    Raises
    ------
    ValueError
        If either budget is set to zero or a negative number. ``None`` is the
        way to say "no limit"; zero would render a table with no content and no
        indication why.

    Notes
    -----
    **Truncation takes the first N, not a sample.** Sort the rows so that the
    interesting ones come first: largest effect, worst error: because the tail
    is what disappears.

    **Floats use six significant figures.** Enough to distinguish values,
    without implying precision a metric on a finite sample does not have.

    Examples
    --------
    >>> rows = [{"feature": "age", "importance": 0.42}]
    >>> "0.42" in render_table(rows, caption="Top features")
    True
    >>> render_table([])
    '<p class="bml-empty">No rows to display.</p>'
    >>> render_table([], empty_message="No leakage detected.")
    '<p class="bml-empty">No leakage detected.</p>'
    """
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
    """One section of a report: a heading, a body, and a link target.

    The only place in this module where a caller supplies raw HTML. That is
    necessary: a section body is assembled from the render helpers, and
    escaping it again would show the markup as text: and it means the
    responsibility for safety moves to the caller.

    Attributes
    ----------
    key:
        A stable identifier, converted to the id and fragment link. Keep it
        constant across runs so a bookmark into a report keeps working.
    title:
        The heading text. Escaped.
    body_html:
        **Trusted HTML.** Built from :func:`render_table`, :func:`render_card`,
        and the rest, which escape their own inputs. Never interpolate a raw
        value here.
    summary:
        An optional sentence under the heading, saying what the section is for.
        Escaped.

    Notes
    -----
    **Assemble ``body_html`` from the render helpers only.** They escape their
    inputs; an f-string does not. This is the one place in the reporting layer
    where a mistake can put caller data into the document unescaped.

    **Keys must produce distinct ids.** :func:`render_report` rejects
    collisions, which are easy to create accidentally since ``"ROC/PR"`` and
    ``"ROC PR"`` reduce to the same thing.

    See Also
    --------
    render_report : Assembling sections into a document.
    """

    key: str
    title: str
    body_html: str
    summary: str | None = None


def render_navigation(sections: Sequence[ReportSection]) -> str:
    """Render a table of contents linking to each section.

    Called by :func:`render_report`, and available separately for a custom
    shell. The ``<nav>`` element with a label is what lets assistive technology
    treat this as a navigation landmark and jump straight to it, rather than
    reading it as an ordinary list of links.

    Parameters
    ----------
    sections:
        The sections, in the order they appear. Each key becomes a fragment link
        to that section's id.

    Returns
    -------
    str
        A labelled ``<nav>`` containing the links.

    Notes
    -----
    **Links resolve only if the sections are rendered in the same document.**
    Building navigation for one set of sections and a body from another produces
    links that go nowhere.

    See Also
    --------
    element_id : How keys become fragment targets.
    """
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
    """Assemble sections into one HTML file that depends on nothing.

    The document shell. Everything is inlined: the stylesheet, the JavaScript,
    and whatever assets the sections encoded: so the result opens on a machine
    with no network, no BuildML, and no Python.

    What the shell adds beyond the sections: a skip link and landmark roles for
    keyboard and screen-reader navigation, a table of contents, a live section
    filter, a dark-theme toggle, per-table sorting and filtering, and print
    styles that drop the interactive furniture. All of it degrades to readable
    static HTML if JavaScript is disabled.

    Parameters
    ----------
    title:
        The document title, used in the ``<title>`` and the page heading.
        Escaped.
    sections:
        The content, in order. At least one is required.
    subtitle:
        A sentence under the title. The place for run context: which dataset,
        which date, which model version.
    metadata:
        Key-value pairs shown in the header. Rows, columns, target, split
        strategy: the facts a reader needs before the first section.
    lang:
        The document language, used by screen readers to pick pronunciation.

    Returns
    -------
    str
        A complete HTML document, ready to write.

    Raises
    ------
    ValueError
        If ``sections`` is empty, or if two section keys reduce to the same id.
        The second is refused rather than tolerated because a duplicate id makes
        one navigation link silently reach the wrong section, which is the kind
        of fault nobody notices in review.

    Notes
    -----
    **Section bodies are inserted unescaped.** They are trusted HTML by
    contract; see :class:`ReportSection`.

    **The whole report is one string in memory** before it is written. Large
    inlined figures push memory up accordingly.

    Examples
    --------
    A minimal report::

        sections = [
            ReportSection(
                key="overview",
                title="Overview",
                summary="What this dataset contains.",
                body_html=render_table(rows, caption="Column summary"),
            )
        ]
        html_text = render_report(
            "Churn dataset review",
            sections,
            subtitle="Snapshot taken 2026-08-01",
            metadata={"Rows": 48_120, "Target": "churned"},
        )

    See Also
    --------
    write_report : Rendering and writing in one call.
    """
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
    """Render a report and write it to disk, creating the directory if needed.

    The usual entry point, behind every ``export_html`` in BuildML. Parent
    directories are created, so a path into a fresh ``artifacts/`` tree works
    without a preceding ``mkdir``.

    UTF-8 explicitly, not the platform default. On Windows that default is
    still often cp1252, which cannot encode the arrows and symbols the report
    uses: and the failure appears as an encoding error partway through writing,
    on one machine and not another.

    Parameters
    ----------
    path:
        Where to write. Overwritten if it exists.
    title:
        The report title.
    sections:
        The content, in order.
    **kwargs:
        Passed to :func:`render_report`: ``subtitle``, ``metadata``, ``lang``.

    Returns
    -------
    Path
        The file written, so the location can be logged or opened.

    Raises
    ------
    ValueError
        If there are no sections, or two section ids collide.
    OSError
        If the directory cannot be created or the file written.

    Examples
    --------
    ::

        path = write_report(
            "artifacts/reports/eda.html",
            "Churn dataset review",
            sections,
            metadata={"Rows": 48_120},
        )

    See Also
    --------
    render_report : Getting the HTML without writing it.
    """
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

