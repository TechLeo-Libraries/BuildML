"""Build a report that opens anywhere, forever, with no network.

Every ``export_html`` across BuildML comes through here. The shared vocabulary :
cards, badges, tables, the five-part reading frame: is what makes an EDA report
and a diagnostics report look like the same product rather than two tools that
happen to ship together.

The constraint that shapes everything is self-containment. A report is one HTML
file with its CSS, its JavaScript, and its images inlined. No CDN, no template
engine, no build step. A file that fetches a stylesheet is a file that renders
as unstyled text the day the CDN moves, or the moment it is opened on a machine
behind a firewall: which is exactly where a model report tends to be read.

See Also
--------
buildml.reporting.html : The components and the document shell.
"""

from buildml.reporting.html import (
    REPORT_CSS,
    REPORT_JS,
    ReportSection,
    element_id,
    encode_asset,
    escape,
    render_badge,
    render_card,
    render_list,
    render_navigation,
    render_reading_frame,
    render_report,
    render_table,
    severity_tone,
    write_report,
)

__all__ = [
    "REPORT_CSS",
    "REPORT_JS",
    "ReportSection",
    "element_id",
    "encode_asset",
    "escape",
    "render_badge",
    "render_card",
    "render_list",
    "render_navigation",
    "render_reading_frame",
    "render_report",
    "render_table",
    "severity_tone",
    "write_report",
]

