"""Shared, offline reporting primitives."""

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

