"""Compatibility narrative view backed by structured EDA findings."""

from __future__ import annotations

from typing import Any

from buildml.eda.findings import build_findings, narrative_view


def build_narrative(report_sections: dict[str, Any]) -> list[str]:
    """Return the legacy string list derived from evidence-linked findings."""
    return narrative_view(build_findings(report_sections))
