"""Walkthrough / teaching hooks for fairness."""

from __future__ import annotations

from typing import Any

from buildml.explain.capability_status import attach_capability_matrix


def fairness_status(session: Any | None = None) -> dict[str, Any]:
    """Domain status payload for fairness reporting."""
    report = getattr(session, "_fairness_report", None) if session is not None else None
    status: dict[str, Any] = {
        "enabled": report is not None,
        "has_report": report is not None,
        "disclosures": [
            "evaluate_fairness is holdout-only observational disparity reporting.",
        ],
    }
    if report is not None and hasattr(report, "to_dict"):
        status["last_report"] = report.to_dict()
    return attach_capability_matrix(status, "fairness_capability_matrix")


def fairness_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing walkthrough status."""
    return fairness_status(session)
