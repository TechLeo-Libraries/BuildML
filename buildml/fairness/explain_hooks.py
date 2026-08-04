"""Walkthrough / teaching hooks for fairness."""

from __future__ import annotations

from typing import Any

from buildml.explain.capability_status import attach_capability_matrix


def fairness_status(session: Any | None = None) -> dict[str, Any]:
    """Domain status payload for fairness reporting."""
    report = getattr(session, "_fairness_report", None) if session is not None else None
    mitigation = (
        getattr(session, "_fairness_mitigation_suggestion", None)
        if session is not None
        else None
    )
    status: dict[str, Any] = {
        "enabled": report is not None,
        "has_report": report is not None,
        "maturity": "observational_analysis",
        "depth": "high",
        "disclosures": [
            "evaluate_fairness is holdout-only observational disparity reporting.",
            "positive_label is hard-validated against observed labels.",
            "Intersectional sensitive columns and stability bands are supported.",
            "Not a legal audit; mitigation helpers are opt-in and non-certifying.",
        ],
    }
    if report is not None and hasattr(report, "to_dict"):
        status["last_report"] = report.to_dict()
        status["demographic_parity_difference"] = getattr(
            report, "demographic_parity_difference", None
        )
        status["intersectional"] = bool(getattr(report, "intersectional", False))
        status["has_stability"] = getattr(report, "stability", None) is not None
        status["n_groups"] = len(getattr(report, "groups", ()) or ())
    if mitigation is not None and hasattr(mitigation, "to_dict"):
        status["last_mitigation_suggestion"] = mitigation.to_dict()
    return attach_capability_matrix(status, "fairness_capability_matrix")


def fairness_status_for_session(session: Any) -> dict[str, Any]:
    """Session-facing walkthrough status."""
    return fairness_status(session)
