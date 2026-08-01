"""Structured preprocess operation results (evidence-linked)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.explain.schemas import Evidence, Finding, Recommendation, _json_value


@dataclass(slots=True)
class PreprocessResult:
    """Evidence-linked result for a train-fitted preprocess operation.

    Session methods still return ``Session`` for fluent chaining. The structured
    report is stored on the session (for example ``last_preprocess``) and written
    into history ``result_summary``.
    """

    operation: str
    plan: dict[str, Any]
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "plan": dict(self.plan),
            "evidence": [item.to_dict() for item in self.evidence],
            "findings": [item.to_dict() for item in self.findings],
            "interpretation": list(self.interpretation),
            "limitations": list(self.limitations),
            "recommendations": [item.to_dict() for item in self.recommendations],
            "methods": list(self.methods),
            "warnings": list(self.warnings),
            "summary": _json_value(
                {
                    "n_findings": len(self.findings),
                    "n_recommendations": len(self.recommendations),
                    "plan_keys": sorted(self.plan),
                }
            ),
        }
