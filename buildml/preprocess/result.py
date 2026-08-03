"""The narrated record of what a preprocessing step did and why.

Preprocessing usually happens silently: you call a method, the frame changes,
and you find out months later that a column was dropped for a reason nobody
recorded. Every preprocessing operation in BuildML returns one of these
alongside the transformed data, so the reasoning survives.

The structure separates what was observed from what it means and what to do
about it — evidence, findings, interpretation, limitations, recommendations.
That separation is what lets the same object serve a notebook reader, an HTML
report, and an audit trail without being rewritten for each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.explain.schemas import Evidence, Finding, Recommendation, _json_value


@dataclass(slots=True)
class PreprocessResult:
    """What one preprocessing step did, what it found, and what you should know.

    Preprocessing methods on :class:`~buildml.session.Session` return the
    session itself so calls can be chained, which leaves nowhere to put the
    explanation. This object is where it goes: the session stores the most
    recent one on :attr:`~buildml.session.Session.last_preprocess`, and a
    condensed form lands in the session history so the whole sequence stays
    auditable.

    Attributes
    ----------
    operation:
        Which step produced this — ``'impute'``, ``'encode'``, ``'scale'``, and
        so on.
    plan:
        The fitted plan as plain data: the learned constants, the columns
        touched, the settings used. Enough to reproduce the step exactly.
    evidence:
        The measurements the step made, each carrying its own provenance so a
        claim can be traced back to the rows it came from.
    findings:
        Things worth your attention, such as a column that was almost entirely
        missing or a category that appeared only in the test rows. A step
        completing successfully does not mean it found nothing concerning.
    interpretation:
        Plain-language readings of the evidence — what the numbers mean for
        this dataset, rather than what they are.
    limitations:
        What this step cannot tell you. Reading these prevents over-reading the
        findings; median imputation, for instance, cannot detect that values
        are missing for a systematic reason.
    recommendations:
        Suggested follow-up actions, each tied to the finding that motivates
        it.
    methods:
        The techniques applied, named precisely enough to write into a methods
        section.
    warnings:
        Conditions that did not stop the step but might change how you read the
        result.

    See Also
    --------
    buildml.session.Session.last_preprocess : Where the most recent one lives.
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
        """Flatten the result into JSON-safe plain data.

        Every nested object is expanded and NumPy scalars are converted to
        built-in types, so the output survives ``json.dumps`` and can be
        embedded in a model card, a checkpoint, or an HTML report.

        Returns
        -------
        dict
            All attributes in expanded form, plus a ``summary`` key holding
            counts of findings and recommendations and the plan's key names —
            enough for a report to render a headline without walking the whole
            structure.
        """
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
