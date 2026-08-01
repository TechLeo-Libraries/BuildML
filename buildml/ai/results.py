"""Typed results for AI operator operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.ai.privacy import EgressManifest
from buildml.ai.types import EgressLevel, Message, ToolCall


@dataclass(slots=True)
class TranscriptEntry:
    """One entry in the AI conversation transcript."""

    timestamp: str
    entry_type: str
    message: Message | None = None
    tool_call: ToolCall | None = None
    tool_result: str | None = None
    egress_manifest: EgressManifest | None = None
    confirmed: bool | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
        }
        if self.message is not None:
            result["message"] = self.message.to_dict()
        if self.tool_call is not None:
            result["tool_call"] = self.tool_call.to_dict()
        if self.tool_result is not None:
            result["tool_result"] = self.tool_result
        if self.egress_manifest is not None:
            result["egress_manifest"] = self.egress_manifest.to_dict()
        if self.confirmed is not None:
            result["confirmed"] = self.confirmed
        if self.error is not None:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TranscriptEntry:
        message = None
        if payload.get("message"):
            message = Message.from_dict(payload["message"])
        tool_call = None
        if payload.get("tool_call"):
            tool_call = ToolCall.from_dict(payload["tool_call"])
        egress_manifest = None
        if payload.get("egress_manifest"):
            em = payload["egress_manifest"]
            egress_manifest = EgressManifest(
                level=EgressLevel(em["level"]),
                columns_sent=tuple(em.get("columns_sent") or []),
                columns_denied=tuple(em.get("columns_denied") or []),
                columns_renamed=dict(em.get("columns_renamed") or {}),
                rows_sent=em.get("rows_sent", 0),
                estimated_tokens=em.get("estimated_tokens"),
                warnings=tuple(em.get("warnings") or []),
            )
        return cls(
            timestamp=str(payload["timestamp"]),
            entry_type=str(payload["entry_type"]),
            message=message,
            tool_call=tool_call,
            tool_result=payload.get("tool_result"),
            egress_manifest=egress_manifest,
            confirmed=payload.get("confirmed"),
            error=payload.get("error"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class PlanStep:
    """One step in a structured AI plan."""

    operation: str
    description: str
    rationale: str
    prerequisites: tuple[str, ...]
    expected_changes: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "description": self.description,
            "rationale": self.rationale,
            "prerequisites": list(self.prerequisites),
            "expected_changes": list(self.expected_changes),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class PlanResult:
    """Result from ai_plan: structured next-step recommendations."""

    goal: str
    steps: tuple[PlanStep, ...]
    current_state_summary: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    egress_manifest: EgressManifest | None = None
    raw_response: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_state_summary": self.current_state_summary,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "alternatives": list(self.alternatives),
            "egress_manifest": self.egress_manifest.to_dict() if self.egress_manifest else None,
            "usage": dict(self.usage),
        }
