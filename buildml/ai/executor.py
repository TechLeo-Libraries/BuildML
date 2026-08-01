"""Propose-confirm-execute flow for AI operator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from buildml.ai.privacy import EgressManifest
from buildml.ai.tools import ToolRegistry
from buildml.ai.types import ConfirmPolicy, ToolCall
from buildml.core.errors import ValidationError


@dataclass(slots=True)
class ExecutorProposal:
    """A proposed tool execution awaiting confirmation."""

    tool_call: ToolCall
    description: str
    rationale: str
    expected_changes: tuple[str, ...]
    requires_confirmation: bool
    confirm_policy: ConfirmPolicy
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call": self.tool_call.to_dict(),
            "description": self.description,
            "rationale": self.rationale,
            "expected_changes": list(self.expected_changes),
            "requires_confirmation": self.requires_confirmation,
            "confirm_policy": self.confirm_policy.value,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ExecutorResult:
    """Result from ai_execute: confirmed tool execution."""

    tool_call: ToolCall
    confirmed: bool
    executed: bool
    result: Any = None
    result_summary: str = ""
    error: str | None = None
    egress_manifest: EgressManifest | None = None
    state_changes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call": self.tool_call.to_dict(),
            "confirmed": self.confirmed,
            "executed": self.executed,
            "result_summary": self.result_summary,
            "error": self.error,
            "egress_manifest": self.egress_manifest.to_dict() if self.egress_manifest else None,
            "state_changes": list(self.state_changes),
        }


def propose_tool_execution(
    tool_name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry,
) -> ExecutorProposal:
    """Create a proposal for a tool execution.

    Validates the tool is in the registry and determines confirmation requirements.
    """
    call = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        call_id=f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    )

    spec = registry.validate_tool_call(call)

    requires_confirm = spec.confirm_policy != ConfirmPolicy.AUTO
    if spec.destructive:
        requires_confirm = True

    warnings: list[str] = []
    if spec.destructive:
        warnings.append("This is a destructive operation that cannot be undone.")
    if not spec.read_only:
        warnings.append("This operation will modify Session state.")

    return ExecutorProposal(
        tool_call=call,
        description=spec.description,
        rationale=f"Tool '{tool_name}' from the allowed registry.",
        expected_changes=_infer_expected_changes(tool_name, arguments),
        requires_confirmation=requires_confirm,
        confirm_policy=spec.confirm_policy,
        warnings=tuple(warnings),
    )


def execute_tool(
    session: Any,
    proposal: ExecutorProposal,
    confirmed: bool,
    registry: ToolRegistry,
) -> ExecutorResult:
    """Execute a proposed tool if confirmed.

    Validates the tool is still in the registry and calls the Session method.
    """
    call = proposal.tool_call

    if proposal.requires_confirmation and not confirmed:
        return ExecutorResult(
            tool_call=call,
            confirmed=False,
            executed=False,
            error="Execution requires confirmation but was not confirmed.",
        )

    spec = registry.validate_tool_call(call)

    if spec.destructive and not confirmed:
        return ExecutorResult(
            tool_call=call,
            confirmed=False,
            executed=False,
            error="Destructive operations always require explicit confirmation.",
        )

    try:
        result, state_changes = _dispatch_tool(session, call, spec)
        result_summary = _summarize_result(result)

        return ExecutorResult(
            tool_call=call,
            confirmed=confirmed,
            executed=True,
            result=result,
            result_summary=result_summary,
            state_changes=state_changes,
        )

    except Exception as e:
        error_msg = _redact_exception_message(str(e))
        return ExecutorResult(
            tool_call=call,
            confirmed=confirmed,
            executed=False,
            error=f"Execution failed: {error_msg}",
        )


def _redact_exception_message(msg: str, max_length: int = 200) -> str:
    """Redact and truncate exception messages before surfacing/storing."""
    import re

    key_patterns = (
        re.compile(r"sk-[a-zA-Z0-9_-]{10,}"),
        re.compile(r"api[_-]?key[\"']?\s*[:=]\s*[\"'][^\"']+[\"']", re.IGNORECASE),
        re.compile(r"bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),
    )

    result = msg
    for pattern in key_patterns:
        result = pattern.sub("***REDACTED***", result)

    if len(result) > max_length:
        result = result[:max_length] + "... [truncated]"

    return result


def _dispatch_tool(
    session: Any,
    call: ToolCall,
    spec: Any,
) -> tuple[Any, tuple[str, ...]]:
    """Dispatch a tool call to the appropriate Session method."""
    state_changes: list[str] = []

    if call.tool_name == "set_roles":
        mapping = call.arguments.get("mapping", {})
        if not mapping:
            raise ValidationError("set_roles requires a non-empty mapping argument.")

        before_roles = dict(getattr(session.dataset, "roles", {}) or {})
        session.set_roles(mapping)

        for col, role in mapping.items():
            old_role = before_roles.get(col, "unassigned")
            state_changes.append(f"Column '{col}': {old_role} -> {role}")

        return {"roles_set": mapping}, tuple(state_changes)

    elif call.tool_name == "describe_dataset":
        return session.metadata(), ()

    elif call.tool_name == "explain_operation":
        op = call.arguments.get("operation", "")
        result = session.explain(op)
        return result, ()

    elif call.tool_name == "workflow_status":
        result = session.workflow()
        return result, ()

    elif call.tool_name == "eda_summary":
        result = session.eda()
        return result, ()

    elif call.tool_name == "dry_run_plan":
        plan = call.arguments.get("plan", "")
        result = session.dry_run(plan)
        return result, ()

    else:
        raise ValidationError(f"No dispatch handler for tool: {call.tool_name}")


def _infer_expected_changes(tool_name: str, arguments: dict[str, Any]) -> tuple[str, ...]:
    """Infer expected state changes from a tool call."""
    changes: list[str] = []

    if tool_name == "set_roles":
        mapping = arguments.get("mapping", {})
        for col, role in mapping.items():
            changes.append(f"Column '{col}' will be assigned role '{role}'.")

    elif tool_name in (
        "describe_dataset", "explain_operation", "workflow_status", "eda_summary", "dry_run_plan"
    ):
        changes.append("No state changes (read-only operation).")

    return tuple(changes) if changes else ("Unknown state changes.",)


def _summarize_result(result: Any) -> str:
    """Create a brief summary of a tool result."""
    if result is None:
        return "No result returned."

    if isinstance(result, dict):
        keys = list(result.keys())[:5]
        return f"Result with keys: {keys}"

    if hasattr(result, "to_dict"):
        return f"Result: {type(result).__name__}"

    text = str(result)
    if len(text) > 200:
        return text[:200] + "..."
    return text


class IterationLimitExceeded(ValidationError):
    """Raised when max tool iterations is exceeded."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f"Maximum tool iterations ({limit}) exceeded. "
            "This limit prevents runaway loops."
        )
