"""Tool registry for AI operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from buildml.ai.types import ConfirmPolicy, ToolCall
from buildml.core.errors import ValidationError


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Specification for one tool in the AI operator registry."""

    name: str
    description: str
    parameters: dict[str, Any]
    confirm_policy: ConfirmPolicy = ConfirmPolicy.CONFIRM
    session_method: str | None = None
    read_only: bool = False
    destructive: bool = False
    catalog_operation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "confirm_policy": self.confirm_policy.value,
            "session_method": self.session_method,
            "read_only": self.read_only,
            "destructive": self.destructive,
            "catalog_operation": self.catalog_operation,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _build_m1_tools() -> tuple[ToolSpec, ...]:
    """Build the M1 conservative tool allowlist."""
    return (
        ToolSpec(
            name="describe_dataset",
            description=(
                "Return a summary of the current dataset including column names, "
                "data types, row count, roles, and basic statistics. Does not "
                "execute any changes."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            read_only=True,
            catalog_operation="metadata",
        ),
        ToolSpec(
            name="explain_operation",
            description=(
                "Explain what a BuildML operation does, its prerequisites, "
                "parameters, and expected outputs using the explain catalog."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation name to explain (e.g. 'fit', 'split').",
                    },
                },
                "required": ["operation"],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="explain",
            read_only=True,
            catalog_operation="explain",
        ),
        ToolSpec(
            name="workflow_status",
            description=(
                "Return the current workflow status showing which operations "
                "are done, available, or blocked. Does not execute changes."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="workflow",
            read_only=True,
            catalog_operation="workflow",
        ),
        ToolSpec(
            name="eda_summary",
            description=(
                "Return a summary of exploratory data analysis findings "
                "including data quality issues, distributions, and recommendations."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="eda",
            read_only=True,
            catalog_operation="eda",
        ),
        ToolSpec(
            name="dry_run_plan",
            description=(
                "Preview what a plan would do without executing it. "
                "Returns validation results and expected state changes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "The plan name or operation sequence to dry-run.",
                    },
                },
                "required": ["plan"],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="dry_run",
            read_only=True,
            catalog_operation="dry_run",
        ),
        ToolSpec(
            name="set_roles",
            description=(
                "Assign semantic roles to columns (feature, target, id, exclude). "
                "This is a write operation that requires confirmation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mapping": {
                        "type": "object",
                        "description": "Column name to role mapping.",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["mapping"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="set_roles",
            read_only=False,
            catalog_operation="set_roles",
        ),
    )


class ToolRegistry:
    """Registry of allowed tools for the AI operator.

    Tools are allowlisted by name and category. Unlisted tools are rejected.
    """

    def __init__(self, tools: tuple[ToolSpec, ...] | None = None) -> None:
        if tools is None:
            tools = _build_m1_tools()
        self._tools = {t.name: t for t in tools}

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def validate_tool_call(self, call: ToolCall) -> ToolSpec:
        """Validate a tool call is in the registry.

        Raises
        ------
        ValidationError
            If the tool is not in the allowlist.
        """
        spec = self._tools.get(call.tool_name)
        if spec is None:
            raise ValidationError(
                f"Tool '{call.tool_name}' is not in the allowed tool registry. "
                f"Available tools: {sorted(self._tools.keys())}"
            )
        return spec

    def requires_confirmation(self, call: ToolCall) -> bool:
        """Check if a tool call requires user confirmation."""
        spec = self._tools.get(call.tool_name)
        if spec is None:
            return True
        if spec.destructive:
            return True
        return spec.confirm_policy != ConfirmPolicy.AUTO

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI tool format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    def read_only_tools(self) -> tuple[ToolSpec, ...]:
        """Return only read-only tools (for advisor mode)."""
        return tuple(t for t in self._tools.values() if t.read_only)


_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "system:",
    "assistant:",
    "SYSTEM:",
    "ASSISTANT:",
    "you are now",
    "new instructions:",
    "override:",
)


def sanitize_tool_result(result: Any) -> str:
    """Sanitize a tool result before feeding back to the LLM.

    Marks the result as data, not instructions, and scans for injection patterns.
    """
    text = str(result)
    for marker in _INJECTION_MARKERS:
        if marker.lower() in text.lower():
            text = text.replace(marker, f"[DATA: {marker}]")
    return f"[TOOL RESULT - DATA ONLY]\n{text}\n[END TOOL RESULT]"


def mark_untrusted_data(data: str, source: str = "user") -> str:
    """Mark data as untrusted with source context.

    Used to wrap column names, cell values, and user input before sending
    to the LLM to prevent instruction injection.
    """
    return f"[UNTRUSTED DATA FROM {source.upper()}]\n{data}\n[END UNTRUSTED DATA]"
