"""Core type definitions for the AI operator domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class EgressLevel(str, Enum):
    """Privacy level for data sent to an external LLM provider."""

    SCHEMA_ONLY = "schema_only"
    STATS_ONLY = "stats_only"
    REDACTED_SAMPLE = "redacted_sample"
    FULL_SAMPLE = "full_sample"


class ConfirmPolicy(str, Enum):
    """When user confirmation is required before tool execution."""

    AUTO = "auto"
    CONFIRM = "confirm"
    ALWAYS_CONFIRM = "always_confirm"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A proposed or executed tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolCall:
        return cls(
            tool_name=str(payload["tool_name"]),
            arguments=dict(payload.get("arguments") or {}),
            call_id=str(payload.get("call_id") or ""),
        )


@runtime_checkable
class SessionLike(Protocol):
    """Minimal Session interface for AI operator access."""

    @property
    def history(self) -> list[dict[str, Any]]: ...

    @property
    def dataset(self) -> Any: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class StateDigest:
    """Compact Session state summary for LLM context."""

    has_dataset: bool
    row_count: int | None
    column_count: int | None
    columns: tuple[str, ...]
    roles: dict[str, str]
    has_split: bool
    has_fit_result: bool
    has_dl_result: bool
    has_rag_index: bool
    history_summary: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_dataset": self.has_dataset,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "roles": dict(self.roles),
            "has_split": self.has_split,
            "has_fit_result": self.has_fit_result,
            "has_dl_result": self.has_dl_result,
            "has_rag_index": self.has_rag_index,
            "history_summary": list(self.history_summary),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class Message:
    """A conversation message for the AI operator."""

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Message:
        tool_calls = tuple(
            ToolCall.from_dict(tc) for tc in payload.get("tool_calls") or []
        )
        return cls(
            role=str(payload["role"]),
            content=str(payload.get("content") or ""),
            tool_calls=tool_calls,
            tool_call_id=payload.get("tool_call_id"),
            name=payload.get("name"),
        )
