"""Transcript storage for AI operator conversations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from buildml.ai.privacy import EgressManifest
from buildml.ai.results import TranscriptEntry
from buildml.ai.types import Message, ToolCall
from buildml.core.errors import ValidationError

TRANSCRIPT_SCHEMA_ID = "buildml.ai_transcript.v1"

_KEY_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9_-]{10,}"),
    re.compile(r"api[_-]?key[\"']?\s*[:=]\s*[\"'][^\"']+[\"']", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),
    re.compile(r"BUILDML_[A-Z_]+_API_KEY=[^\s]+"),
)

_KEY_MASK = "***REDACTED_KEY***"


def _redact_secrets(text: str) -> str:
    """Redact potential API keys and secrets from text."""
    result = text
    for pattern in _KEY_PATTERNS:
        result = pattern.sub(_KEY_MASK, result)
    return result


def _redact_entry(entry: TranscriptEntry) -> TranscriptEntry:
    """Redact secrets from a transcript entry."""
    message = entry.message
    if message is not None:
        message = Message(
            role=message.role,
            content=_redact_secrets(message.content),
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )

    tool_result = entry.tool_result
    if tool_result is not None:
        tool_result = _redact_secrets(tool_result)

    error = entry.error
    if error is not None:
        error = _redact_secrets(error)

    return TranscriptEntry(
        timestamp=entry.timestamp,
        entry_type=entry.entry_type,
        message=message,
        tool_call=entry.tool_call,
        tool_result=tool_result,
        egress_manifest=entry.egress_manifest,
        confirmed=entry.confirmed,
        error=error,
        metadata=entry.metadata,
    )


@dataclass(slots=True)
class TranscriptStore:
    """Storage for AI operator conversation transcripts.

    Secrets (API keys, raw data) are redacted before persistence by default.
    Egress manifests are recorded instead of raw data.
    """

    entries: list[TranscriptEntry] = field(default_factory=list)
    session_id: str = ""
    created_at: str = ""
    schema_id: str = TRANSCRIPT_SCHEMA_ID
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.session_id:
            self.session_id = f"ai_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    def add_message(self, message: Message) -> None:
        """Add a conversation message to the transcript."""
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="message",
            message=message,
        ))

    def add_tool_call(
        self,
        tool_call: ToolCall,
        *,
        confirmed: bool | None = None,
    ) -> None:
        """Add a tool call to the transcript."""
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="tool_call",
            tool_call=tool_call,
            confirmed=confirmed,
        ))

    def add_tool_result(
        self,
        tool_call: ToolCall,
        result: str,
    ) -> None:
        """Add a tool result to the transcript."""
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="tool_result",
            tool_call=tool_call,
            tool_result=result,
        ))

    def add_egress_manifest(self, manifest: EgressManifest) -> None:
        """Record an egress event (what left the machine)."""
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="egress",
            egress_manifest=manifest,
        ))

    def add_error(self, error: str, tool_call: ToolCall | None = None) -> None:
        """Record an error in the transcript."""
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="error",
            tool_call=tool_call,
            error=error,
        ))

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Convert to dictionary for persistence.

        Parameters
        ----------
        redact
            If True, redact potential secrets from the output.
        """
        entries = self.entries
        if redact:
            entries = [_redact_entry(e) for e in entries]

        return {
            "schema_id": self.schema_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "entry_count": len(entries),
            "entries": [e.to_dict() for e in entries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TranscriptStore:
        """Restore a transcript from a dictionary."""
        schema_id = payload.get("schema_id", "")
        if schema_id and schema_id != TRANSCRIPT_SCHEMA_ID:
            raise ValidationError(
                f"Incompatible transcript schema: expected {TRANSCRIPT_SCHEMA_ID}, "
                f"got {schema_id}"
            )

        entries = [
            TranscriptEntry.from_dict(e)
            for e in payload.get("entries", [])
        ]

        return cls(
            entries=entries,
            session_id=payload.get("session_id", ""),
            created_at=payload.get("created_at", ""),
            schema_id=TRANSCRIPT_SCHEMA_ID,
            metadata=dict(payload.get("metadata") or {}),
        )


def save_transcript(
    transcript: TranscriptStore,
    path: str | Path,
    *,
    redact: bool = True,
) -> Path:
    """Save a transcript to a JSON file.

    Parameters
    ----------
    transcript
        The transcript to save.
    path
        Output file path.
    redact
        If True, redact potential secrets before saving.

    Returns
    -------
    Path
        The resolved output path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = transcript.to_dict(redact=redact)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return out_path


def load_transcript(path: str | Path) -> TranscriptStore:
    """Load a transcript from a JSON file.

    Parameters
    ----------
    path
        Input file path.

    Returns
    -------
    TranscriptStore
        The loaded transcript.

    Raises
    ------
    ValidationError
        If the file schema is incompatible.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise ValidationError(f"Transcript file not found: {in_path}")

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    return TranscriptStore.from_dict(data)
