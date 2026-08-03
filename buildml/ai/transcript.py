"""Keep a record of what the AI operator said, did, and disclosed.

An agent that acts on your Session needs to leave an account of itself. A
:class:`TranscriptStore` is that account: every message, every proposed call,
every result, every confirmation, every failure, and — crucially — every
:class:`~buildml.ai.privacy.EgressManifest`, so the question of what a provider
received always has a written answer.

Transcripts are redacted on the way out. Secrets have a way of ending up in
conversation text — a key pasted into a prompt, a bearer token in an error
message, a connection string in a stack trace — and a transcript written to
disk is a file that gets copied, attached, and committed. Known credential
shapes are masked before anything is persisted.

Notes
-----
**Redaction covers message content, tool results, and error text.** Tool call
arguments are not scanned, because they are validated against a schema and are
structured rather than free text. A tool that accepts an opaque string argument
is a place to be careful.

See Also
--------
buildml.ai.results.TranscriptEntry : One recorded event.
buildml.ai.privacy : What the manifests record.
"""

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
    """The running record of one AI operator session.

    Append-only in practice: the ``add_`` methods each timestamp an event and
    push it on. Reading the entries in order reconstructs the run.

    Attributes
    ----------
    entries:
        Events in chronological order.
    session_id:
        Identifier for this conversation. Generated from the current time when
        not supplied.
    created_at:
        When the store was created, as an ISO 8601 UTC string.
    schema_id:
        The persistence format version, checked on load so an incompatible file
        is refused rather than half-read.
    metadata:
        Anything else worth keeping alongside the run.

    Notes
    -----
    **Redaction happens at serialisation, not on append.** Entries are held in
    memory as they occurred, so live inspection shows the truth; masking is
    applied when :meth:`to_dict` runs with ``redact=True``, which is the
    default and the state anything reaching disk is in.

    Examples
    --------
    Record a turn and persist it::

        store = TranscriptStore(session_id="run-42")
        store.add_message(Message(role="user", content="what next?"))
        store.add_egress_manifest(manifest)
        save_transcript(store, "runs/run-42.json")

    See Also
    --------
    save_transcript : Writing one out.
    load_transcript : Reading one back.
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
        """Record a conversation turn.

        Timestamped at the moment of the call, so the transcript reflects
        wall-clock order.

        Parameters
        ----------
        message:
            The turn — from you, the model, the system prompt, or a tool.

        See Also
        --------
        add_tool_call : Recording a proposal rather than a turn.
        """
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
        """Record that an action was proposed, and whether it was approved.

        Called for proposals, not only for executions — a rejected call is part
        of the account, and a transcript that only showed what ran would be a
        partial one.

        Parameters
        ----------
        tool_call:
            What was proposed.
        confirmed:
            Whether you approved it. ``None`` when no confirmation was
            required, ``False`` when it was declined.

        See Also
        --------
        add_tool_result : Recording what came back.
        """
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
        """Record what a tool returned.

        The call is stored alongside the result, so an entry read on its own
        still says what produced it.

        Parameters
        ----------
        tool_call:
            The call this answers.
        result:
            What came back, as text.

        Notes
        -----
        **Results are redacted at serialisation.** A tool that surfaces a
        connection string or a token in its output will have it masked before
        the transcript reaches disk.

        See Also
        --------
        add_error : Recording a failure instead.
        """
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="tool_result",
            tool_call=tool_call,
            tool_result=result,
        ))

    def add_egress_manifest(self, manifest: EgressManifest) -> None:
        """Record what data left the machine.

        **The entry that makes a transcript an audit trail.** Everything else
        describes the conversation; this describes the disclosure.

        Parameters
        ----------
        manifest:
            The record produced alongside the payload — columns sent, columns
            withheld, renames applied, rows sent.

        Notes
        -----
        Manifests are never redacted. They contain column names and counts, no
        values, and their whole purpose is to be readable afterwards.

        See Also
        --------
        buildml.ai.privacy.build_egress_payload : Where manifests come from.
        """
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="egress",
            egress_manifest=manifest,
        ))

    def add_error(self, error: str, tool_call: ToolCall | None = None) -> None:
        """Record a failure, with the call that caused it when there was one.

        Failures belong in the account as much as successes do. An agent that
        tried something invalid five times before succeeding behaved differently
        from one that succeeded first, and only the transcript shows that.

        Parameters
        ----------
        error:
            What went wrong.
        tool_call:
            The call that failed, when the failure came from one.

        Notes
        -----
        **Error text is redacted at serialisation.** Stack traces and driver
        messages are a common route for credentials to reach a log file.
        """
        self.entries.append(TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type="error",
            tool_call=tool_call,
            error=error,
        ))

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Return the whole transcript as JSON-safe values.

        The form that gets written to disk. Redaction is applied here rather
        than on append, so the in-memory record stays faithful while anything
        persisted is masked.

        Parameters
        ----------
        redact:
            Mask credential-shaped text in message content, tool results, and
            error text. On by default, and worth leaving on.

        Returns
        -------
        dict
            Schema id, session id, creation time, entry count, the entries, and
            metadata.

        Notes
        -----
        **Redaction matches known credential shapes**, not everything secret. A
        password with no distinguishing format passes through. It reliably
        catches API keys, bearer tokens, and BuildML's own key variables.

        **Turning it off means the file may contain live credentials.** Only do
        so for in-memory inspection.

        See Also
        --------
        from_dict : The inverse.
        save_transcript : This plus writing the file.
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
        """Rebuild a transcript from its serialised form.

        Checks the schema version before reading anything, so a file written by
        an incompatible version is refused rather than partially interpreted.

        Parameters
        ----------
        payload:
            A mapping from :meth:`to_dict`.

        Returns
        -------
        TranscriptStore
            The reconstructed record.

        Raises
        ------
        ValidationError
            If the payload declares a different schema. A payload with no
            schema id is accepted and treated as current, which keeps
            hand-written fixtures usable.

        Notes
        -----
        **A loaded transcript is already redacted**, if it was saved with
        redaction on. What was masked is gone; loading does not recover it.
        """
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
    """Write a transcript to a JSON file, masked by default.

    Creates any missing parent directories, then writes indented JSON. Values
    that are not JSON-serialisable fall back to their string form rather than
    failing the write — a transcript that saves imperfectly is more useful than
    one that does not save.

    Parameters
    ----------
    transcript:
        What to write.
    path:
        Where to write it. Parents are created as needed.
    redact:
        Mask credential-shaped text before writing. On by default.

    Returns
    -------
    Path
        The path written to.

    Raises
    ------
    OSError
        If the path cannot be created or written.

    Notes
    -----
    **Existing files are overwritten without warning.** Use a distinct name per
    run — the store's ``session_id`` is generated for exactly this.

    **Leave redaction on for anything that leaves your machine.** Transcripts
    get attached to issues and committed to repositories, which is where a key
    in a message body becomes a problem.

    Examples
    --------
    Save under the generated session id::

        save_transcript(store, f"transcripts/{store.session_id}.json")

    See Also
    --------
    load_transcript : Reading one back.
    TranscriptStore.to_dict : The serialisation itself.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = transcript.to_dict(redact=redact)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return out_path


def load_transcript(path: str | Path) -> TranscriptStore:
    """Read a transcript back from a JSON file.

    Reconstructs the entries, including their nested messages, tool calls, and
    egress manifests, so a stored run can be reviewed as fully as a live one.

    Parameters
    ----------
    path:
        The file to read.

    Returns
    -------
    TranscriptStore
        The reconstructed record.

    Raises
    ------
    ValidationError
        If the file does not exist, or declares an incompatible schema.
    json.JSONDecodeError
        If the file is not valid JSON.

    Notes
    -----
    **What was redacted at save time stays redacted.** Loading reads what was
    written; masking is not reversible.

    See Also
    --------
    save_transcript : Writing one.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise ValidationError(f"Transcript file not found: {in_path}")

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    return TranscriptStore.from_dict(data)
