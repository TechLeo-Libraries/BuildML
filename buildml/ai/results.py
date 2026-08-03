"""What the AI operator hands back, and why it is shaped this way.

A model's advice is only as trustworthy as the record of how it was produced.
Three types carry that record.

:class:`TranscriptEntry` is one event in the conversation: a message, a
proposed call, a result, a confirmation, or a failure: timestamped and paired
with the egress manifest that says what left the machine at that moment.

:class:`PlanStep` is one recommended action. It carries not just the operation
but the reasoning behind it, what must already be true, and what will change.
Advice you cannot interrogate is advice you cannot check.

:class:`PlanResult` is the plan as a whole, with its assumptions, limitations,
and alternatives stated rather than implied.

Notes
-----
**A plan is a suggestion, not a validated pipeline.** Language models produce
confident, well-structured, wrong plans as readily as right ones. The
``rationale``, ``assumptions``, and ``limitations`` fields exist so the
reasoning is visible enough to disagree with.

See Also
--------
buildml.ai.planner : Producing and executing plans.
buildml.ai.transcript : Persisting the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.ai.privacy import EgressManifest
from buildml.ai.types import EgressLevel, Message, ToolCall


@dataclass(slots=True)
class TranscriptEntry:
    """One recorded event in the conversation.

    Deliberately wide: an entry can be a message, a tool call, a tool result, a
    confirmation decision, or an error. Which fields are populated depends on
    ``entry_type``, and unpopulated fields stay ``None`` rather than being
    invented.

    Attributes
    ----------
    timestamp:
        When it happened, as an ISO 8601 string.
    entry_type:
        What kind of event this is, and therefore which fields to read.
    message:
        The conversation turn, on message entries.
    tool_call:
        What was proposed, on tool-call entries.
    tool_result:
        What the tool returned, as text.
    egress_manifest:
        What data left the machine at this point. **Present on every entry that
        sent anything**, which is what makes a transcript auditable rather than
        merely descriptive.
    confirmed:
        Whether you approved this action. ``None`` when no confirmation was
        required.
    error:
        What went wrong, on failure entries.
    metadata:
        Anything else worth recording.

    Notes
    -----
    **The transcript is the audit trail.** If a question later arises about
    what was sent to a provider, the answer is in the manifests attached to
    these entries, not in the provider's logs.

    See Also
    --------
    buildml.ai.transcript.TranscriptStore : Where these are kept.
    """

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
        """Return the entry as JSON-safe values.

        Absent fields are omitted rather than written as ``None``, so a stored
        transcript stays readable and a message entry does not carry five empty
        tool-related keys.

        Returns
        -------
        dict
            Timestamp and entry type, plus whichever of message, tool call,
            tool result, egress manifest, confirmation, error, and metadata
            are present.

        See Also
        --------
        from_dict : The inverse.
        """
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
        """Rebuild an entry from its serialised form.

        Reconstructs the nested message, tool call, and egress manifest, so a
        transcript loaded from disk is as inspectable as one held in memory.

        Parameters
        ----------
        payload:
            A mapping from :meth:`to_dict`.

        Returns
        -------
        TranscriptEntry
            The reconstructed event.

        Raises
        ------
        KeyError
            If ``timestamp`` or ``entry_type`` is absent. Everything else is
            optional by design; an entry without a time or a kind is not an
            entry.
        """
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
    """One recommended action, with the reasoning attached.

    A bare list of operations is not a plan you can evaluate. This carries the
    justification, the preconditions, and the expected effect alongside the
    operation, so you can tell a good recommendation from a plausible one.

    Attributes
    ----------
    operation:
        Which Session operation to run.
    description:
        What it does, in plain terms.
    rationale:
        Why it is recommended here. **The field most worth reading**: a
        rationale that does not follow from your data is the clearest sign the
        plan is generic rather than considered.
    prerequisites:
        What must already be true. A step whose prerequisites are unmet will
        fail, and the ordering of a plan is only meaningful through these.
    expected_changes:
        What will differ afterwards, so a surprising outcome is recognisable as
        one.
    evidence:
        What in your Session state supports this. Empty when the model offered
        none, which is itself informative.
    warnings:
        Risks specific to this step.
    parameters:
        Arguments for the operation. Validated at execution, not here.

    Notes
    -----
    **A step is a proposal.** Nothing runs until it goes through the executor,
    where the tool's confirmation policy applies.

    See Also
    --------
    PlanResult : The containing plan.
    buildml.ai.planner : Executing steps under a budget.
    """

    operation: str
    description: str
    rationale: str
    prerequisites: tuple[str, ...]
    expected_changes: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the step as JSON-safe values.

        Keeps the rationale and evidence with the operation, so a logged plan
        can still be argued with later.

        Returns
        -------
        dict
            Operation, description, rationale, prerequisites, expected changes,
            evidence, warnings, and parameters.
        """
        return {
            "operation": self.operation,
            "description": self.description,
            "rationale": self.rationale,
            "prerequisites": list(self.prerequisites),
            "expected_changes": list(self.expected_changes),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "parameters": dict(self.parameters),
        }


@dataclass(slots=True)
class PlanResult:
    """A proposed sequence of steps toward a stated goal.

    What ``ai_plan`` returns. The steps are ordered, but the ordering is the
    model's suggestion: the real dependencies live in each step's
    ``prerequisites``.

    Attributes
    ----------
    goal:
        What you asked for, echoed back. Worth checking: a plan aimed at a
        misread goal is coherent and useless.
    steps:
        The recommended actions in order.
    current_state_summary:
        The model's reading of where your Session is. **Check this first**: if
        it is wrong, nothing after it can be right.
    assumptions:
        What the model took for granted. Often where an unstated leap hides.
    limitations:
        What the plan does not cover.
    alternatives:
        Other approaches considered. Their presence signals the model weighed
        options rather than producing the first thing that fit.
    egress_manifest:
        What data was sent to produce this. ``None`` for offline providers.
    raw_response:
        The unparsed model output, for when the structured form lost something.
    usage:
        Token counts, which is how a run's cost is accounted for.

    Notes
    -----
    **Nothing here has been validated against your data.** The plan was produced
    from a state digest and whatever the egress level allowed; it can reference
    operations that will fail, or recommend steps whose preconditions do not
    hold. Execution is where that gets discovered.

    **Confidence in the prose is not evidence.** A model states a wrong plan in
    the same tone as a right one. Read ``current_state_summary`` and the
    per-step ``rationale`` for whether the reasoning actually engages with your
    situation.

    See Also
    --------
    PlanStep : One entry.
    buildml.ai.planner.PlanExecutionResult : What happens when it runs.
    """

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
        """Return the plan as JSON-safe values.

        Everything except ``raw_response``, which is omitted because it
        duplicates the structured fields at length and can be large.

        Returns
        -------
        dict
            Goal, steps, state summary, assumptions, limitations,
            alternatives, egress manifest, and token usage.

        Notes
        -----
        The egress manifest is kept, so a logged plan records what was
        disclosed to produce it.
        """
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
