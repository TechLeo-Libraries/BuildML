"""The vocabulary the AI operator is built from.

Four ideas, and everything else in the domain composes them.

:class:`EgressLevel` says how much of your data may leave the machine. It is the
single most consequential setting in this domain, because an LLM provider is a
third party and the data you send is data you have disclosed.

:class:`ConfirmPolicy` says whether the model may act on its own. Language models
propose plausible things, and plausible is not the same as correct — the policy
decides who checks.

:class:`ToolCall` and :class:`Message` are the units of conversation: what the
model asked to run, and what was said. Both round-trip through dictionaries so
a transcript survives being written to disk.

:class:`StateDigest` is what the model is told about your Session — shape, roles,
and progress, never values.

See Also
--------
buildml.ai.privacy : Enforcing the egress level.
buildml.ai.tools : Enforcing the confirmation policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class EgressLevel(str, Enum):
    """How much of your data may be sent to an external model.

    Ordered from most private to least. Each level permits everything the
    previous one did, plus more.

    Attributes
    ----------
    SCHEMA_ONLY:
        Column names, dtypes, and roles. No values of any kind — not even
        aggregates. The safest useful level, and enough for the model to reason
        about structure.
    STATS_ONLY:
        Adds aggregates: counts, means, quantiles, cardinalities. No individual
        rows. Note that an aggregate over very few rows can still identify
        someone.
    REDACTED_SAMPLE:
        Adds actual rows, with configured columns removed or renamed. Redaction
        is only as good as the configuration, and a column you forgot to deny
        is a column that was sent.
    FULL_SAMPLE:
        Rows as they are. Appropriate for data you would be comfortable posting
        publicly, and not otherwise.

    Notes
    -----
    **Once data has been sent, it has been sent.** Provider retention policies
    vary and change; the only reliable control is what leaves the machine.
    Choose the lowest level that lets the model do the job — for most planning
    and advisory work, ``SCHEMA_ONLY`` is enough.

    **This is a ``str`` enum**, so it compares equal to its value and serialises
    as a plain string.

    See Also
    --------
    buildml.ai.privacy.EgressConfig : Where a level becomes a rule.
    buildml.ai.privacy.EgressManifest : The record of what was actually sent.
    """

    SCHEMA_ONLY = "schema_only"
    STATS_ONLY = "stats_only"
    REDACTED_SAMPLE = "redacted_sample"
    FULL_SAMPLE = "full_sample"


class ConfirmPolicy(str, Enum):
    """Whether a proposed action runs on its own or waits for you.

    Attributes
    ----------
    AUTO:
        Run without asking. Reserved for tools that only read — describing the
        data, summarising history, listing what is available.
    CONFIRM:
        Ask before running. The default for anything that changes Session state.
    ALWAYS_CONFIRM:
        Ask every time, and never treat a previous approval as covering this
        one. For actions that are expensive, destructive, or hard to undo.

    Notes
    -----
    **The policy belongs to the tool, not the model.** A model cannot request a
    weaker one, and cannot escalate its own permissions by asking differently.
    The registry decides, and the executor enforces.

    **``AUTO`` is a claim that the tool cannot cause harm**, which is a stronger
    claim than it first appears. A read-only tool that returns raw rows has
    disclosed those rows. Weigh what a tool reveals, not only what it changes.

    See Also
    --------
    buildml.ai.tools.ToolSpec : Where a tool declares its policy.
    buildml.ai.executor : Where the policy is enforced.
    """

    AUTO = "auto"
    CONFIRM = "confirm"
    ALWAYS_CONFIRM = "always_confirm"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """An action the model wants taken, named and with its arguments.

    Immutable, because a call that has been proposed should be the same call
    that gets reviewed and the same call that runs. Nothing between proposal and
    execution can quietly alter it.

    Attributes
    ----------
    tool_name:
        Which registered tool. Names not in the registry are rejected rather
        than guessed at.
    arguments:
        Keyword arguments, validated against the tool's schema before anything
        runs.
    call_id:
        The provider's identifier for this call, used to match a result back to
        its request in multi-call turns. Empty when the provider supplied none.

    Notes
    -----
    **A proposal is not a permission.** This records what was asked for; whether
    it happens depends on the tool's :class:`ConfirmPolicy` and on you.

    See Also
    --------
    buildml.ai.tools.ToolRegistry : Resolving and validating a call.
    buildml.ai.executor : Turning a call into an effect.
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the call as JSON-safe values.

        Used when writing a transcript and when sending the call back to a
        provider as conversation context.

        Returns
        -------
        dict
            Tool name, arguments, and call id.

        See Also
        --------
        from_dict : The inverse.
        """
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolCall:
        """Rebuild a call from its serialised form.

        Reads a proposal back from a stored transcript or a provider response.

        Parameters
        ----------
        payload:
            A mapping from :meth:`to_dict`.

        Returns
        -------
        ToolCall
            The reconstructed call.

        Raises
        ------
        KeyError
            If ``tool_name`` is absent. Arguments and call id default to empty,
            since a call with neither is still meaningful; a call without a
            name is not.

        Notes
        -----
        **Reconstruction does not validate against the registry.** A tool that
        has since been renamed or removed rebuilds fine here and fails at
        execution, which is the right place for that check.
        """
        return cls(
            tool_name=str(payload["tool_name"]),
            arguments=dict(payload.get("arguments") or {}),
            call_id=str(payload.get("call_id") or ""),
        )


@runtime_checkable
class SessionLike(Protocol):
    """The narrow slice of a Session the AI domain is allowed to see.

    A structural protocol, not a base class: anything with these three members
    satisfies it. That keeps the AI domain from importing
    :class:`~buildml.session.Session`, which would make the dependency circular,
    and makes it straightforward to pass a stand-in during testing.

    Notes
    -----
    **The narrowness is deliberate.** Reading history, the dataset, and metadata
    is enough to describe what has happened and what could happen next. Anything
    beyond that would let the AI domain reach into state it has no business
    touching directly — state changes go through the tool registry, where they
    are named, validated, and confirmable.

    See Also
    --------
    StateDigest : What actually gets summarised for the model.
    """

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return the recorded operations, oldest first.

        Read to summarise what has already happened, so advice continues the
        workflow instead of restarting it.

        Returns
        -------
        list of dict
            One entry per recorded operation.
        """
        ...

    @property
    def dataset(self) -> Any:
        """Return the loaded dataset, or ``None`` when nothing is loaded.

        Read for shape, columns, and roles. Values are read only when the
        egress level permits sending them.

        Returns
        -------
        Dataset or None
            The current dataset.
        """
        ...

    def metadata(self) -> dict[str, Any]:
        """Return Session-level state describing workflow progress.

        Read to determine which stages have completed — whether the data has
        been split, whether a model has been fitted.

        Returns
        -------
        dict
            Progress flags and related state.
        """
        ...


@dataclass(slots=True)
class StateDigest:
    """What the model is told about your Session, and nothing more.

    A model advising on next steps needs to know where you are: what the data
    looks like, which columns mean what, and how far the workflow has gone. It
    does not need the values, and this digest does not carry them.

    Attributes
    ----------
    has_dataset:
        Whether anything has been loaded.
    row_count, column_count:
        Shape. ``None`` when no dataset is loaded.
    columns:
        Column names in order. **Names can themselves be sensitive** — a column
        called ``patient_hiv_status`` discloses something before any value is
        sent.
    roles:
        Column to role, which is how the model knows what is being predicted.
    has_split:
        Whether the data has been partitioned. Advice about fitting is wrong
        without this.
    has_fit_result, has_dl_result, has_rag_index:
        Which paths have produced something.
    history_summary:
        Recent operation names, so advice follows from what you have already
        done rather than restarting.
    warnings:
        Problems found while building the digest — typically state that could
        not be read.

    Notes
    -----
    **This is the whole picture the model has, at ``SCHEMA_ONLY``.** Values
    reach it only at higher egress levels, through a different path.

    See Also
    --------
    EgressLevel : What may be added on top of this.
    """

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
        """Return the state summary as JSON-safe values.

        This is the form that goes into a prompt, so it is also the form worth
        inspecting when you want to see exactly what the model was told.

        Returns
        -------
        dict
            Shape, columns, roles, progress flags, history summary, and
            warnings.
        """
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
    """One turn in the conversation with the model.

    Follows the chat-completions shape that hosted providers expect, so a list
    of these can be sent as-is.

    Attributes
    ----------
    role:
        ``'system'`` for instructions, ``'user'`` for your input, ``'assistant'``
        for the model, ``'tool'`` for a tool's output being fed back.
    content:
        The text. Empty when an assistant turn is only requesting tool calls.
    tool_calls:
        Actions the model wants taken. Assistant messages only.
    tool_call_id:
        Which call this message answers. Tool messages only — without it the
        provider cannot match a result to its request.
    name:
        The tool's name, on tool messages.

    Notes
    -----
    **The conversation is the model's entire memory.** It carries no state
    between turns beyond what these messages contain, which is why dropping
    earlier ones to save tokens changes what the model knows.

    See Also
    --------
    ToolCall : What an assistant turn can request.
    buildml.ai.transcript : Persisting the conversation.
    """

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the message in the shape a chat provider expects.

        Optional fields are omitted when unset rather than sent as ``None``,
        because providers reject unexpected nulls on several of them.

        Returns
        -------
        dict
            Role and content, plus tool calls, call id, and name when present.

        See Also
        --------
        from_dict : The inverse.
        """
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
        """Rebuild a message from its serialised form.

        Reads a turn back from a stored transcript or a provider response,
        reconstructing any nested tool calls along the way.

        Parameters
        ----------
        payload:
            A mapping from :meth:`to_dict`.

        Returns
        -------
        Message
            The reconstructed turn.

        Raises
        ------
        KeyError
            If ``role`` is absent. Content defaults to empty, since an
            assistant turn that only requests tool calls legitimately has none.
        """
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
