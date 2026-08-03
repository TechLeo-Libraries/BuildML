"""Turn a plan into actions, one confirmable step at a time.

A :class:`~buildml.ai.results.PlanResult` is prose and structure; nothing in it
runs. This module bridges that gap. Each step's operation is matched to a
registered tool, turned into a proposal, and executed under the confirmation
policy that tool carries.

The execution model is **pause, do not push through**. Hitting a step that needs
confirmation stops the run and reports where: it does not skip ahead, and it
does not assume. You supply confirmations by step index and run again. A
destructive step requires explicit confirmation whatever else is configured.

Two bounds guard the loop. :class:`BudgetTracker` caps tokens and money across a
run, raising :class:`BudgetExceeded` when either ceiling is passed. ``max_steps``
caps how far a single call proceeds.

Notes
-----
**A plan that executes cleanly is not a plan that was correct.** Every step
succeeding means the operations were valid and their preconditions held. Whether
the resulting pipeline is sound is a separate question, and the answer is in the
results, not in the absence of errors.

See Also
--------
buildml.ai.results.PlanResult : What gets executed.
buildml.ai.executor : Executing a single call.
buildml.ai.tools : The allowlist steps are matched against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from buildml.ai.executor import (
    ExecutorProposal,
    ExecutorResult,
    execute_tool,
    propose_tool_execution,
)
from buildml.ai.privacy import EgressManifest
from buildml.ai.results import PlanResult, PlanStep
from buildml.ai.tools import ToolRegistry
from buildml.core.errors import ValidationError


@dataclass(slots=True)
class PlanStepExecution:
    """What happened to one step: skipped, blocked, failed, or run.

    Four outcomes, distinguished by which fields are set. ``skipped`` means the
    step never mapped to a tool. ``requires_confirmation`` with ``confirmed``
    false means it was blocked awaiting approval. An ``error`` means it was
    attempted and failed. ``executed`` means it ran.

    Attributes
    ----------
    step_index:
        Position in the plan. **The key you supply confirmations against.**
    operation:
        What the step asked for.
    proposal:
        The validated call. ``None`` when the step was skipped.
    result:
        What execution returned. ``None`` unless it ran.
    skipped:
        Whether the step was passed over.
    skip_reason:
        Why: usually that the operation matched no registered tool.
    requires_confirmation:
        Whether approval was needed.
    confirmed:
        Whether approval was given.
    executed:
        Whether it actually ran. **The only field that means the Session
        changed.**
    error:
        What went wrong.

    Notes
    -----
    **Not executed is not the same as failed.** A step can be skipped or
    blocked with no error at all, which is the normal way a run pauses.

    See Also
    --------
    PlanExecutionResult : The whole run.
    """

    step_index: int
    operation: str
    proposal: ExecutorProposal | None = None
    result: ExecutorResult | None = None
    skipped: bool = False
    skip_reason: str = ""
    requires_confirmation: bool = False
    confirmed: bool = False
    executed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the step outcome as JSON-safe values.

        Keeps the proposal alongside the result, so a logged step records what
        was going to run as well as what happened.

        Returns
        -------
        dict
            Index, operation, proposal, result, skip state and reason,
            confirmation state, execution state, and error.
        """
        return {
            "step_index": self.step_index,
            "operation": self.operation,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "result": self.result.to_dict() if self.result else None,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "requires_confirmation": self.requires_confirmation,
            "confirmed": self.confirmed,
            "executed": self.executed,
            "error": self.error,
        }


@dataclass(slots=True)
class PlanExecutionResult:
    """How far a plan got, and what stopped it.

    A run rarely completes in one call. It pauses at the first step needing
    approval, and this says where and why: enough to confirm and resume.

    Attributes
    ----------
    plan:
        The plan that was executed.
    step_executions:
        One outcome per step, in order.
    completed_steps:
        How many actually ran. Skipped and blocked steps do not count.
    total_steps:
        How many the plan had.
    stopped_at_step:
        Where it halted. ``None`` when it reached the end.
    stop_reason:
        Why it halted, in words.
    requires_confirmation_at:
        **The index to confirm to resume.** Set when a run paused for approval
        rather than for an error.
    all_confirmed:
        Whether every step needing approval got it.
    all_executed:
        Whether every step either ran or was skipped.
    egress_manifest:
        What was disclosed producing the plan. Execution itself sends nothing.
    usage:
        Token counts.

    Notes
    -----
    **``completed_steps < total_steps`` is normal, not a failure.** It is what a
    confirmation pause looks like. Check ``requires_confirmation_at`` and
    ``stop_reason`` to tell a pause from a fault.

    **Steps already executed stay executed.** Resuming re-runs from the start of
    the plan; operations that already applied will apply again. Confirm
    deliberately.

    Examples
    --------
    Confirm the blocking step and continue::

        result = run_plan(session, plan, registry)
        if result.requires_confirmation_at is not None:
            result = run_plan(
                session, plan, registry,
                confirmations={result.requires_confirmation_at: True},
            )

    See Also
    --------
    run_plan : Produces this.
    PlanStepExecution : One entry.
    """

    plan: PlanResult
    step_executions: tuple[PlanStepExecution, ...]
    completed_steps: int
    total_steps: int
    stopped_at_step: int | None = None
    stop_reason: str = ""
    requires_confirmation_at: int | None = None
    all_confirmed: bool = False
    all_executed: bool = False
    egress_manifest: EgressManifest | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the run as JSON-safe values.

        Includes the plan, so a logged run records what was attempted as well
        as what happened.

        Returns
        -------
        dict
            Plan, per-step outcomes, completion counts, stop position and
            reason, the confirmation index, the two aggregate flags, and token
            usage.
        """
        return {
            "plan": self.plan.to_dict(),
            "step_executions": [s.to_dict() for s in self.step_executions],
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "stopped_at_step": self.stopped_at_step,
            "stop_reason": self.stop_reason,
            "requires_confirmation_at": self.requires_confirmation_at,
            "all_confirmed": self.all_confirmed,
            "all_executed": self.all_executed,
            "usage": dict(self.usage),
        }


_OPERATION_TO_TOOL: dict[str, tuple[str, dict[str, Any]]] = {
    "set_roles": ("set_roles", {}),
    "split": ("split", {}),
    "impute": ("impute", {}),
    "encode": ("encode", {}),
    "scale": ("scale", {}),
    "fit": ("fit", {}),
    "evaluate": ("evaluate", {}),
    "eda": ("eda_summary", {}),
    "describe": ("describe_dataset", {}),
    "workflow": ("workflow_status", {}),
    "explain": ("explain_operation", {}),
    "learn": ("learn_concept", {}),
    "rag_retrieve": ("rag_retrieve", {}),
    "rag_generate": ("rag_generate", {}),
    "rag_ingest_corpus": ("rag_ingest_corpus", {}),
    "rag_embed_and_index": ("rag_embed_and_index", {}),
    "make_torch_loaders": ("make_torch_loaders", {}),
    "make_text_torch_loaders": ("make_text_torch_loaders", {}),
    "make_multimodal_torch_loaders": ("make_multimodal_torch_loaders", {}),
    "make_image_multimodal_torch_loaders": ("make_image_multimodal_torch_loaders", {}),
    "make_audio_multimodal_torch_loaders": ("make_audio_multimodal_torch_loaders", {}),
    "make_speech_torch_loaders": ("make_speech_torch_loaders", {}),
    "fit_speech_torch": ("fit_speech_torch", {}),
    "transcribe_speech": ("transcribe_speech", {}),
    "load_pretrained_backbone": ("load_pretrained_backbone", {}),
    "pack_torchserve": ("pack_torchserve", {}),
    "prepare_tensorrt_export": ("prepare_tensorrt_export", {}),
    "emit_k8s_ddp_job": ("emit_k8s_ddp_job", {}),
    "emit_k8s_serve_deployment": ("emit_k8s_serve_deployment", {}),
    "domain_adapt_speech_torch": ("domain_adapt_speech_torch", {}),
    "attach_backbone_head": ("attach_backbone_head", {}),
    "evaluate_asr": ("evaluate_asr", {}),
    "fit_torch": ("fit_torch", {}),
    "evaluate_torch": ("evaluate_torch", {}),
    "cross_validate_torch": ("cross_validate_torch", {}),
    "search_torch": ("search_torch", {}),
    "nested_cv_torch": ("nested_cv_torch", {}),
    "export_torch": ("export_torch", {}),
}


def map_plan_step_to_tool(
    step: PlanStep,
    registry: ToolRegistry,
) -> tuple[str, dict[str, Any]] | None:
    """Find the registered tool a plan step is asking for.

    A model writes ``'split'``, ``'split_data'``, or ``'Split the data'`` for
    the same intent. Matching normalises case and separators and accepts a
    substring match in either direction, then falls back to an exact registry
    name.

    Parameters
    ----------
    step:
        The plan step. Its ``parameters`` override the mapped defaults.
    registry:
        The allowlist. A mapping that resolves to an unregistered tool is not
        used.

    Returns
    -------
    tuple of (str, dict) or None
        The tool name and merged arguments, or ``None`` when nothing matched.

    Notes
    -----
    **Substring matching is lenient in both directions**, which is what makes
    it tolerant of a model's phrasing and also what makes it capable of a wrong
    match on a short or unusual operation name. The first match in the mapping
    table wins, so order there decides ambiguous cases.

    **``None`` is the safe outcome.** An unmatched step is skipped and
    reported, never guessed into something adjacent.

    See Also
    --------
    build_step_proposals : Applies this across a plan.
    """
    op = step.operation.lower().replace("_", "").replace("-", "")
    step_params = dict(getattr(step, "parameters", None) or {})

    for key, (tool_name, default_args) in _OPERATION_TO_TOOL.items():
        if key.replace("_", "") in op or op in key.replace("_", ""):
            if tool_name in registry:
                merged = dict(default_args)
                merged.update(step_params)
                return tool_name, merged

    if step.operation in registry:
        return step.operation, step_params

    return None


def build_step_proposals(
    plan: PlanResult,
    registry: ToolRegistry,
    *,
    skip_unmapped: bool = True,
) -> list[tuple[PlanStep, ExecutorProposal | None, str]]:
    """Turn every step of a plan into a validated proposal, or a reason it is not.

    Runs before anything executes, so a plan referencing tools you have not
    allowed is visible up front rather than discovered halfway through.

    Parameters
    ----------
    plan:
        The plan to prepare.
    registry:
        The allowlist.
    skip_unmapped:
        Record unmapped steps as skipped and carry on. Set ``False`` to refuse
        the whole plan when any step cannot be mapped.

    Returns
    -------
    list of tuple
        ``(step, proposal or None, skip_reason)`` per step. The reason is empty
        when a proposal was built.

    Raises
    ------
    ValidationError
        If ``skip_unmapped`` is false and a step maps to nothing. The message
        lists the available tools.

    Notes
    -----
    **Argument validation happens here too.** A step whose parameters fail the
    tool's schema comes back with no proposal and the validation message as its
    reason, alongside the genuinely unmapped ones.

    **Skipping is silent progress, which can mislead.** A plan where half the
    steps were skipped reports no errors and achieved half of what it
    described. Read the reasons before reading the outcome.

    See Also
    --------
    map_plan_step_to_tool : The matching.
    run_plan : Preparing and executing in one call.
    """
    proposals: list[tuple[PlanStep, ExecutorProposal | None, str]] = []

    for step in plan.steps:
        mapping = map_plan_step_to_tool(step, registry)

        if mapping is None:
            if skip_unmapped:
                proposals.append((step, None, f"Operation '{step.operation}' not in tool registry"))
            else:
                raise ValidationError(
                    f"Plan step '{step.operation}' is not mapped to an allowed tool. "
                    f"Available tools: {sorted(t.name for t in registry.tools)}"
                )
            continue

        tool_name, default_args = mapping
        try:
            proposal = propose_tool_execution(tool_name, default_args, registry)
            proposals.append((step, proposal, ""))
        except ValidationError as e:
            proposals.append((step, None, str(e)))

    return proposals


def run_plan_step(
    session: Any,
    step: PlanStep,
    proposal: ExecutorProposal,
    registry: ToolRegistry,
    *,
    confirm: bool = False,
    auto_confirm_read_only: bool = True,
) -> PlanStepExecution:
    """Run one step, if it is permitted to run.

    Checks the tool's flags first. A destructive tool without explicit
    confirmation is refused outright. A tool needing confirmation that has none
    returns blocked. Otherwise it executes.

    Parameters
    ----------
    session:
        What to execute against.
    step:
        The plan step, used for its operation name in the outcome.
    proposal:
        The validated call.
    registry:
        The allowlist, consulted for the tool's flags.
    confirm:
        Explicit approval for this step.
    auto_confirm_read_only:
        Treat read-only tools as approved. Reasonable, since they cannot change
        anything: though they can still disclose.

    Returns
    -------
    PlanStepExecution
        The outcome, with ``step_index`` left at 0 for the caller to set.

    Notes
    -----
    **``auto_confirm_read_only`` does not cover destructive tools.** Nothing
    does. A destructive tool requires ``confirm=True`` and returns an error
    without it.

    **Execution failures are caught, not raised.** They become the ``error``
    field, so one bad step does not abort the surrounding run. Messages are
    scrubbed of credentials and truncated first.

    See Also
    --------
    run_plan : The loop around this.
    buildml.ai.executor.execute_tool : The execution itself.
    """
    spec = registry.get(proposal.tool_call.tool_name)
    is_read_only = spec is not None and spec.read_only
    is_destructive = spec is not None and spec.destructive

    should_confirm = confirm
    if is_read_only and auto_confirm_read_only:
        should_confirm = True

    if is_destructive:
        if not confirm:
            return PlanStepExecution(
                step_index=0,
                operation=step.operation,
                proposal=proposal,
                requires_confirmation=True,
                confirmed=False,
                executed=False,
                error="Destructive operations always require explicit confirmation.",
            )

    if proposal.requires_confirmation and not should_confirm:
        return PlanStepExecution(
            step_index=0,
            operation=step.operation,
            proposal=proposal,
            requires_confirmation=True,
            confirmed=False,
            executed=False,
        )

    try:
        result = execute_tool(session, proposal, should_confirm, registry)
        return PlanStepExecution(
            step_index=0,
            operation=step.operation,
            proposal=proposal,
            result=result,
            requires_confirmation=proposal.requires_confirmation,
            confirmed=result.confirmed,
            executed=result.executed,
            error=result.error,
        )
    except Exception as e:
        error_msg = _redact_exception_message(str(e))
        return PlanStepExecution(
            step_index=0,
            operation=step.operation,
            proposal=proposal,
            requires_confirmation=proposal.requires_confirmation,
            confirmed=should_confirm,
            executed=False,
            error=f"Execution failed: {error_msg}",
        )


def run_plan(
    session: Any,
    plan: PlanResult,
    registry: ToolRegistry,
    *,
    confirmations: dict[int, bool] | None = None,
    auto_confirm_read_only: bool = True,
    stop_on_error: bool = True,
    stop_on_unconfirmed: bool = True,
    max_steps: int | None = None,
) -> PlanExecutionResult:
    """Execute a plan, stopping the moment something needs you.

    Prepares every step, then walks them in order. By default the run halts at
    the first step requiring approval it does not have, reporting where: you
    supply the confirmation and call again.

    Parameters
    ----------
    session:
        What to execute against.
    plan:
        The plan.
    registry:
        The allowlist.
    confirmations:
        Step index to approval. Steps absent from the mapping are treated as
        unconfirmed, so an incomplete mapping pauses rather than proceeds.
    auto_confirm_read_only:
        Treat read-only steps as approved.
    stop_on_error:
        Halt at the first failure. When false, the run continues and the
        failures accumulate in the outcomes.
    stop_on_unconfirmed:
        Halt at the first unapproved step. When false, such steps are recorded
        as blocked and the run continues: useful for seeing everything a plan
        would ask for in one pass.
    max_steps:
        Cap on how many steps may execute. Counts executions, not iterations,
        so skipped and blocked steps do not consume it.

    Returns
    -------
    PlanExecutionResult
        Per-step outcomes, how far it got, and why it stopped.

    Raises
    ------
    ValidationError
        If preparing the proposals fails.

    Notes
    -----
    **Resuming re-runs the plan from the beginning.** There is no partial
    resume: steps that already executed will execute again. For anything not
    idempotent, confirm the remaining steps in one pass rather than iterating.

    **``stop_on_unconfirmed=False`` is the way to preview a plan.** Every step
    reports what it would need, and nothing unapproved runs.

    **Failures are per-step, not exceptions.** Even with ``stop_on_error``, the
    error is in the outcome rather than raised.

    Examples
    --------
    Preview, then approve the whole plan::

        preview = run_plan(
            session, plan, registry, stop_on_unconfirmed=False,
        )
        needed = {
            e.step_index: True
            for e in preview.step_executions
            if e.requires_confirmation
        }
        result = run_plan(session, plan, registry, confirmations=needed)

    See Also
    --------
    run_plan_step : One step.
    PlanExecutionResult : Reading the outcome.
    """
    confirmations = confirmations or {}
    proposals = build_step_proposals(plan, registry)
    executions: list[PlanStepExecution] = []
    completed = 0
    stopped_at: int | None = None
    stop_reason = ""
    requires_confirm_at: int | None = None

    for idx, (step, proposal, skip_reason) in enumerate(proposals):
        if max_steps is not None and completed >= max_steps:
            stopped_at = idx
            stop_reason = f"Max steps ({max_steps}) reached."
            break

        if proposal is None:
            executions.append(PlanStepExecution(
                step_index=idx,
                operation=step.operation,
                skipped=True,
                skip_reason=skip_reason,
            ))
            continue

        confirm_this_step = confirmations.get(idx, False)
        spec = registry.get(proposal.tool_call.tool_name)
        is_read_only = spec is not None and spec.read_only

        if is_read_only and auto_confirm_read_only:
            confirm_this_step = True

        execution = run_plan_step(
            session,
            step,
            proposal,
            registry,
            confirm=confirm_this_step,
            auto_confirm_read_only=auto_confirm_read_only,
        )
        execution.step_index = idx
        executions.append(execution)

        if execution.requires_confirmation and not execution.confirmed:
            if stop_on_unconfirmed:
                stopped_at = idx
                stop_reason = f"Step {idx} ('{step.operation}') requires confirmation."
                requires_confirm_at = idx
                break
        elif execution.error:
            if stop_on_error:
                stopped_at = idx
                stop_reason = f"Step {idx} error: {execution.error}"
                break
        elif execution.executed:
            completed += 1

    all_confirmed = all(
        e.confirmed or e.skipped or (not e.requires_confirmation)
        for e in executions
    )
    all_executed = all(
        e.executed or e.skipped
        for e in executions
    )

    return PlanExecutionResult(
        plan=plan,
        step_executions=tuple(executions),
        completed_steps=completed,
        total_steps=len(plan.steps),
        stopped_at_step=stopped_at,
        stop_reason=stop_reason,
        requires_confirmation_at=requires_confirm_at,
        all_confirmed=all_confirmed,
        all_executed=all_executed,
        egress_manifest=plan.egress_manifest,
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


class BudgetExceeded(ValidationError):
    """Raised when a run has spent its allowance.

    Agent loops cost money in proportion to how long they run, and a loop that
    fails to converge can run a long time. This is what the ceiling raises when
    it is crossed.

    Attributes
    ----------
    budget_type:
        Which ceiling: tokens or cost.
    limit:
        The ceiling.
    used:
        What had been spent. Exceeds the limit, since the check happens after
        the spend rather than before it.

    Notes
    -----
    A subclass of :class:`~buildml.core.errors.ValidationError`, catchable
    either as that or specifically.

    See Also
    --------
    BudgetTracker : Where it is raised.
    BudgetTracker.can_proceed : Checking before spending instead.
    """

    def __init__(
        self,
        budget_type: str,
        limit: int | float,
        used: int | float,
    ) -> None:
        """Build the error from the ceiling and the spend.

        All three values are kept as attributes as well as formatted into the
        message, so a handler can act on the numbers without parsing text.

        Parameters
        ----------
        budget_type:
            Which ceiling was crossed, as it should read in the message.
        limit:
            The ceiling.
        used:
            What had been spent.
        """
        self.budget_type = budget_type
        self.limit = limit
        self.used = used
        super().__init__(
            f"{budget_type} budget exceeded: used {used}, limit {limit}. "
            "Increase budget or reduce scope."
        )


@dataclass(slots=True)
class BudgetTracker:
    """Keep a running total of what an AI session has spent.

    Records usage as it accrues and raises once a ceiling is crossed. Both
    ceilings are optional; leaving one ``None`` means it is not enforced.

    Attributes
    ----------
    max_tokens:
        Token ceiling. ``None`` for no limit.
    max_cost_usd:
        Cost ceiling in dollars. ``None`` for no limit.
    tokens_used:
        Running total.
    cost_used_usd:
        Running total.

    Notes
    -----
    **The check happens after the spend.** :meth:`record_usage` adds first and
    raises second, so the tracker stops the *next* call rather than the one that
    crossed the line. Use :meth:`can_proceed` to check before committing.

    **Cost is whatever you record.** Nothing here knows provider pricing;
    ``cost_usd`` is supplied by the caller, and a tracker fed only token counts
    tracks only tokens.

    Examples
    --------
    Cap a session and check before each call::

        budget = BudgetTracker(max_tokens=50_000, max_cost_usd=1.0)
        if budget.can_proceed(estimated_tokens=2_000):
            response = provider.chat(messages)
            budget.record_usage(
                response.usage["total_tokens"], operation="plan",
            )

    See Also
    --------
    BudgetExceeded : What it raises.
    buildml.ai.security.check_iteration_limit : The turn-count bound.
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    tokens_used: int = 0
    cost_used_usd: float = 0.0
    _history: list[dict[str, Any]] = field(default_factory=list)

    def record_usage(
        self,
        tokens: int,
        cost_usd: float = 0.0,
        operation: str = "",
    ) -> None:
        """Add a call's usage to the totals, then enforce the ceilings.

        Appends a timestamped entry to the internal history before checking, so
        the spend is recorded even on the call that breaches the limit: a
        budget that lost its last entry on failure would not add up.

        Parameters
        ----------
        tokens:
            Tokens consumed.
        cost_usd:
            Cost in dollars, computed by the caller from provider pricing.
        operation:
            What the spend was for, recorded in the history.

        Returns
        -------
        None
            Returns nothing on success; the value is the absence of an
            exception.

        Raises
        ------
        BudgetExceeded
            If either ceiling is now exceeded.

        Notes
        -----
        **The spend has already happened when this raises.** The call was made
        and the tokens are gone; the exception prevents the next one.

        See Also
        --------
        can_proceed : Checking beforehand.
        """
        self.tokens_used += tokens
        self.cost_used_usd += cost_usd

        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "tokens": tokens,
            "cost_usd": cost_usd,
            "total_tokens": self.tokens_used,
            "total_cost_usd": self.cost_used_usd,
        })

        self.check_limits()

    def check_limits(self) -> None:
        """Enforce the ceilings against the current totals.

        Called automatically by :meth:`record_usage`. Call it directly to
        enforce after adjusting the totals or the limits by hand.

        Returns
        -------
        None
            Returns nothing on success; the value is the absence of an
            exception.

        Raises
        ------
        BudgetExceeded
            If either total has passed its ceiling. Tokens are checked first.

        Notes
        -----
        The comparison is strict, so spending exactly the budget is permitted.

        See Also
        --------
        can_proceed : The non-raising form.
        """
        if self.max_tokens is not None and self.tokens_used > self.max_tokens:
            raise BudgetExceeded("Token", self.max_tokens, self.tokens_used)
        if self.max_cost_usd is not None and self.cost_used_usd > self.max_cost_usd:
            raise BudgetExceeded("Cost (USD)", self.max_cost_usd, self.cost_used_usd)

    def can_proceed(self, estimated_tokens: int = 0, estimated_cost: float = 0.0) -> bool:
        """Ask whether a call would fit inside the remaining budget.

        The way to stop before overspending rather than after. Adds the
        estimate to the current totals and compares, without recording
        anything.

        Parameters
        ----------
        estimated_tokens:
            Expected token cost.
        estimated_cost:
            Expected dollar cost.

        Returns
        -------
        bool
            True when the call would fit under both ceilings.

        Notes
        -----
        **Only as good as the estimate.** Prompt tokens are countable in
        advance; completion tokens are not, so estimates skew low. Budget with
        headroom.

        **The comparison is strict here too**, so a call landing exactly on the
        ceiling is permitted.

        See Also
        --------
        remaining_tokens : What is left.
        record_usage : Recording the actual spend.
        """
        if self.max_tokens is not None:
            if self.tokens_used + estimated_tokens > self.max_tokens:
                return False
        if self.max_cost_usd is not None:
            if self.cost_used_usd + estimated_cost > self.max_cost_usd:
                return False
        return True

    def remaining_tokens(self) -> int | None:
        """Return how many tokens are left.

        For displaying progress against a ceiling, and for deciding whether a
        further call is worth attempting.

        Returns
        -------
        int or None
            The remainder, floored at zero, or ``None`` when no token ceiling
            is set.

        Notes
        -----
        **``None`` means unlimited, not exhausted.** Distinguish it from ``0``
        before treating it as a stop signal.

        See Also
        --------
        remaining_cost_usd : The same for money.
        """
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens_used)

    def remaining_cost_usd(self) -> float | None:
        """Return how much money is left.

        The cost counterpart to :meth:`remaining_tokens`, and the figure worth
        surfacing to whoever is paying.

        Returns
        -------
        float or None
            The remainder in dollars, floored at zero, or ``None`` when no cost
            ceiling is set.

        Notes
        -----
        Reflects only what has been recorded. A tracker fed token counts
        without costs reports its full cost budget however much has been spent.

        See Also
        --------
        remaining_tokens : The same for tokens.
        """
        if self.max_cost_usd is None:
            return None
        return max(0.0, self.max_cost_usd - self.cost_used_usd)

    def to_dict(self) -> dict[str, Any]:
        """Return the budget state as JSON-safe values.

        Reports the ceilings, the totals, and both remainders, so a logged
        state answers what was spent and what was left without recomputation.

        Returns
        -------
        dict
            Ceilings, totals, remainders, and the number of recorded entries.

        Notes
        -----
        The history itself is summarised to a count rather than included, since
        it grows with every call and adds nothing the totals do not already
        say.
        """
        return {
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "tokens_used": self.tokens_used,
            "cost_used_usd": self.cost_used_usd,
            "remaining_tokens": self.remaining_tokens(),
            "remaining_cost_usd": self.remaining_cost_usd(),
            "history_count": len(self._history),
        }
