"""Multi-step planner for AI operator E2E workflow orchestration."""

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
    """Execution result for one step in a multi-step plan."""

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
    """Result from executing a multi-step plan."""

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
}


def map_plan_step_to_tool(
    step: PlanStep,
    registry: ToolRegistry,
) -> tuple[str, dict[str, Any]] | None:
    """Map a plan step operation to a tool name and default arguments.

    Returns None if the operation is not in the tool registry.
    """
    op = step.operation.lower().replace("_", "").replace("-", "")
    
    for key, (tool_name, default_args) in _OPERATION_TO_TOOL.items():
        if key.replace("_", "") in op or op in key.replace("_", ""):
            if tool_name in registry:
                return tool_name, dict(default_args)
    
    if step.operation in registry:
        return step.operation, {}
    
    return None


def build_step_proposals(
    plan: PlanResult,
    registry: ToolRegistry,
    *,
    skip_unmapped: bool = True,
) -> list[tuple[PlanStep, ExecutorProposal | None, str]]:
    """Build executor proposals for each step in a plan.

    Returns list of (step, proposal_or_none, skip_reason).
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
    """Execute a single plan step with confirmation gating.

    Parameters
    ----------
    session
        The Session object to execute against.
    step
        The plan step being executed.
    proposal
        The executor proposal for this step.
    registry
        Tool registry for validation.
    confirm
        If True, confirms and executes write operations.
    auto_confirm_read_only
        If True, auto-confirms read-only operations.

    Returns
    -------
    PlanStepExecution
        The execution result for this step.
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
    """Execute a multi-step plan with confirmation gating.

    Default behavior: pauses at the first step requiring confirmation that
    hasn't been confirmed. Caller must provide confirmations dict or set
    auto_confirm_read_only=False to require manual confirmation for everything.

    Parameters
    ----------
    session
        The Session object to execute against.
    plan
        The plan to execute.
    registry
        Tool registry for validation.
    confirmations
        Dict mapping step_index → True/False for confirmation decisions.
        Steps not in the dict use default confirmation behavior.
    auto_confirm_read_only
        If True, auto-confirms read-only operations.
    stop_on_error
        If True, stops execution on first error.
    stop_on_unconfirmed
        If True, stops at steps requiring confirmation that aren't confirmed.
    max_steps
        Maximum number of steps to execute (None = no limit).

    Returns
    -------
    PlanExecutionResult
        Combined result of the plan execution.
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
    """Raised when token or cost budget is exceeded."""

    def __init__(
        self,
        budget_type: str,
        limit: int | float,
        used: int | float,
    ) -> None:
        self.budget_type = budget_type
        self.limit = limit
        self.used = used
        super().__init__(
            f"{budget_type} budget exceeded: used {used}, limit {limit}. "
            "Increase budget or reduce scope."
        )


@dataclass(slots=True)
class BudgetTracker:
    """Track token and cost budgets across AI operations."""

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
        """Record token/cost usage and check limits."""
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
        """Raise BudgetExceeded if any limit is exceeded."""
        if self.max_tokens is not None and self.tokens_used > self.max_tokens:
            raise BudgetExceeded("Token", self.max_tokens, self.tokens_used)
        if self.max_cost_usd is not None and self.cost_used_usd > self.max_cost_usd:
            raise BudgetExceeded("Cost (USD)", self.max_cost_usd, self.cost_used_usd)

    def can_proceed(self, estimated_tokens: int = 0, estimated_cost: float = 0.0) -> bool:
        """Check if an operation can proceed without exceeding limits."""
        if self.max_tokens is not None:
            if self.tokens_used + estimated_tokens > self.max_tokens:
                return False
        if self.max_cost_usd is not None:
            if self.cost_used_usd + estimated_cost > self.max_cost_usd:
                return False
        return True

    def remaining_tokens(self) -> int | None:
        """Return remaining token budget, or None if unlimited."""
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens_used)

    def remaining_cost_usd(self) -> float | None:
        """Return remaining cost budget, or None if unlimited."""
        if self.max_cost_usd is None:
            return None
        return max(0.0, self.max_cost_usd - self.cost_used_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "tokens_used": self.tokens_used,
            "cost_used_usd": self.cost_used_usd,
            "remaining_tokens": self.remaining_tokens(),
            "remaining_cost_usd": self.remaining_cost_usd(),
            "history_count": len(self._history),
        }
