"""Explicit autonomy mode for the AI operator (plan-and-execute with safeguards).

Default Session AI remains propose→confirm→execute. This module implements an
opt-in loop that can auto-confirm allowlisted tools under hard caps:

- max steps / max iterations
- tool allowlist (subset of the registry)
- egress level unchanged (FULL_SAMPLE still requires prior confirm policy)
- destructive tools still require explicit ``allow_destructive=True``
- full transcript audit of every auto-confirmed step

This is **operator automation**, not unconstrained agency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.planner import map_plan_step_to_tool
from buildml.ai.privacy import EgressLevel, EgressManifest
from buildml.ai.results import PlanResult
from buildml.ai.security import (
    MaxIterationsExceeded,
    check_iteration_limit,
    validate_tool_call_safety,
)
from buildml.ai.tools import ToolRegistry, build_default_registry
from buildml.ai.types import Message
from buildml.core.errors import ValidationError

DEFAULT_AUTONOMY_MAX_STEPS = 8
DEFAULT_AUTONOMY_ALLOWLIST = (
    "describe_dataset",
    "workflow_status",
    "explain_operation",
    "ai_status",
    "eda_summary",
    "set_roles",
    "split",
    "impute",
    "encode",
    "scale",
    "fit",
    "evaluate",
    "make_torch_loaders",
    "make_text_torch_loaders",
    "make_multimodal_torch_loaders",
    "make_image_multimodal_torch_loaders",
    "make_audio_multimodal_torch_loaders",
    "fit_torch",
    "evaluate_torch",
    "cross_validate_torch",
    "search_torch",
    "nested_cv_torch",
    "rag_retrieve",
    "rag_generate",
    "rag_ingest_corpus",
    "rag_embed_and_index",
)


@dataclass(slots=True)
class AutonomyConfig:
    """Hard controls for :func:`run_autonomous`."""

    max_steps: int = DEFAULT_AUTONOMY_MAX_STEPS
    tool_allowlist: tuple[str, ...] = DEFAULT_AUTONOMY_ALLOWLIST
    allow_destructive: bool = False
    stop_on_error: bool = True
    require_explicit: bool = True
    egress_levels_blocked: tuple[str, ...] = (
        EgressLevel.FULL_SAMPLE.value,
        EgressLevel.REDACTED_SAMPLE.value,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "tool_allowlist": list(self.tool_allowlist),
            "allow_destructive": self.allow_destructive,
            "stop_on_error": self.stop_on_error,
            "require_explicit": self.require_explicit,
            "egress_levels_blocked": list(self.egress_levels_blocked),
        }


@dataclass(slots=True)
class AutonomyStepRecord:
    """Audit record for one autonomous step."""

    step_index: int
    tool_name: str
    arguments: dict[str, Any]
    auto_confirmed: bool
    executed: bool
    skipped: bool = False
    skip_reason: str = ""
    error: str | None = None
    result_summary: str = ""
    safety_warnings: tuple[str, ...] = ()
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "auto_confirmed": self.auto_confirmed,
            "executed": self.executed,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "result_summary": self.result_summary,
            "safety_warnings": list(self.safety_warnings),
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class AutonomyResult:
    """Outcome of an autonomous plan-and-execute run."""

    goal: str
    plan: PlanResult | None
    steps: tuple[AutonomyStepRecord, ...]
    completed_steps: int
    total_steps: int
    stopped_at_step: int | None
    stop_reason: str
    config: AutonomyConfig
    disclosures: tuple[str, ...]
    limitations: tuple[str, ...]
    residual_risks: tuple[str, ...]
    egress_manifest: EgressManifest | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "stopped_at_step": self.stopped_at_step,
            "stop_reason": self.stop_reason,
            "config": self.config.to_dict(),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "residual_risks": list(self.residual_risks),
            "usage": dict(self.usage),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blocked_egress(session: Any, config: AutonomyConfig) -> str | None:
    egress = getattr(session, "_ai_egress_config", None)
    if egress is None:
        return None
    level = getattr(egress.level, "value", str(egress.level))
    if level in config.egress_levels_blocked:
        return (
            f"Autonomy refused: egress level {level!r} can send sample/raw data to the "
            "provider. Lower egress to stats_only/schema_only or use confirmed "
            "ai_plan/ai_execute instead of ai_run_autonomous."
        )
    return None


def _filter_allowlist(registry: ToolRegistry, allowlist: tuple[str, ...]) -> ToolRegistry:
    allowed = set(allowlist)
    specs = [t for t in registry.tools if t.name in allowed]
    if not specs:
        raise ValidationError("Autonomy tool_allowlist matched no registry tools")
    return ToolRegistry(tuple(specs))


def run_autonomous(
    session: Any,
    goal: str,
    *,
    plan: PlanResult | None = None,
    config: AutonomyConfig | None = None,
    registry: ToolRegistry | None = None,
    confirm_autonomy: bool = False,
    provider_plan: bool = True,
) -> AutonomyResult:
    """Plan (optional) and auto-execute allowlisted tools under hard safety caps.

    Parameters
    ----------
    goal:
        Operator goal string (recorded in the audit transcript).
    plan:
        Optional precomputed :class:`PlanResult`. When omitted and
        ``provider_plan=True``, calls the advisor planner once.
    confirm_autonomy:
        **Required True** — explicit opt-in. Default False refuses to run.
    config:
        Caps and allowlist. Defaults are conservative.
    """
    cfg = config or AutonomyConfig()
    if cfg.require_explicit and not confirm_autonomy:
        raise ValidationError(
            "ai_run_autonomous requires confirm_autonomy=True. "
            "Default Session AI stays propose→confirm→execute; autonomy is an "
            "explicit operator automation mode with residual risk."
        )
    if cfg.max_steps < 1:
        raise ValidationError("max_steps must be >= 1")

    egress_block = _blocked_egress(session, cfg)
    if egress_block:
        raise ValidationError(egress_block)

    base_registry = registry or getattr(session, "_ai_registry", None) or build_default_registry()
    auto_registry = _filter_allowlist(base_registry, cfg.tool_allowlist)

    resolved_plan = plan
    usage: dict[str, int] = {}
    if resolved_plan is None and provider_plan:
        from buildml.ai.advisor import run_plan

        provider = getattr(session, "_ai_provider", None)
        if provider is None:
            raise ValidationError("No AI provider configured. Call ai_configure() first.")
        egress_config = getattr(session, "_ai_egress_config", None)
        resolved_plan = run_plan(session, goal, provider, egress_config=egress_config)
        if resolved_plan.usage:
            usage.update(resolved_plan.usage)

    if resolved_plan is None:
        raise ValidationError("No plan available; pass plan= or enable provider_plan=True")

    records: list[AutonomyStepRecord] = []
    completed = 0
    stopped_at: int | None = None
    stop_reason = "completed"
    max_iter = int(getattr(session, "_ai_max_iterations", cfg.max_steps) or cfg.max_steps)

    for idx, step in enumerate(resolved_plan.steps):
        try:
            check_iteration_limit(idx, min(cfg.max_steps, max_iter), tool_name=step.operation)
        except MaxIterationsExceeded as exc:
            stopped_at = idx
            stop_reason = str(exc)
            break
        if completed >= cfg.max_steps:
            stopped_at = idx
            stop_reason = f"Max autonomous steps ({cfg.max_steps}) reached."
            break

        mapping = map_plan_step_to_tool(step, auto_registry)
        if mapping is None:
            records.append(
                AutonomyStepRecord(
                    step_index=idx,
                    tool_name=step.operation,
                    arguments=dict(getattr(step, "parameters", None) or {}),
                    auto_confirmed=False,
                    executed=False,
                    skipped=True,
                    skip_reason=f"Operation '{step.operation}' not in autonomy allowlist/registry",
                    timestamp=_now(),
                )
            )
            continue

        tool_name, args = mapping
        spec = auto_registry.get(tool_name)
        if spec is None:
            records.append(
                AutonomyStepRecord(
                    step_index=idx,
                    tool_name=tool_name,
                    arguments=args,
                    auto_confirmed=False,
                    executed=False,
                    skipped=True,
                    skip_reason="Tool missing from filtered registry",
                    timestamp=_now(),
                )
            )
            continue

        if spec.destructive and not cfg.allow_destructive:
            records.append(
                AutonomyStepRecord(
                    step_index=idx,
                    tool_name=tool_name,
                    arguments=args,
                    auto_confirmed=False,
                    executed=False,
                    skipped=True,
                    skip_reason="Destructive tool blocked (allow_destructive=False)",
                    timestamp=_now(),
                )
            )
            stopped_at = idx
            stop_reason = records[-1].skip_reason
            break

        proposal = propose_tool_execution(tool_name, args, auto_registry)
        safety = validate_tool_call_safety(proposal.tool_call)
        # Autonomy auto-confirms allowlisted non-blocked tools.
        try:
            result = execute_tool(session, proposal, True, auto_registry)
            records.append(
                AutonomyStepRecord(
                    step_index=idx,
                    tool_name=tool_name,
                    arguments=args,
                    auto_confirmed=True,
                    executed=bool(result.executed),
                    error=result.error,
                    result_summary=str(result.result_summary or ""),
                    safety_warnings=tuple(safety),
                    timestamp=_now(),
                )
            )
            transcript = getattr(session, "_ai_transcript", None)
            if transcript is not None:
                transcript.add_tool_call(proposal.tool_call, confirmed=True)
                if result.executed:
                    transcript.add_tool_result(proposal.tool_call, result.result_summary)
                if result.error:
                    transcript.add_error(result.error, proposal.tool_call)
            if result.error and cfg.stop_on_error:
                stopped_at = idx
                stop_reason = f"Step {idx} error: {result.error}"
                break
            if result.executed:
                completed += 1
        except Exception as exc:  # noqa: BLE001
            records.append(
                AutonomyStepRecord(
                    step_index=idx,
                    tool_name=tool_name,
                    arguments=args,
                    auto_confirmed=True,
                    executed=False,
                    error=str(exc),
                    safety_warnings=tuple(safety),
                    timestamp=_now(),
                )
            )
            if cfg.stop_on_error:
                stopped_at = idx
                stop_reason = f"Step {idx} exception: {exc}"
                break

    disclosures = (
        "Autonomy mode auto-confirmed allowlisted tools without per-step human confirm.",
        f"Hard cap max_steps={cfg.max_steps}; allowlist size={len(cfg.tool_allowlist)}.",
        "Sample/raw egress levels remain blocked in this mode.",
    )
    limitations = (
        "This is operator automation inside an allowlist — not an unconstrained agent.",
        "Planner quality depends on the provider; MockProvider is for CI only.",
        "Auto-confirm does not imply the workflow is production-safe or leakage-free.",
    )
    residual_risks = (
        "A compromised or confused planner can still run any allowlisted write tool "
        "(split/fit/impute/…). Keep allowlists tight and review transcripts.",
        "Injection defenses are best-effort pattern filters, not a formal sandbox.",
        "Do not point autonomy at production databases or irreplaceable artifacts.",
    )

    transcript = getattr(session, "_ai_transcript", None)
    if transcript is not None:
        transcript.add_message(Message(role="user", content=f"Autonomy goal: {goal}"))
        transcript.add_message(
            Message(
                role="assistant",
                content=(
                    f"Autonomy finished: {completed}/{len(resolved_plan.steps)} steps; "
                    f"stop_reason={stop_reason}"
                ),
            )
        )

    return AutonomyResult(
        goal=goal,
        plan=resolved_plan,
        steps=tuple(records),
        completed_steps=completed,
        total_steps=len(resolved_plan.steps),
        stopped_at_step=stopped_at,
        stop_reason=stop_reason,
        config=cfg,
        disclosures=disclosures,
        limitations=limitations,
        residual_risks=residual_risks,
        egress_manifest=getattr(resolved_plan, "egress_manifest", None),
        usage=usage,
    )


def autonomy_status_dict(result: AutonomyResult | None) -> dict[str, Any]:
    """Compact status fragment for ``ai_status`` disclosures."""
    if result is None:
        return {
            "autonomy_enabled_last_run": False,
            "note": "Default AI path remains propose→confirm→execute.",
        }
    return {
        "autonomy_enabled_last_run": True,
        "completed_steps": result.completed_steps,
        "total_steps": result.total_steps,
        "stop_reason": result.stop_reason,
        "residual_risks": list(result.residual_risks),
    }
