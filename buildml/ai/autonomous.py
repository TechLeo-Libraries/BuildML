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
    "learn_concept",
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
    "make_speech_torch_loaders",
    "fit_speech_torch",
    "transcribe_speech",
    "load_pretrained_backbone",
    "pack_torchserve",
    "prepare_tensorrt_export",
    "emit_k8s_ddp_job",
    "emit_k8s_serve_deployment",
    "domain_adapt_speech_torch",
    "attach_backbone_head",
    "evaluate_asr",
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
    """The bounds an autonomous run may not exceed.

    Every field narrows what can happen. The defaults are deliberately
    restrictive, and widening any of them is a decision worth making
    explicitly.

    Attributes
    ----------
    max_steps:
        How many operations may execute. Bounds both cost and how far a wrong
        plan gets before you see it.
    tool_allowlist:
        Which tools may run unattended. A subset of the registry, and the
        primary control: a tool absent from this list cannot be
        auto-confirmed however the plan is phrased.
    allow_destructive:
        Whether destructive tools may run at all. **Off by default, and
        deliberately awkward to turn on.**
    stop_on_error:
        Halt at the first failure. Usually right: a plan's later steps assume
        the earlier ones worked.
    require_explicit:
        Require ``confirm_autonomy=True`` at the call site. Leave on: it is
        what stops autonomy from being entered by accident.
    egress_levels_blocked:
        Levels under which autonomy refuses to run. Sample levels are blocked
        by default, since unattended execution and raw-row disclosure are a bad
        combination.

    Notes
    -----
    **The allowlist is the control that matters most.** Step caps limit how
    much happens; the allowlist limits what can happen at all.

    Examples
    --------
    Read-only exploration, unattended::

        config = AutonomyConfig(
            max_steps=4,
            tool_allowlist=("describe_dataset", "eda_summary", "workflow_status"),
        )

    See Also
    --------
    run_autonomous : What enforces this.
    """

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
        """Return the autonomy bounds as JSON-safe values.

        Records the constraints a run operated under, which is what makes the
        audit trail meaningful: the record of what happened means little
        without the record of what was permitted.

        Returns
        -------
        dict
            Step cap, allowlist, the destructive and explicit-confirmation
            flags, error behaviour, and blocked egress levels.
        """
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
    """What one unattended step did, recorded for later review.

    Nobody watched this happen, so the record has to be complete enough to
    reconstruct it afterwards.

    Attributes
    ----------
    step_index:
        Position in the plan.
    tool_name:
        Which tool ran, or would have.
    arguments:
        With what arguments. **Kept in full**: a summary would defeat the
        purpose of an audit record.
    auto_confirmed:
        Whether it was approved without a person.
    executed:
        Whether it ran.
    skipped:
        Whether it was passed over.
    skip_reason:
        Why: unmapped operation, or not on the allowlist.
    error:
        What went wrong.
    result_summary:
        A short form of what came back.
    safety_warnings:
        What the safety checks noticed. **Worth reviewing even on steps that
        succeeded.**
    timestamp:
        When, as an ISO 8601 UTC string.

    See Also
    --------
    AutonomyResult : The containing run.
    """

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
        """Return the step record as JSON-safe values.

        Complete: arguments and safety warnings are kept in full, because a
        record of unattended execution that omits details is not much of a
        record.

        Returns
        -------
        dict
            Index, tool, arguments, the confirmation and execution flags, skip
            state and reason, error, result summary, safety warnings, and
            timestamp.
        """
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
    """What an unattended run did, and what it leaves you responsible for.

    Attributes
    ----------
    goal:
        What was asked for.
    plan:
        The plan executed, whether supplied or generated.
    steps:
        A record per step.
    completed_steps:
        How many ran.
    total_steps:
        How many the plan had.
    stopped_at_step:
        Where it halted. ``None`` when it reached the end.
    stop_reason:
        Why.
    config:
        The bounds it ran under.
    disclosures:
        What the mode does.
    limitations:
        What it does not do.
    residual_risks:
        **What remains your responsibility.** Operations ran without review;
        this is the list of what that means and is the part to read.
    egress_manifest:
        What was disclosed producing the plan.
    usage:
        Token counts.

    Notes
    -----
    **Completing every step is not evidence of a good outcome.** It means the
    operations were valid and their preconditions held. Whether the resulting
    pipeline is sound is a separate question, and unattended execution means
    nobody has looked at it yet.

    See Also
    --------
    run_autonomous : Produces this.
    AutonomyStepRecord : One step.
    """

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
        """Return the autonomous run as JSON-safe values.

        Includes the plan, every step record, and the configuration, so the
        stored account says what was permitted as well as what happened.

        Returns
        -------
        dict
            Goal, plan, step records, completion counts, stop position and
            reason, configuration, the three prose lists, and token usage.
        """
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
    """Execute a plan without stopping to ask, inside hard limits.

    The one path where operations run unattended. It exists because a long
    sequence of routine steps is tedious to confirm one at a time, and it is
    fenced accordingly: an explicit opt-in at the call site, a tool allowlist,
    a step cap, destructive tools off, sample egress levels refused, and a full
    audit record of everything that happened.

    Every step still passes the safety checks and still goes through the
    executor. What changes is who says yes.

    Parameters
    ----------
    session:
        What to execute against.
    goal:
        What you want done. Recorded in the audit trail, and used to generate a
        plan when none is supplied.
    plan:
        A plan to execute. When omitted and ``provider_plan`` is true, one is
        generated by calling the planner once.
    config:
        The bounds. Conservative by default.
    registry:
        The tool set, narrowed to the allowlist before anything runs.
    confirm_autonomy:
        **Must be true.** The opt-in that stops this mode being entered by
        accident.
    provider_plan:
        Allow generating a plan when none was supplied. Set false to require a
        plan you have already reviewed.

    Returns
    -------
    AutonomyResult
        Per-step records, how far it got, why it stopped, and the residual
        risks.

    Raises
    ------
    ValidationError
        If ``confirm_autonomy`` is false, if ``max_steps`` is below 1, if the
        egress level is one autonomy refuses to run under, if the allowlist
        matches no registered tool, or if no plan is available and generation
        is disabled.
    MaxIterationsExceeded
        If the step cap is reached mid-run.

    Notes
    -----
    **Reviewing the plan first is the safer pattern.** Generate it with
    :func:`buildml.ai.advisor.run_plan`, read it, then pass it here with
    ``provider_plan=False``. A plan generated inside the same call is executed
    without anyone having seen it.

    **Sample egress levels are refused outright.** Unattended execution and
    raw-row disclosure compound each other, so the combination is blocked
    rather than warned about.

    **Every step is recorded, including the skipped ones.** A run that
    completed few steps because most were off the allowlist looks successful
    until you read the records.

    **The residual risks are real.** Operations ran without review, and
    ``residual_risks`` says what that leaves you responsible for.

    Examples
    --------
    Review the plan, then run it unattended::

        plan = advisor.run_plan(session, "prepare the data for fitting", provider)
        result = run_autonomous(
            session,
            "prepare the data for fitting",
            plan=plan,
            provider_plan=False,
            confirm_autonomy=True,
            config=AutonomyConfig(max_steps=5),
        )
        result.residual_risks

    See Also
    --------
    buildml.ai.planner.run_plan : The confirmed alternative.
    AutonomyConfig : The bounds.
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
        "This is operator automation inside an allowlist: not an unconstrained agent.",
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
    """Summarise whether autonomy was used, for status surfaces.

    Folded into the walkthrough status so a Session that ran unattended says so
    prominently, rather than leaving it buried in a transcript.

    Parameters
    ----------
    result:
        The most recent autonomous run, or ``None`` when there was none.

    Returns
    -------
    dict
        When there was no run, a flag and a note that the default path requires
        confirmation. Otherwise the completion counts, the stop reason, and the
        residual risks.

    Notes
    -----
    **The residual risks are carried through deliberately.** They are the part
    of an autonomous run a reader most needs to see, and burying them in a
    result object nobody opens would defeat the purpose.

    See Also
    --------
    buildml.ai.explain_hooks.ai_status : Where this appears.
    """
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
