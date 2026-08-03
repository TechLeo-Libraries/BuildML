"""Thin Session facades over buildml.ai (confirm-by-default; autonomy opt-in)."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


def ai_configure(
    session,
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_key_env: str = "BUILDML_OPENAI_API_KEY",
    egress_level: str = "stats_only",
    max_iterations: int = 10,
    max_tokens: int | None = None,
    max_cost_usd: float | None = None,
) -> Session:
    """Configure an AI provider for LLM-assisted workflow guidance.

    API keys are read from environment variables by default. Keys are never

    logged, persisted in transcripts/checkpoints, or echoed in errors.

    Parameters
    ----------
        provider
        Provider name (currently ``"openai"`` for OpenAI-compatible APIs,
        or ``"mock"`` for CI testing without real keys).
        model
        Model identifier for the provider.
        api_key
        API key (if None, reads from ``api_key_env`` environment variable).
        api_key_env
        Environment variable name for the API key.
        egress_level
        Default egress level: ``"schema_only"``, ``"stats_only"`` (default),
        ``"redacted_sample"``, or ``"full_sample"``.
        max_iterations
        Maximum tool iterations per AI call (default 10).
        max_tokens
        Optional token budget limit across all AI calls.
        max_cost_usd
        Optional cost budget limit (USD) across all AI calls.
    session:
        Active Session with dataset and optional split plan attached.
    provider:
        LLM provider identifier (``openai`` or ``mock``).
    model:
        Provider model identifier.
    api_key:
        Optional API key; defaults to the named environment variable.
    api_key_env:
        Environment variable name holding the API key.
    egress_level:
        Default data egress level sent to the provider.
    max_iterations:
        Hard cap on iterative algorithm loops (search, autonomy, clustering refinements, …).
    max_tokens:
        Maximum generation length for provider calls.
    max_cost_usd:
        Soft dollar budget for provider-backed AI autonomy loops.

    Returns
    -------
    Session
    Self for chaining.
    """
    from buildml.ai.planner import BudgetTracker
    from buildml.ai.privacy import EgressConfig, EgressLevel
    from buildml.ai.provider import MockProvider, OpenAIProvider, ProviderConfig
    from buildml.ai.tools import build_default_registry
    from buildml.ai.transcript import TranscriptStore

    level_map = {
        "schema_only": EgressLevel.SCHEMA_ONLY,
        "stats_only": EgressLevel.STATS_ONLY,
        "redacted_sample": EgressLevel.REDACTED_SAMPLE,
        "full_sample": EgressLevel.FULL_SAMPLE,
    }
    egress = level_map.get(egress_level, EgressLevel.STATS_ONLY)
    config = ProviderConfig(
        provider=provider, model=model, api_key=api_key, api_key_env=api_key_env
    )
    if provider == "mock":
        session._ai_provider = MockProvider()
    else:
        session._ai_provider = OpenAIProvider(config)
    session._ai_egress_config = EgressConfig(level=egress)
    session._ai_transcript = TranscriptStore()
    session._ai_registry = build_default_registry()
    session._ai_max_iterations = max_iterations
    session._ai_budget_tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)
    session._record(
        "ai_configure",
        {
            "provider": provider,
            "model": model,
            "egress_level": egress_level,
            "max_iterations": max_iterations,
            "max_tokens": max_tokens,
            "max_cost_usd": max_cost_usd,
        },
    )
    return session


def ai_egress_preview(
    session,
    *,
    level: str | None = None,
    allow_columns: Sequence[str] | None = None,
    deny_columns: Sequence[str] | None = None,
) -> Any:
    """Preview what data will leave the machine before an LLM call.

    Returns an :class:`~buildml.ai.privacy.EgressManifest` showing columns,

    row counts, and estimated tokens that would be sent to the provider.

    Parameters
    ----------
        level
        Override egress level for this preview (``"schema_only"``,
        ``"stats_only"``, ``"redacted_sample"``, ``"full_sample"``).
        allow_columns
        Explicit allowlist of columns to include.
        deny_columns
        Explicit denylist of columns to exclude.
    session:
        Active Session with dataset and optional split plan attached.
    level:
        Optional egress level override for this call.
    allow_columns:
        Explicit allowlist of columns included in egress.
    deny_columns:
        Explicit denylist of columns excluded from egress.

    Returns
    -------
    EgressManifest
    What would leave the machine at this egress level.
    """
    from buildml.ai.privacy import EgressConfig, EgressLevel, build_egress_payload

    base_config = session._ai_egress_config
    if base_config is None:
        base_config = EgressConfig(level=EgressLevel.STATS_ONLY)
    level_map = {
        "schema_only": EgressLevel.SCHEMA_ONLY,
        "stats_only": EgressLevel.STATS_ONLY,
        "redacted_sample": EgressLevel.REDACTED_SAMPLE,
        "full_sample": EgressLevel.FULL_SAMPLE,
    }
    config = EgressConfig(
        level=level_map.get(level, base_config.level) if level else base_config.level,
        allow_columns=tuple(allow_columns) if allow_columns else base_config.allow_columns,
        deny_columns=tuple(deny_columns) if deny_columns else base_config.deny_columns,
    )
    df: pd.DataFrame | None = None
    if session._dataset is not None:
        df = session._dataset.frame
    _, manifest = build_egress_payload(df, config)
    return manifest


def ai_dry_run(session, question: str, *, level: str | None = None) -> dict[str, Any]:
    """Preview the full prompt payload without calling the provider.

    Returns the system prompt, user message, tools, and egress manifest

    that would be sent to the LLM.

    Parameters
    ----------
        question
        The question or goal to preview.
        level
        Override egress level for this preview.
    session:
        Active Session with dataset and optional split plan attached.
    question:
        Natural-language question or goal for the advisor.
    level:
        Optional egress level override for this call.

    Returns
    -------
    dict
    Prompt payload including messages, tools, and egress manifest.
    """
    from buildml.ai.advisor import build_advisor_context, build_state_digest
    from buildml.ai.privacy import EgressConfig, EgressLevel
    from buildml.ai.tools import ToolRegistry

    base_config = session._ai_egress_config
    if base_config is None:
        base_config = EgressConfig(level=EgressLevel.STATS_ONLY)
    level_map = {
        "schema_only": EgressLevel.SCHEMA_ONLY,
        "stats_only": EgressLevel.STATS_ONLY,
        "redacted_sample": EgressLevel.REDACTED_SAMPLE,
        "full_sample": EgressLevel.FULL_SAMPLE,
    }
    config = EgressConfig(
        level=level_map.get(level, base_config.level) if level else base_config.level,
        allow_columns=base_config.allow_columns,
        deny_columns=base_config.deny_columns,
    )
    registry = session._ai_registry or ToolRegistry()
    messages, manifest = build_advisor_context(session, config, question, registry)
    return {
        "messages": [m.to_dict() for m in messages],
        "tools": registry.to_openai_tools(),
        "egress_manifest": manifest.to_dict(),
        "state_digest": build_state_digest(session).to_dict(),
    }


def ai_advisor(session, question: str, *, level: str | None = None, confirm: bool = False) -> Any:
    """Get advisory Q&A guidance about the current workflow (read-only).

    The advisor can describe data, explain operations, and suggest next

    steps, but cannot execute state-changing operations.

    Parameters
    ----------
        question
        The question to ask about the workflow.
        level
        Override egress level for this call.
        confirm
        Required True for FULL_SAMPLE egress (raw data). REDACTED_SAMPLE
        also requires explicit confirmation.
    session:
        Active Session with dataset and optional split plan attached.
    question:
        Natural-language question or goal for the advisor.
    level:
        Optional egress level override for this call.
    confirm:
        Explicit confirmation for sensitive egress or write operations.

    Returns
    -------
    AdvisorResult
    Advisory response with evidence and recommendations.

    Raises
    ------
    ValidationError
    If FULL_SAMPLE or REDACTED_SAMPLE egress is requested without
    confirm=True.
    """
    from buildml.ai.advisor import run_advisor
    from buildml.ai.explain_hooks import advisor_result_summary
    from buildml.ai.privacy import EgressConfig, EgressLevel
    from buildml.ai.tools import ToolRegistry

    if session._ai_provider is None:
        raise ValidationError("No AI provider configured. Call ai_configure() first.")
    base_config = session._ai_egress_config
    if base_config is None:
        base_config = EgressConfig(level=EgressLevel.STATS_ONLY)
    level_map = {
        "schema_only": EgressLevel.SCHEMA_ONLY,
        "stats_only": EgressLevel.STATS_ONLY,
        "redacted_sample": EgressLevel.REDACTED_SAMPLE,
        "full_sample": EgressLevel.FULL_SAMPLE,
    }
    resolved_level = level_map.get(level, base_config.level) if level else base_config.level
    if resolved_level == EgressLevel.FULL_SAMPLE and (not confirm):
        raise ValidationError(
            "FULL_SAMPLE egress sends raw data to the provider and requires explicit confirmation. Pass confirm=True to proceed, or use a safer egress level (stats_only, schema_only, redacted_sample)."
        )
    if resolved_level == EgressLevel.REDACTED_SAMPLE and (not confirm):
        raise ValidationError(
            "REDACTED_SAMPLE egress sends sample rows (with PII masked) to the provider and requires explicit confirmation. Pass confirm=True to proceed, or use stats_only or schema_only."
        )
    config = EgressConfig(
        level=resolved_level,
        allow_columns=base_config.allow_columns,
        deny_columns=base_config.deny_columns,
    )
    registry = session._ai_registry or ToolRegistry()
    use_rag = getattr(session, "_rag_index", None) is not None
    if use_rag:
        from buildml.ai.advisor import run_advisor_with_rag

        result = run_advisor_with_rag(
            session,
            question,
            session._ai_provider,
            egress_config=config,
            registry=registry,
            max_iterations=session._ai_max_iterations,
        )
    else:
        result = run_advisor(
            session,
            question,
            session._ai_provider,
            egress_config=config,
            registry=registry,
            max_iterations=session._ai_max_iterations,
        )
    session._ai_result = result
    session._ai_advisor_result = result
    if session._ai_budget_tracker is not None and result.usage:
        total_tokens = result.usage.get("total_tokens", 0)
        session._ai_budget_tracker.record_usage(total_tokens, operation="ai_advisor")
    if session._ai_transcript is not None:
        from buildml.ai.types import Message

        session._ai_transcript.add_message(Message(role="user", content=question))
        session._ai_transcript.add_message(Message(role="assistant", content=result.answer))
        if result.egress_manifest:
            session._ai_transcript.add_egress_manifest(result.egress_manifest)
    session._record(
        "ai_advisor",
        {"question": question[:100], "egress_level": config.level.value},
        result_summary=advisor_result_summary(result),
    )
    return result


def ai_plan(session, goal: str, *, level: str | None = None, confirm: bool = False) -> Any:
    """Generate a structured workflow plan for a goal (read-only).

    Returns a plan with steps, prerequisites, and expected changes based

    on the current Session state.

    Parameters
    ----------
        goal
        The workflow goal to plan for.
        level
        Override egress level for this call.
        confirm
        Required True for FULL_SAMPLE or REDACTED_SAMPLE egress levels.
    session:
        Active Session with dataset and optional split plan attached.
    goal:
        Workflow goal for planning or autonomous execution.
    level:
        Optional egress level override for this call.
    confirm:
        Explicit confirmation for sensitive egress or write operations.

    Returns
    -------
    PlanResult
    Structured plan with steps, rationale, and limitations.

    Raises
    ------
    ValidationError
    If FULL_SAMPLE or REDACTED_SAMPLE egress is requested without
    confirm=True.
    """
    from buildml.ai.advisor import run_plan
    from buildml.ai.explain_hooks import plan_result_summary
    from buildml.ai.privacy import EgressConfig, EgressLevel

    if session._ai_provider is None:
        raise ValidationError("No AI provider configured. Call ai_configure() first.")
    base_config = session._ai_egress_config
    if base_config is None:
        base_config = EgressConfig(level=EgressLevel.STATS_ONLY)
    level_map = {
        "schema_only": EgressLevel.SCHEMA_ONLY,
        "stats_only": EgressLevel.STATS_ONLY,
        "redacted_sample": EgressLevel.REDACTED_SAMPLE,
        "full_sample": EgressLevel.FULL_SAMPLE,
    }
    resolved_level = level_map.get(level, base_config.level) if level else base_config.level
    if resolved_level == EgressLevel.FULL_SAMPLE and (not confirm):
        raise ValidationError(
            "FULL_SAMPLE egress sends raw data to the provider and requires explicit confirmation. Pass confirm=True to proceed, or use a safer egress level (stats_only, schema_only, redacted_sample)."
        )
    if resolved_level == EgressLevel.REDACTED_SAMPLE and (not confirm):
        raise ValidationError(
            "REDACTED_SAMPLE egress sends sample rows (with PII masked) to the provider and requires explicit confirmation. Pass confirm=True to proceed, or use stats_only or schema_only."
        )
    config = EgressConfig(
        level=resolved_level,
        allow_columns=base_config.allow_columns,
        deny_columns=base_config.deny_columns,
    )
    result = run_plan(session, goal, session._ai_provider, egress_config=config)
    session._ai_result = result
    session._ai_plan_result = result
    if session._ai_budget_tracker is not None and result.usage:
        total_tokens = result.usage.get("total_tokens", 0)
        session._ai_budget_tracker.record_usage(total_tokens, operation="ai_plan")
    if session._ai_transcript is not None:
        from buildml.ai.types import Message

        session._ai_transcript.add_message(Message(role="user", content=f"Plan: {goal}"))
        session._ai_transcript.add_message(Message(role="assistant", content=result.raw_response))
        if result.egress_manifest:
            session._ai_transcript.add_egress_manifest(result.egress_manifest)
    session._record(
        "ai_plan",
        {"goal": goal[:100], "egress_level": config.level.value},
        result_summary=plan_result_summary(result),
    )
    return result


def ai_execute(
    session, tool: str, params: dict[str, Any] | None = None, *, confirm: bool = False
) -> Any:
    """Execute a single tool with propose-confirm-execute flow.

    Proposes the tool execution and requires explicit confirmation for

    write operations. Read-only tools may auto-confirm.

    Parameters
    ----------
        tool
        Name of the tool to execute (must be in the allowed registry).
        params
        Tool arguments as a dictionary.
        confirm
        If True, confirms and executes; otherwise returns a proposal.
    session:
        Active Session with dataset and optional split plan attached.
    tool:
        Tool name from the AI registry to execute.
    params:
        Optional parameters forwarded to the underlying callable.
    confirm:
        Explicit confirmation for sensitive egress or write operations.

    Returns
    -------
    ExecutorProposal or ExecutorResult
    Proposal (if not confirmed) or execution result (if confirmed).
    """
    from buildml.ai.executor import execute_tool, propose_tool_execution
    from buildml.ai.explain_hooks import executor_result_summary
    from buildml.ai.tools import ToolRegistry

    registry = session._ai_registry or ToolRegistry()
    proposal = propose_tool_execution(tool, params or {}, registry)
    if not confirm and proposal.requires_confirmation:
        return proposal
    confirmed = confirm or not proposal.requires_confirmation
    result = execute_tool(session, proposal, confirmed, registry)
    session._ai_result = result
    session._ai_executor_result = result
    if session._ai_transcript is not None:
        session._ai_transcript.add_tool_call(proposal.tool_call, confirmed=result.confirmed)
        if result.executed:
            session._ai_transcript.add_tool_result(proposal.tool_call, result.result_summary)
        if result.error:
            session._ai_transcript.add_error(result.error, proposal.tool_call)
    session._record(
        "ai_execute",
        {
            "tool": tool,
            "params": params,
            "confirmed": result.confirmed,
            "executed": result.executed,
        },
        result_summary=executor_result_summary(result),
    )
    return result


def ai_run_plan(
    session,
    plan: Any | None = None,
    *,
    confirmations: dict[int, bool] | None = None,
    auto_confirm_read_only: bool = True,
    stop_on_error: bool = True,
    stop_on_unconfirmed: bool = True,
    max_steps: int | None = None,
) -> Any:
    """Execute a multi-step plan with confirmation gating.

    Default behavior pauses at the first step requiring confirmation that

    hasn't been confirmed. Read-only steps auto-confirm by default.

    Parameters
    ----------
        plan
        The PlanResult to execute. If None, uses the last ai_plan result.
        confirmations
        Dict mapping step_index -> True/False for confirmation decisions.
        Steps not in the dict use default confirmation behavior.
        auto_confirm_read_only
        If True (default), auto-confirm read-only operations.
        stop_on_error
        If True (default), stop execution on first error.
        stop_on_unconfirmed
        If True (default), stop at steps requiring unconfirmed confirmation.
        max_steps
        Maximum number of steps to execute (None = no limit).
    session:
        Active Session with dataset and optional split plan attached.
    plan:
        Structured plan object from a prior planning call.
    confirmations:
        Step index to confirmation flag for plan execution.
    auto_confirm_read_only:
        When True, auto-approve read-only AI tool calls without interactive confirmation.
    stop_on_error:
        When True, abort a multi-step plan/search at the first error instead of continuing.
    stop_on_unconfirmed:
        When True, stop plan execution when a mutating step lacks confirmation.
    max_steps:
        Hard cap on advisor/planner tool-calling rounds to bound cost and loops.

    Returns
    -------
    PlanExecutionResult
    Combined result of the plan execution with per-step details.

    Raises
    ------
    ValidationError
    If no plan is provided and no prior ai_plan result exists.
    """
    from buildml.ai.planner import run_plan as execute_plan
    from buildml.ai.results import PlanResult
    from buildml.ai.tools import build_default_registry

    if plan is None:
        plan = session._ai_plan_result
    if plan is None:
        plan = session._ai_result
    if not isinstance(plan, PlanResult):
        raise ValidationError(
            "No plan provided and no prior ai_plan result available. Call ai_plan(goal) first or pass a PlanResult."
        )
    registry = session._ai_registry or build_default_registry()
    result = execute_plan(
        session,
        plan,
        registry,
        confirmations=confirmations,
        auto_confirm_read_only=auto_confirm_read_only,
        stop_on_error=stop_on_error,
        stop_on_unconfirmed=stop_on_unconfirmed,
        max_steps=max_steps,
    )
    session._ai_result = result
    session._ai_plan_result = result
    if session._ai_transcript is not None:
        from buildml.ai.types import Message

        session._ai_transcript.add_message(
            Message(
                role="assistant",
                content=f"Executed plan: {result.completed_steps}/{result.total_steps} steps",
            )
        )
    session._record(
        "ai_run_plan",
        {
            "completed_steps": result.completed_steps,
            "total_steps": result.total_steps,
            "stopped_at_step": result.stopped_at_step,
            "stop_reason": result.stop_reason,
            "requires_confirmation_at": result.requires_confirmation_at,
        },
        result_summary={
            "completed": result.completed_steps,
            "total": result.total_steps,
            "all_executed": result.all_executed,
        },
    )
    return result


def ai_run_autonomous(
    session,
    goal: str,
    *,
    plan: Any | None = None,
    confirm_autonomy: bool = False,
    max_steps: int = 8,
    tool_allowlist: Sequence[str] | None = None,
    allow_destructive: bool = False,
    provider_plan: bool = True,
) -> Any:
    """Explicit autonomy mode with hard caps (see :mod:`buildml.ai.autonomous`).

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    goal:
        Workflow goal for planning or autonomous execution.
    plan:
        Structured plan object from a prior planning call.
    confirm_autonomy:
        When True, require an explicit confirmation token before autonomous mutating AI actions.
    max_steps:
        Hard cap on advisor/planner tool-calling rounds to bound cost and loops.
    tool_allowlist:
        Allowlist of AI tool names the planner/advisor may invoke for this call.
    allow_destructive:
        When True, permit destructive Session mutations (drops/overwrites) from AI plan execution.
    provider_plan:
        Provider-side plan/config object used when executing a structured AI plan.

    Returns
    -------
    Any
        Structured domain result recorded on the Session for follow-up evaluate/explain/export steps.
    """
    from buildml.ai.autonomous import AutonomyConfig, run_autonomous

    allowlist = (
        tuple(tool_allowlist)
        if tool_allowlist is not None
        else AutonomyConfig().tool_allowlist
    )
    result = run_autonomous(
        session,
        goal,
        plan=plan,
        config=AutonomyConfig(
            max_steps=max_steps,
            tool_allowlist=allowlist,
            allow_destructive=allow_destructive,
        ),
        registry=session._ai_registry,
        confirm_autonomy=confirm_autonomy,
        provider_plan=provider_plan,
    )
    session._ai_autonomy_result = result
    session._ai_result = result
    if session._ai_budget_tracker is not None and result.usage:
        total_tokens = result.usage.get("total_tokens", 0)
        session._ai_budget_tracker.record_usage(total_tokens, operation="ai_run_autonomous")
    session._record(
        "ai_run_autonomous",
        {
            "goal": goal[:100],
            "confirm_autonomy": confirm_autonomy,
            "max_steps": max_steps,
            "completed_steps": result.completed_steps,
            "stop_reason": result.stop_reason,
        },
        result_summary=result.to_dict(),
    )
    return result


def ai_status(session) -> dict[str, Any]:
    """Get AI operator status including provider, egress, budget, and autonomy.

    Returns factual walkthrough disclosure about the current AI configuration.

    Default path remains propose→confirm→execute; autonomy is opt-in only.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    dict
    Status including provider, egress level, budget, and transcript info.
    """
    from buildml.ai.autonomous import autonomy_status_dict
    from buildml.ai.explain_hooks import ai_status_for_session

    status = ai_status_for_session(session)
    if session._ai_budget_tracker is not None:
        status["budget"] = session._ai_budget_tracker.to_dict()
    status["max_iterations"] = session._ai_max_iterations
    status["registry_tools"] = (
        sorted(t.name for t in session._ai_registry.tools) if session._ai_registry else []
    )
    status["autonomy"] = autonomy_status_dict(getattr(session, "_ai_autonomy_result", None))
    return status


def save_ai_transcript(session, path: str | Path, *, redact: bool = True) -> Path:
    """Save the AI transcript to a JSON file (secrets redacted by default).

    Transcripts record conversation history, tool calls, and egress

    manifests. API keys and raw data are redacted before saving.

    Parameters
    ----------
        path
        Output file path.
        redact
        If True (default), redact potential secrets before saving.
    session:
        Active Session with dataset and optional split plan attached.
    path:
        Filesystem path for load or save.
    redact:
        When True, redact secrets before persisting transcripts.

    Returns
    -------
    Path
    The resolved output path.
    """
    from buildml.ai.transcript import TranscriptStore, save_transcript

    transcript = session._ai_transcript
    if transcript is None:
        transcript = TranscriptStore()
    destination = save_transcript(transcript, path, redact=redact)
    session._record("save_ai_transcript", {"path": str(destination), "redact": redact})
    return destination


def load_ai_transcript(session, path: str | Path) -> Session:
    """Load an AI transcript for resume or audit.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
        path
        Input file path.
    session:
        Active Session with dataset and optional split plan attached.
    path:
        Filesystem path for load or save.

    Returns
    -------
    Session
    Self for chaining.
    """
    from buildml.ai.transcript import load_transcript

    session._ai_transcript = load_transcript(path)
    session._record("load_ai_transcript", {"path": str(path)})
    return session
