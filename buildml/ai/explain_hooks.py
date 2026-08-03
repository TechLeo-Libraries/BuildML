"""Condense AI operator results for history and teaching surfaces.

Session history, walkthroughs, and the Teaching Studio need short, JSON-safe
summaries — not whole result objects. An advisory answer can run to paragraphs
and a plan to a dozen steps; neither belongs verbatim in a history entry.

The summarisers here take counts and previews instead: how much evidence was
cited, how many steps a plan had, whether a tool actually executed. Enough to
see the shape of what happened, small enough to store on every operation.

:func:`ai_status` reports the state of the AI domain for walkthroughs, and is
deliberately conservative in what it claims — that a provider is configured, not
that advice is reliable; that confirmation is required, not that the system is
safe.

Notes
-----
**Every function here accepts ``None`` and objects that are not results.** They
run against whatever a Session happens to hold, and a teaching surface should
degrade to an empty summary rather than fail.

See Also
--------
buildml.ai.results : The full result objects.
"""

from __future__ import annotations

from typing import Any


def advisor_result_summary(result: Any) -> dict[str, Any]:
    """Summarise an advisory answer for a history entry.

    Truncates the question and answer to previews and reduces the rest to
    counts, keeping the entry small enough to store on every operation.

    Parameters
    ----------
    result:
        An :class:`~buildml.ai.advisor.AdvisorResult`, a mapping, or ``None``.

    Returns
    -------
    dict
        Question preview, answer preview, evidence and recommendation counts,
        and the egress level. Empty when there is nothing to summarise.

    Notes
    -----
    **The egress level is kept while the manifest is dropped**, because the
    level is what a reader scanning history wants to see. The full manifest
    lives in the transcript.

    See Also
    --------
    buildml.ai.advisor.AdvisorResult : The full object.
    """
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = dict(result) if isinstance(result, dict) else {}
    return {
        "question": payload.get("question", "")[:100],
        "answer_preview": (payload.get("answer") or "")[:200],
        "evidence_count": len(payload.get("evidence") or []),
        "recommendations_count": len(payload.get("recommendations") or []),
        "egress_level": (
            payload.get("egress_manifest", {}).get("level")
            if payload.get("egress_manifest")
            else None
        ),
    }


def executor_result_summary(result: Any) -> dict[str, Any]:
    """Summarise a tool execution for a history entry.

    Records what was attempted and how it ended, without the returned object —
    which may be a fitted model or a frame and has no place in history.

    Parameters
    ----------
    result:
        An :class:`~buildml.ai.executor.ExecutorResult`, a mapping, or ``None``.

    Returns
    -------
    dict
        Tool name, the confirmation and execution flags, any error, and how
        many state changes were recorded. Empty when there is nothing to
        summarise.

    Notes
    -----
    **``executed`` is the field that matters when scanning history.** A run of
    entries with ``confirmed`` true and ``executed`` false is a sequence of
    approved operations that all failed.

    See Also
    --------
    buildml.ai.executor.ExecutorResult : The full object.
    """
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = dict(result) if isinstance(result, dict) else {}
    return {
        "tool_name": (payload.get("tool_call") or {}).get("tool_name"),
        "confirmed": payload.get("confirmed"),
        "executed": payload.get("executed"),
        "error": payload.get("error"),
        "state_changes_count": len(payload.get("state_changes") or []),
    }


def plan_result_summary(result: Any) -> dict[str, Any]:
    """Summarise a generated plan for a history entry.

    Keeps the goal and the first few operation names — enough to recognise the
    plan later — and reduces the reasoning to counts.

    Parameters
    ----------
    result:
        A :class:`~buildml.ai.results.PlanResult`, a mapping, or ``None``.

    Returns
    -------
    dict
        Goal preview, step count, the first five operations, and the numbers of
        assumptions and limitations. Empty when there is nothing to summarise.

    Notes
    -----
    **Only the first five operations are listed.** A longer plan is truncated
    here; ``step_count`` still reports the true total.

    **The rationales are dropped entirely.** They are the most valuable part of
    a plan and the least suited to a history entry. Read the
    :class:`~buildml.ai.results.PlanResult` for those.

    See Also
    --------
    buildml.ai.results.PlanResult : The full object.
    """
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = dict(result) if isinstance(result, dict) else {}
    steps = payload.get("steps") or []
    return {
        "goal": payload.get("goal", "")[:100],
        "step_count": len(steps),
        "operations": [s.get("operation") for s in steps[:5]],
        "assumptions_count": len(payload.get("assumptions") or []),
        "limitations_count": len(payload.get("limitations") or []),
    }


def ai_status(
    *,
    provider_configured: bool = False,
    provider_type: str | None = None,
    egress_level: str | None = None,
    transcript_entries: int = 0,
    last_advisor_result: Any | None = None,
    last_executor_result: Any | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report the state of the AI domain, claiming nothing it should not.

    Builds the walkthrough view: whether a provider is configured, what egress
    level applies, how much transcript exists, and what the last advisory and
    execution did. The disclosures are written to be accurate rather than
    reassuring — that confirmation is required, and that the advice needs
    verifying.

    Parameters
    ----------
    provider_configured:
        Whether a provider is attached.
    provider_type:
        Its class name.
    egress_level:
        The configured level, as a string.
    transcript_entries:
        How many events have been recorded.
    last_advisor_result:
        The most recent advisory answer, summarised into the output.
    last_executor_result:
        The most recent execution, summarised into the output.
    history:
        Session history, scanned for operations beginning ``ai_``.

    Returns
    -------
    dict
        Enabled and present flags, the disclosures, provider and egress
        details, transcript size, and the two summaries.

    Notes
    -----
    **The disclosures state limits, not capabilities.** No claim of autonomy,
    no implication that keys are persisted, and no suggestion that an available
    catalog entry means an operation is production-ready.

    **``present`` covers past use as well as current configuration**, so a
    Session that used AI earlier still surfaces its history after the provider
    is detached.

    See Also
    --------
    ai_status_for_session : This, read from a Session.
    """
    records = list(history or [])
    saw_ai = any(
        str(r.get("operation_id") or r.get("action")).startswith("ai_")
        for r in records
    )

    disclosures = []

    if not provider_configured:
        disclosures.append(
            "No AI provider configured. Call ai_configure() with API key from "
            "environment variable before using AI methods."
        )
    else:
        disclosures.append(f"Provider type: {provider_type or 'unknown'}.")
        disclosures.append(
            "API keys are never persisted in transcripts, checkpoints, or bundles."
        )

    if egress_level:
        disclosures.append(f"Default egress level: {egress_level}.")
        if egress_level == "stats_only":
            disclosures.append(
                "STATS_ONLY egress sends aggregates and schema, not raw row values."
            )

    disclosures.append(
        "AI operator uses propose-confirm-execute flow; write operations "
        "require explicit confirmation."
    )

    if saw_ai:
        ai_ops = [
            r.get("operation_id") or r.get("action")
            for r in records
            if str(r.get("operation_id") or r.get("action")).startswith("ai_")
        ]
        disclosures.append(f"AI operations in history: {ai_ops[-5:]}")

    disclosures.append(
        "AI advice is not infallible; verify recommendations before production use."
    )

    return {
        "enabled": provider_configured,
        "present": saw_ai or provider_configured,
        "disclosures": disclosures,
        "provider": {
            "configured": provider_configured,
            "type": provider_type,
        },
        "egress": {
            "default_level": egress_level,
        },
        "transcript": {
            "entry_count": transcript_entries,
        },
        "last_advisor": advisor_result_summary(last_advisor_result),
        "last_executor": executor_result_summary(last_executor_result),
    }


def ai_status_for_session(session: Any) -> dict[str, Any]:
    """Report AI domain status by reading it off a Session.

    Pulls the provider, egress configuration, transcript, last results, and
    history, then hands them to :func:`ai_status`. The convenience form, used
    by walkthroughs that hold a Session and nothing else.

    Parameters
    ----------
    session:
        The Session to inspect.

    Returns
    -------
    dict
        The status payload from :func:`ai_status`.

    Notes
    -----
    Every attribute is read defensively, so a Session that never touched the AI
    domain reports a coherent "not configured" status rather than failing.

    See Also
    --------
    ai_status : The underlying builder.
    """
    return ai_status(
        provider_configured=bool(getattr(session, "_ai_provider", None)),
        provider_type=getattr(
            getattr(session, "_ai_provider", None), "__class__", type(None)
        ).__name__
        if getattr(session, "_ai_provider", None)
        else None,
        egress_level=getattr(
            getattr(session, "_ai_egress_config", None), "level", None
        ),
        transcript_entries=len(
            getattr(getattr(session, "ai_transcript", None), "entries", [])
        ),
        last_advisor_result=getattr(session, "_ai_advisor_result", None),
        last_executor_result=getattr(session, "_ai_executor_result", None),
        history=list(getattr(session, "history", []) or []),
    )
