"""History / catalog / walkthrough helpers for AI operator operations."""

from __future__ import annotations

from typing import Any


def advisor_result_summary(result: Any) -> dict[str, Any]:
    """Compact result_summary for ai_advisor history."""
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
    """Compact result_summary for ai_execute history."""
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
    """Compact result_summary for ai_plan history."""
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
    """Factual walkthrough disclosure for AI operator status.

    Does not claim autonomous capability, does not imply keys are persisted,
    and does not treat catalog availability as production safety.
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
    """Build walkthrough ai_status from a Session."""
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
