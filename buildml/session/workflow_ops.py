"""Dry-run, history summary, walkthrough, and explain orchestration."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


def dry_run(
    session,
    operation: str | Sequence[str] | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> DryRunReport:
    """Preview intended operations without mutating Session state.

    Parameters
    ----------
    operation:
        One operation name, a sequence of names, or ``None`` for a focused
        default preview of available/blocked next steps.
    parameters:
        Optional parameters attached to a single-operation preview.

    Notes
    -----
    Dry-run does not fit, transform, or append history. Availability means
    API prerequisites pass, not that the operation is appropriate.
    """
    report = run_dry_run(session, operation, parameters=parameters)
    session._last_dry_run = report
    return report


def summarize_history(session) -> HistorySummary:
    """Summarize operation history and list unresolved risks.

    Notes
    -----
    Read-only. Does not append history. Risks are heuristic review cues,
    not proof of leakage or invalid results.
    """
    summary = build_history_summary(session)
    session._last_history_summary = summary
    return summary


def workflow(session) -> tuple[WorkflowStep, ...]:
    """Resolve every public operation against current workflow state."""
    return resolve_workflow(session)


def walkthrough(session, *, export_html: str | Path | None = None) -> WorkflowWalkthroughReport:
    """Build a workflow walkthrough from resolver state and history."""
    report = build_walkthrough(session)
    if export_html is not None:
        report.export_html(export_html)
    session._last_walkthrough = report
    return report


def explain(
    session,
    operation: str | None = None,
    *,
    moment: Literal['before', 'after'] = "before",
    level: str = "beginner",
) -> Any:
    """Explain an operation before/after execution, or return the workflow."""
    return explain_session(session, operation, moment=moment, level=level)


def learn(session, topic: str | None = None, *, level: str = "beginner") -> LearningBrief:
    """Return teaching material for a concept, an operation, or a term.

    Read-only and session-independent: the material comes from the catalog and
    concept notes, so the answer does not depend on how far along the workflow
    is. The session argument keeps the call available where every other
    operation lives.
    """
    return academy_learn(topic, level=level)
