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

    Delegates to :func:`buildml.session.audit.run_dry_run` and stores the
    report on Session for later inspection. Use before expensive fits to
    confirm prerequisites.

    Parameters
    ----------
    session:
        Active Session whose workflow state drives availability checks.
    operation:
        One operation name, a sequence of names, or ``None`` for a focused
        default preview of available/blocked next steps.
    parameters:
        Optional parameters attached to a single-operation preview.

    Returns
    -------
    DryRunReport
        Availability, blockers, and disclosure messages for each operation.

    Notes
    -----
    Dry-run does not fit, transform, or append history. Availability means
    API prerequisites pass, not that the operation is appropriate.
    """
    report = run_dry_run(session, operation, parameters=parameters)
    session._last_dry_run = report
    return report


def summarize_history(session) -> HistorySummary:
    """Summarize operation history and list unresolved workflow risks.

    Delegates to :func:`buildml.session.audit.build_history_summary` and
    stores the summary on Session. Read-only — does not append history.

    Parameters
    ----------
    session:
        Active Session whose ``_history`` log is summarized.

    Returns
    -------
    HistorySummary
        Operation counts, last results, and heuristic risk flags.

    Notes
    -----
    Read-only. Does not append history. Risks are heuristic review cues,
    not proof of leakage or invalid results.
    """
    summary = build_history_summary(session)
    session._last_history_summary = summary
    return summary


def workflow(session) -> tuple[WorkflowStep, ...]:
    """Resolve every public operation against current workflow state.

    Delegates to :func:`buildml.session.walkthrough.resolve_workflow` and
    returns the ordered list of workflow steps with availability metadata.

    Parameters
    ----------
    session:
        Active Session whose dataset, split, and history drive resolution.

    Returns
    -------
    tuple of WorkflowStep
        Ordered workflow steps with status, prerequisites, and disclosures.
    """
    return resolve_workflow(session)


def walkthrough(session, *, export_html: str | Path | None = None) -> WorkflowWalkthroughReport:
    """Build a workflow walkthrough from resolver state and history.

    Delegates to :func:`buildml.session.walkthrough.build_walkthrough`,
    optionally exports HTML, and stores the report on Session.

    Parameters
    ----------
    session:
        Active Session whose workflow state and history populate the report.
    export_html:
        Optional path to write a self-contained HTML walkthrough artifact.

    Returns
    -------
    WorkflowWalkthroughReport
        Narrative walkthrough with recommended next steps and disclosures.
    """
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
    """Explain an operation or return the full workflow teaching surface.

    Delegates to :func:`buildml.session.walkthrough.explain_session`.
    When ``operation`` is ``None``, returns the workflow resolver output.

    Parameters
    ----------
    session:
        Active Session whose state contextualizes before/after explanations.
    operation:
        Operation name to explain, or ``None`` for the full workflow.
    moment:
        Explain ``before`` calling the operation or ``after`` it ran.
    level:
        Teaching depth (``beginner``, ``intermediate``, etc.).

    Returns
    -------
    ExplainResult or workflow
        Operation explanation or workflow resolver output when ``operation`` is
        ``None``.
    """
    return explain_session(session, operation, moment=moment, level=level)


def learn(session, topic: str | None = None, *, level: str = "beginner") -> LearningBrief:
    """Return teaching material for a concept, operation, or glossary term.

    Delegates to :func:`buildml.session.walkthrough.academy_learn`. The
    material comes from the concept catalog and does not depend on Session
    progress; the session argument keeps the call on the Session surface.

    Parameters
    ----------
    session:
        Session instance (unused; kept for API consistency with other ops).
    topic:
        Concept, operation, or term to look up; ``None`` returns the index.
    level:
        Teaching depth (``beginner``, ``intermediate``, etc.).

    Returns
    -------
    LearningBrief
        Structured teaching content with links to related operations.
    """
    return academy_learn(topic, level=level)
