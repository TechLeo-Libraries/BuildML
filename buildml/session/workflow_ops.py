"""Dry-run, history summary, walkthrough, and explain orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

if TYPE_CHECKING:
    from buildml.session.session import Session

from buildml.session._imports import (
    DryRunReport,
    HistorySummary,
    LearningBrief,
    WorkflowStep,
    WorkflowWalkthroughReport,
    academy_learn,
    build_history_summary,
    build_walkthrough,
    explain_session,
    resolve_workflow,
    run_dry_run,
)


def dry_run(
    session,
    operation: str | Sequence[str] | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> DryRunReport:
    """See what an operation would do, without doing it.

    Some steps are expensive and some are hard to undo. A dry run checks
    whether an operation could run right now, what it would need, and what
    it would change: and then changes nothing. No fitting, no
    transforming, no history entry.

    It is the natural companion to :meth:`workflow`: that tells you which
    steps are available, this tells you what a particular one would
    actually do here.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    operation:
        One operation name, several names to preview as a sequence, or
        ``None`` for an overview of what is currently available and what is
        blocked, with the reason for each block.
    parameters:
        The arguments you intend to pass, so the preview reflects your
        specific call rather than the defaults. Applies to a
        single-operation preview.

    Returns
    -------
    ~buildml.session.audit.DryRunReport
        What each previewed operation requires, whether those requirements
        are met, what it would change, and any warnings. Also stored on
        :attr:`last_dry_run`.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A named operation is not one BuildML knows.

    Notes
    -----
    Dry-run does not fit, transform, or append history. Availability means
    API prerequisites pass, not that the operation is appropriate.

    That distinction matters. A dry run confirms that :meth:`split` *can*
    run; it cannot tell you that :meth:`group_split` is the one your data
    requires. Statistical judgement is still yours.

    Examples
    --------
    >>> session.dry_run("scale", parameters={"method": "minmax"})  # doctest: +SKIP
    >>> session.dry_run()  # doctest: +SKIP

    See Also
    --------
    Session.workflow : Availability across every operation.
    Session.explain : What an operation means, rather than whether it runs.
    """
    report = run_dry_run(session, operation, parameters=parameters)
    session._last_dry_run = report
    return report


def summarize_history(session) -> HistorySummary:
    """Condense what this session did, and flag what looks risky.

    The raw :attr:`history` is complete but long. This summarises it :
    which operations ran, in what order, which choices were explicit and
    which were defaults: and adds a list of unresolved risks worth a
    second look.

    The risk list is the reason to call it. Preprocessing that ran before
    the split, an evaluation on test taken more than once, a model fitted
    without a stratified split on imbalanced data: these are easy to do and
    easy to forget, and each quietly changes what your numbers mean.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.

    Returns
    -------
    ~buildml.session.audit.HistorySummary
        The condensed record with its risk list. Also stored on
        :attr:`last_history_summary`.

    Notes
    -----
    Read-only. Does not append history. Risks are heuristic review cues,
    not proof of leakage or invalid results.

    Treat a flagged risk as a question rather than a verdict. Some are
    deliberate: you may have every reason to preprocess before splitting
    on a dataset you are only exploring. The point is that the decision
    should be one you made rather than one that happened.

    Examples
    --------
    >>> summary = session.summarize_history()  # doctest: +SKIP
    >>> summary.risks  # doctest: +SKIP
    ['Session-global scale ran before cv_score; fold estimates may be optimistic.']

    See Also
    --------
    Session.walkthrough : The narrative version, exportable to HTML.
    Session.history : The raw records.
    """
    summary = build_history_summary(session)
    session._last_history_summary = summary
    return summary


def workflow(session) -> tuple[WorkflowStep, ...]:
    """List every operation, with what it needs and whether it can run now.

    A session exposes several hundred methods, and which of them make sense
    depends entirely on where you are: you cannot fit before splitting, or
    evaluate before fitting. This resolves the whole surface against the
    session's current state and reports the status of each step.

    It answers "what can I do next?" without reading the documentation
    first, which is also why it backs the AI tooling: an agent needs the
    same answer, in the same machine-readable form.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.

    Returns
    -------
    tuple of ~buildml.explain.schemas.WorkflowStep
        One entry per public operation, with its identifier, what it
        requires, whether those requirements are currently met, and whether
        it has already run.

    Examples
    --------
    >>> ready = [s for s in session.workflow() if s.available]  # doctest: +SKIP

    See Also
    --------
    Session.explain : What one specific operation will do.
    Session.walkthrough : A narrative of what has already happened.
    Session.dry_run : Preview a step's effect without running it.
    """
    return resolve_workflow(session)


def walkthrough(
    session,
    *,
    export_html: str | Path | None = None,
    capability_probe: str = "lazy",
) -> WorkflowWalkthroughReport:
    """Narrate everything this session did, and why.

    Turns the operation history into a readable account: which steps ran,
    what they were given, which choices were yours and which were BuildML's
    defaults, and what each one changed. It is the report you produce when
    someone asks how a number was arrived at: a colleague reviewing the
    work, an auditor, or yourself in three months.

    Because it is generated from the recorded history rather than written
    by hand, it cannot drift away from what actually happened.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    export_html:
        Path to write a self-contained HTML version to. ``None`` returns
        the report without writing anything.
    capability_probe:
        ``lazy`` (default) probes capability matrices only for domains that
        already have Session artifacts. ``eager`` probes every domain
        (cached process-wide). ``skip`` never loads industry stacks.

    Returns
    -------
    ~buildml.session.walkthrough.WorkflowWalkthroughReport
        The narrated report: the ordered steps, the reasoning behind each,
        and any warnings raised along the way. Also stored on
        :attr:`last_walkthrough`.

    Examples
    --------
    >>> report = session.walkthrough(export_html="reports/run.html")  # doctest: +SKIP

    See Also
    --------
    Session.summarize_history : A shorter, structured summary.
    Session.model_card : The equivalent artefact for a saved pipeline.
    Session.history : The raw records underneath.
    """
    report = build_walkthrough(session, capability_probe=capability_probe)
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
    """Ask what an operation does, in plain language, at any point.

    BuildML's explanations are part of the library rather than a separate
    manual, so you can ask from inside your code. Name an operation and you
    get an account of what it does, what it needs, what it will change, and
    the traps worth knowing about: written for someone meeting the concept
    for the first time.

    The ``moment`` argument changes the tense and therefore the usefulness.
    Before running a step, you get what it is about to do and what to watch
    for. After running it, you get what it actually did to *this* session,
    with the real numbers.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    operation:
        The operation to explain, named as the method is
        (``'split'``, ``'encode'``, ``'cv_score'``). ``None`` returns the
        whole workflow view, the same as :meth:`workflow`.
    moment:
        ``'before'`` for what the step will do and what it requires;
        ``'after'`` for what it did here, grounded in this session's state.
    level:
        How much depth to render: ``'beginner'`` (the default) leads with a
        plain-language primer, an analogy, the steps in order, and a
        glossary of the terms it uses; ``'intermediate'`` trims the
        introductory material; ``'advanced'`` assumes the vocabulary and
        keeps the full risk and assumption lists.

    Returns
    -------
    object
        An explanation record for the named operation, or the full workflow
        tuple when ``operation`` is ``None``. Operation explanations carry a
        ``beginner`` primer alongside the expert sections.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The named operation is not one BuildML knows.
    ValueError
        ``level`` is not one of the three reading levels.

    Notes
    -----
    The conceptual material lives in :mod:`buildml.explain`, which is also
    where the guides and the AI tooling read from: so what you are told
    here is the same thing every other surface is told. The level changes
    how much is shown, never what is true.

    Examples
    --------
    >>> session.explain("group_split")  # doctest: +SKIP
    >>> session.explain("split").beginner.analogy  # doctest: +SKIP
    >>> session.explain("encode", moment="after", level="advanced")  # doctest: +SKIP

    See Also
    --------
    Session.learn : Teach a concept, operation, or term from first principles.
    Session.workflow : Every operation and its current availability.
    Session.walkthrough : What this session has already done.
    Session.dry_run : Preview an operation's effect on real data.
    """
    return explain_session(session, operation, moment=moment, level=level)


def learn(session, topic: str | None = None, *, level: str = "beginner") -> LearningBrief:
    """Teach a concept, an operation, or a term: and say what to read first.

    :meth:`explain` answers "what will this call do here, now?".
    :meth:`learn` answers the prior question: "what is this, and what do I
    need to understand before it makes sense?". You can name either side of
    the vocabulary: the operation (``'split'``), the concept behind it
    (``'data-splitting'``), or the word you tripped over (``'leakage'``) :
    and BuildML works out which you meant.

    Called with no topic it returns the foundation concepts, which is the
    sensible place to start if you are new to this.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    topic:
        A concept key, an operation name, or a glossary term. ``None``
        returns the foundation reading list.
    level:
        ``'beginner'`` (the default), ``'intermediate'``, or ``'advanced'``.

    Returns
    -------
    ~buildml.explain.academy.LearningBrief
        The material for the topic, plus ``read_first`` and ``read_next``
        concept notes giving a reading order rather than an index.

    Raises
    ------
    KeyError
        No concept, operation, or term matches; close matches are suggested
        in the message.
    ValueError
        ``level`` is not one of the three reading levels.

    Examples
    --------
    >>> session.learn()                        # doctest: +SKIP
    >>> session.learn("leakage-boundary")      # doctest: +SKIP
    >>> session.learn("fit", level="advanced") # doctest: +SKIP

    See Also
    --------
    Session.explain : What an operation does at this point in this session.
    Session.workflow : Which operations can run right now.
    """
    return academy_learn(topic, level=level)
