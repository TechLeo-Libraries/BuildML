"""Session mixin: workflow domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import workflow_ops
from buildml.session.mixins._shared import *  # noqa: F403


class WorkflowSessionMixin:
    """Public Session methods for the workflow domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _history: Any
        _last_dry_run: Any
        _last_history_summary: Any
        _last_walkthrough: Any

    @property
    def history(self) -> list[dict[str, Any]]:
        """Every operation this session has performed, in order.

        Session-held result for ``history``.
        """
        return cast("list[dict[str, Any]]", list(self._history))

    def dry_run(
        self,
        operation: str | Sequence[str] | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> DryRunReport:
        """See what an operation would do, without doing it.

        Session facade over :func:`buildml.session.workflow_ops.dry_run`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.session.audit.DryRunReport
            What each previewed operation requires, whether those requirements

        See Also
        --------
        :func:`buildml.session.workflow_ops.dry_run`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("DryRunReport", workflow_ops.dry_run(self, operation=operation, parameters=parameters))

    @property
    def last_dry_run(self) -> DryRunReport | None:
        """The most recent :meth:`dry_run` report.

        Kept so a preview can be re-read after the fact. ``None`` until
        :meth:`dry_run` runs.
        """
        return cast("DryRunReport | None", self._last_dry_run)

    def summarize_history(self) -> HistorySummary:
        """Condense what this session did, and flag what looks risky.

        Session facade over :func:`buildml.session.workflow_ops.summarize_history`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.session.audit.HistorySummary
            The condensed record with its risk list. Also stored on

        See Also
        --------
        :func:`buildml.session.workflow_ops.summarize_history`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("HistorySummary", workflow_ops.summarize_history(self))

    @property
    def last_history_summary(self) -> HistorySummary | None:
        """The most recent :meth:`summarize_history` result.

        Kept so the summary and its risk list can be re-read without
        recomputing. ``None`` until :meth:`summarize_history` runs.
        """
        return cast("HistorySummary | None", self._last_history_summary)

    def workflow(self) -> tuple[WorkflowStep, ...]:
        """List every operation, with what it needs and whether it can run now.

        Session facade over :func:`buildml.session.workflow_ops.workflow`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        tuple of ~buildml.explain.schemas.WorkflowStep
            One entry per public operation, with its identifier, what it

        See Also
        --------
        :func:`buildml.session.workflow_ops.workflow`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("tuple[WorkflowStep, ...]", workflow_ops.workflow(self))

    def walkthrough(
        self,
        *,
        export_html: str | Path | None = None,
        capability_probe: str = "lazy",
    ) -> WorkflowWalkthroughReport:
        """Narrate everything this session did, and why.

        Session facade over :func:`buildml.session.workflow_ops.walkthrough`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.session.walkthrough.WorkflowWalkthroughReport
            The narrated report: the ordered steps, the reasoning behind each,

        See Also
        --------
        :func:`buildml.session.workflow_ops.walkthrough`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast(
            "WorkflowWalkthroughReport",
            workflow_ops.walkthrough(
                self,
                export_html=export_html,
                capability_probe=capability_probe,
            ),
        )

    @property
    def last_walkthrough(self) -> WorkflowWalkthroughReport | None:
        """The most recently generated walkthrough report.

        Set by :meth:`walkthrough`. Kept on the session so a report built
        earlier can be re-read or re-exported without regenerating it.

        ``None`` until :meth:`walkthrough` runs.
        """
        return cast("WorkflowWalkthroughReport | None", self._last_walkthrough)

    def explain(
        self,
        operation: str | None = None,
        *,
        moment: Literal["before", "after"] = "before",
        level: str = "beginner",
    ) -> Any:
        """Ask what an operation does, in plain language, at any point.

        Session facade over :func:`buildml.session.workflow_ops.explain`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        object
            An explanation record for the named operation, or the full workflow

        See Also
        --------
        :func:`buildml.session.workflow_ops.explain`
            Canonical documentation for parameters, raises, and examples.
        """
        return workflow_ops.explain(self, operation=operation, moment=moment, level=level)

    def learn(self, topic: str | None = None, *, level: str = "beginner") -> Any:
        """Teach a concept, an operation, or a term: and say what to read first.

        Session facade over :func:`buildml.session.workflow_ops.learn`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.explain.academy.LearningBrief
            The material for the topic, plus ``read_first`` and ``read_next``

        See Also
        --------
        :func:`buildml.session.workflow_ops.learn`
            Canonical documentation for parameters, raises, and examples.
        """
        return workflow_ops.learn(self, topic=topic, level=level)
