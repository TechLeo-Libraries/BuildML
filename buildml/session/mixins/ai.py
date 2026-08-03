"""Session mixin: ai domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import ai_ops
from buildml.session.mixins._shared import *  # noqa: F403


class AiSessionMixin:
    """Public Session methods for the ai domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _ai_result: Any
        _ai_transcript: Any

    def ai_configure(
        self,
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

        Session facade over :func:`buildml.session.ai_ops.ai_configure`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            Self for chaining.

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_configure`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", ai_ops.ai_configure(
            self,
            provider=provider,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            egress_level=egress_level,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
        ))

    def ai_egress_preview(
        self,
        *,
        level: str | None = None,
        allow_columns: Sequence[str] | None = None,
        deny_columns: Sequence[str] | None = None,
    ) -> EgressManifest:
        """Preview what data will leave the machine before an LLM call.

        Session facade over :func:`buildml.session.ai_ops.ai_egress_preview`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        EgressManifest
            What would leave the machine at this egress level.

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_egress_preview`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("EgressManifest", ai_ops.ai_egress_preview(
            self, level=level, allow_columns=allow_columns, deny_columns=deny_columns
        ))

    def ai_dry_run(
        self,
        question: str,
        *,
        level: str | None = None,
    ) -> dict[str, Any]:
        """Preview the full prompt payload without calling the provider.

        Session facade over :func:`buildml.session.ai_ops.ai_dry_run`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Prompt payload including messages, tools, and egress manifest.

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_dry_run`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", ai_ops.ai_dry_run(self, question=question, level=level))

    def ai_advisor(
        self,
        question: str,
        *,
        level: str | None = None,
        confirm: bool = False,
    ) -> AdvisorResult:
        """Get advisory Q&A guidance about the current workflow (read-only).

        Session facade over :func:`buildml.session.ai_ops.ai_advisor`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        AdvisorResult
            Advisory response with evidence and recommendations.

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_advisor`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("AdvisorResult", ai_ops.ai_advisor(self, question=question, level=level, confirm=confirm))

    def ai_plan(
        self,
        goal: str,
        *,
        level: str | None = None,
        confirm: bool = False,
    ) -> PlanResult:
        """Generate a structured workflow plan for a goal (read-only).

        Session facade over :func:`buildml.session.ai_ops.ai_plan`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        PlanResult
            Structured plan with steps, rationale, and limitations.

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_plan`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("PlanResult", ai_ops.ai_plan(self, goal=goal, level=level, confirm=confirm))

    def ai_execute(
        self,
        tool: str,
        params: dict[str, Any] | None = None,
        *,
        confirm: bool = False,
    ) -> ExecutorProposal | ExecutorResult:
        """Execute a single tool with propose-confirm-execute flow.

        Session facade over :func:`buildml.session.ai_ops.ai_execute`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ExecutorProposal or ExecutorResult
            Proposal (if not confirmed) or execution result (if confirmed).

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_execute`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ExecutorProposal | ExecutorResult", ai_ops.ai_execute(self, tool=tool, params=params, confirm=confirm))

    def ai_run_plan(
        self,
        plan: Any | None = None,
        *,
        confirmations: dict[int, bool] | None = None,
        auto_confirm_read_only: bool = True,
        stop_on_error: bool = True,
        stop_on_unconfirmed: bool = True,
        max_steps: int | None = None,
    ) -> PlanExecutionResult:
        """Execute a multi-step plan with confirmation gating.

        Session facade over :func:`buildml.session.ai_ops.ai_run_plan`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        PlanExecutionResult
            Combined result of the plan execution with per-step details.

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_run_plan`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("PlanExecutionResult", ai_ops.ai_run_plan(
            self,
            plan=plan,
            confirmations=confirmations,
            auto_confirm_read_only=auto_confirm_read_only,
            stop_on_error=stop_on_error,
            stop_on_unconfirmed=stop_on_unconfirmed,
            max_steps=max_steps,
        ))

    def ai_run_autonomous(
        self,
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

        Session facade over :func:`buildml.session.ai_ops.ai_run_autonomous`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Any
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_run_autonomous`
            Canonical documentation for parameters, raises, and examples.
        """
        return ai_ops.ai_run_autonomous(
            self,
            goal,
            plan=plan,
            confirm_autonomy=confirm_autonomy,
            max_steps=max_steps,
            tool_allowlist=tool_allowlist,
            allow_destructive=allow_destructive,
            provider_plan=provider_plan,
        )

    def ai_status(self) -> dict[str, Any]:
        """Get AI operator status including provider, egress, budget, and autonomy.

        Session facade over :func:`buildml.session.ai_ops.ai_status`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        dict
            Status including provider, egress level, budget, and transcript info.

        See Also
        --------
        :func:`buildml.session.ai_ops.ai_status`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", ai_ops.ai_status(self))

    def save_ai_transcript(self, path: str | Path, *, redact: bool = True) -> Path:
        """Save the AI transcript to a JSON file (secrets redacted by default).

        Session facade over :func:`buildml.session.ai_ops.save_ai_transcript`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Path

        See Also
        --------
        :func:`buildml.session.ai_ops.save_ai_transcript`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", ai_ops.save_ai_transcript(self, path=path, redact=redact))

    def load_ai_transcript(self, path: str | Path) -> Session:
        """Load an AI transcript for resume or audit.

        Session facade over :func:`buildml.session.ai_ops.load_ai_transcript`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session

        See Also
        --------
        :func:`buildml.session.ai_ops.load_ai_transcript`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", ai_ops.load_ai_transcript(self, path=path))

    @property
    def ai_result(
        self,
    ) -> AdvisorResult | PlanResult | ExecutorResult | PlanExecutionResult | None:
        """Return the most recent AI advisor, plan, execute, or run-plan result.

        Session-held result for ``ai_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("AdvisorResult | PlanResult | ExecutorResult | PlanExecutionResult | None", self._ai_result)

    @property
    def ai_transcript(self) -> TranscriptStore | None:
        """Return the active AI transcript store for this Session.

        Created by :meth:`ai_configure` and populated by AI calls; reload with
        :meth:`load_ai_transcript`.

        Returns
        -------
        TranscriptStore or None
            ``None`` until :meth:`ai_configure` or :meth:`load_ai_transcript` has run."""
        return cast("TranscriptStore | None", self._ai_transcript)
