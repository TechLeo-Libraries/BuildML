"""LLM operator domain for BuildML.

This package provides AI-assisted workflow guidance via typed tool registry,
privacy-aware egress, and propose-confirm-execute patterns.

Public exports are lazy-loaded to keep ``import buildml`` lightweight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from buildml.ai.advisor import AdvisorResult
    from buildml.ai.executor import ExecutorProposal, ExecutorResult
    from buildml.ai.planner import (
        BudgetExceeded,
        BudgetTracker,
        PlanExecutionResult,
        PlanStepExecution,
    )
    from buildml.ai.privacy import EgressConfig, EgressLevel, EgressManifest
    from buildml.ai.provider import MockProvider, OpenAIProvider, ProviderConfig, ProviderProtocol
    from buildml.ai.results import PlanResult, TranscriptEntry
    from buildml.ai.tools import ConfirmPolicy, ToolRegistry, ToolSpec, build_default_registry
    from buildml.ai.transcript import TranscriptStore


def __getattr__(name: str) -> object:
    if name == "EgressLevel":
        from buildml.ai.privacy import EgressLevel

        return EgressLevel
    if name == "EgressConfig":
        from buildml.ai.privacy import EgressConfig

        return EgressConfig
    if name == "EgressManifest":
        from buildml.ai.privacy import EgressManifest

        return EgressManifest
    if name == "ProviderConfig":
        from buildml.ai.provider import ProviderConfig

        return ProviderConfig
    if name == "ProviderProtocol":
        from buildml.ai.provider import ProviderProtocol

        return ProviderProtocol
    if name == "OpenAIProvider":
        from buildml.ai.provider import OpenAIProvider

        return OpenAIProvider
    if name == "MockProvider":
        from buildml.ai.provider import MockProvider

        return MockProvider
    if name == "ToolSpec":
        from buildml.ai.tools import ToolSpec

        return ToolSpec
    if name == "ToolRegistry":
        from buildml.ai.tools import ToolRegistry

        return ToolRegistry
    if name == "ConfirmPolicy":
        from buildml.ai.tools import ConfirmPolicy

        return ConfirmPolicy
    if name == "AdvisorResult":
        from buildml.ai.advisor import AdvisorResult

        return AdvisorResult
    if name == "ExecutorProposal":
        from buildml.ai.executor import ExecutorProposal

        return ExecutorProposal
    if name == "ExecutorResult":
        from buildml.ai.executor import ExecutorResult

        return ExecutorResult
    if name == "PlanResult":
        from buildml.ai.results import PlanResult

        return PlanResult
    if name == "TranscriptEntry":
        from buildml.ai.results import TranscriptEntry

        return TranscriptEntry
    if name == "TranscriptStore":
        from buildml.ai.transcript import TranscriptStore

        return TranscriptStore
    if name == "BudgetTracker":
        from buildml.ai.planner import BudgetTracker

        return BudgetTracker
    if name == "BudgetExceeded":
        from buildml.ai.planner import BudgetExceeded

        return BudgetExceeded
    if name == "PlanExecutionResult":
        from buildml.ai.planner import PlanExecutionResult

        return PlanExecutionResult
    if name == "PlanStepExecution":
        from buildml.ai.planner import PlanStepExecution

        return PlanStepExecution
    if name == "build_default_registry":
        from buildml.ai.tools import build_default_registry

        return build_default_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdvisorResult",
    "BudgetExceeded",
    "BudgetTracker",
    "ConfirmPolicy",
    "EgressConfig",
    "EgressLevel",
    "EgressManifest",
    "ExecutorProposal",
    "ExecutorResult",
    "MockProvider",
    "OpenAIProvider",
    "PlanExecutionResult",
    "PlanResult",
    "PlanStepExecution",
    "ProviderConfig",
    "ProviderProtocol",
    "ToolRegistry",
    "ToolSpec",
    "TranscriptEntry",
    "TranscriptStore",
    "build_default_registry",
]
