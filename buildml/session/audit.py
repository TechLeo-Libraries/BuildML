"""Dry-run previews and history/audit helpers (read-only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from buildml.explain.catalog import OPERATION_CATALOG, get_operation
from buildml.explain.history import normalize_history
from buildml.explain.resolver import resolve_workflow
from buildml.explain.schemas import WorkflowStepStatus


@dataclass(slots=True)
class DryRunStep:
    """One operation preview inside a dry-run report."""

    operation: str
    available: bool
    status: str
    parameters: dict[str, Any] = field(default_factory=dict)
    blocked_reasons: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    leakage_risks: list[str] = field(default_factory=list)
    estimated_effects: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "available": self.available,
            "status": self.status,
            "parameters": dict(self.parameters),
            "blocked_reasons": list(self.blocked_reasons),
            "prerequisites": list(self.prerequisites),
            "leakage_risks": list(self.leakage_risks),
            "estimated_effects": list(self.estimated_effects),
            "anti_patterns": list(self.anti_patterns),
        }


@dataclass(slots=True)
class DryRunReport:
    """Non-mutating preview of intended Session operations."""

    steps: list[DryRunStep]
    unresolved_risks: list[str] = field(default_factory=list)
    would_mutate: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "unresolved_risks": list(self.unresolved_risks),
            "would_mutate": self.would_mutate,
            "notes": list(self.notes),
            "n_available": sum(1 for step in self.steps if step.available),
            "n_blocked": sum(1 for step in self.steps if not step.available),
        }

    def show(self) -> None:
        print("Dry-run preview (Session state is unchanged)")
        for step in self.steps:
            mark = "ok" if step.available else "blocked"
            print(f"[{mark}] {step.operation}")
            for reason in step.blocked_reasons[:4]:
                print(f"  - {reason}")
            for effect in step.estimated_effects[:3]:
                print(f"  ~ {effect}")
        for risk in self.unresolved_risks[:8]:
            print(f"! {risk}")


@dataclass(slots=True)
class HistorySummary:
    """Compact view of operation history and open risks."""

    n_operations: int
    operation_counts: dict[str, int]
    warning_count: int
    recent_operations: list[str]
    unresolved_risks: list[str]
    decision_origins: dict[str, int] = field(default_factory=dict)
    has_split: bool = False
    has_fit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_operations": self.n_operations,
            "operation_counts": dict(self.operation_counts),
            "warning_count": self.warning_count,
            "recent_operations": list(self.recent_operations),
            "unresolved_risks": list(self.unresolved_risks),
            "decision_origins": dict(self.decision_origins),
            "has_split": self.has_split,
            "has_fit": self.has_fit,
        }

    def show(self) -> None:
        print(
            f"History summary · {self.n_operations} operation(s), "
            f"{self.warning_count} warning(s)"
        )
        print("Recent:", ", ".join(self.recent_operations) or "(empty)")
        for risk in self.unresolved_risks[:10]:
            print(f"! {risk}")


def dry_run_session(
    session: Any,
    operation: str | Sequence[str] | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> DryRunReport:
    """Preview operations against current Session state without mutation."""
    workflow = {step.operation: step for step in resolve_workflow(session)}
    if operation is None:
        # Default: available + blocked prep/fit-adjacent steps that matter next.
        candidates = [
            step.operation
            for step in workflow.values()
            if step.status
            in {
                WorkflowStepStatus.AVAILABLE,
                WorkflowStepStatus.BLOCKED,
                WorkflowStepStatus.READY,
            }
            and OPERATION_CATALOG.get(step.operation)
            and OPERATION_CATALOG[step.operation].kind.value
            in {"transform", "model", "split", "inspect", "diagnostic"}
        ]
        # Keep the preview focused.
        ordered = _priority_order(candidates)[:12]
    elif isinstance(operation, str):
        ordered = [operation]
    else:
        ordered = [str(item) for item in operation]

    params = dict(parameters or {})
    steps: list[DryRunStep] = []
    step_params = params if len(ordered) == 1 else {}
    for name in ordered:
        steps.append(_preview_step(session, name, workflow.get(name), step_params))

    risks = collect_unresolved_risks(session)
    notes = [
        "Dry-run does not fit, transform, or append history.",
        "Availability means API prerequisites pass, not that the operation is appropriate.",
    ]
    if any(not step.available for step in steps):
        notes.append("Blocked steps list missing prerequisites; resolve them before execution.")
    return DryRunReport(steps=steps, unresolved_risks=risks, would_mutate=False, notes=notes)


def summarize_history(session: Any) -> HistorySummary:
    """Summarize Session history and surface unresolved risks."""
    raw_history = getattr(session, "history", None) or getattr(session, "_history", None)
    history = normalize_history(raw_history)
    counts: dict[str, int] = {}
    origins: dict[str, int] = {}
    warning_count = 0
    for record in history:
        op = str(record.get("operation_id") or record.get("action") or "unknown")
        counts[op] = counts.get(op, 0) + 1
        origin = str(record.get("decision_origin") or "explicit")
        origins[origin] = origins.get(origin, 0) + 1
        warning_count += len(record.get("warnings") or [])
    recent = [
        str(record.get("operation_id") or record.get("action"))
        for record in history[-8:]
    ]
    split = getattr(session, "_split_plan", None) is not None
    fit = getattr(session, "_fit_result", None) is not None
    return HistorySummary(
        n_operations=len(history),
        operation_counts=dict(sorted(counts.items())),
        warning_count=warning_count,
        recent_operations=recent,
        unresolved_risks=collect_unresolved_risks(session),
        decision_origins=origins,
        has_split=split,
        has_fit=fit,
    )


def collect_unresolved_risks(session: Any) -> list[str]:
    """Heuristic open risks from state, history warnings, and catalog notes."""
    risks: list[str] = []
    dataset = getattr(session, "_dataset", None)
    split = getattr(session, "_split_plan", None)
    history = normalize_history(getattr(session, "_history", None))
    ops = {str(r.get("operation_id") or r.get("action")) for r in history}

    if dataset is not None and split is None and any(
        name in ops
        for name in (
            "impute",
            "encode",
            "scale",
            "bin",
            "select_features",
            "text_features",
            "reduce_dimensions",
            "apply_custom_transform",
            "fit",
        )
    ):
        risks.append(
            "History records fit-capable operations but no SplitPlan is attached; "
            "reattach or recreate partitions before trusting scores."
        )

    if getattr(session, "_fit_result", None) is not None and "evaluate" not in ops:
        risks.append(
            "An estimator is fitted but evaluate has not been recorded in history yet."
        )

    if getattr(session, "_session_preprocess_applied", lambda: False)():
        if "cv_score" in ops or "nested_cv_score" in ops or "grid_search" in ops:
            risks.append(
                "Session-global preprocess plans exist alongside CV/search history; "
                "fold-local PreprocessRecipe is safer when the recipe itself is tuned."
            )

    for record in history:
        for warning in record.get("warnings") or []:
            text = str(warning)
            if "leak" in text.lower() or "lineage" in text.lower():
                risks.append(f"History warning ({record.get('operation_id')}): {text}")

    last = getattr(session, "_last_preprocess", None)
    if last is not None:
        for finding in getattr(last, "findings", []) or []:
            severity = getattr(finding, "severity", None)
            value = getattr(severity, "value", severity)
            if str(value) in {"high", "critical"}:
                risks.append(
                    f"Unresolved preprocess finding '{getattr(finding, 'key', finding)}' "
                    f"(severity={value})."
                )

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in risks:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _preview_step(
    session: Any,
    operation: str,
    workflow_step: Any | None,
    parameters: Mapping[str, Any],
) -> DryRunStep:
    try:
        spec = get_operation(operation)
    except Exception:
        return DryRunStep(
            operation=operation,
            available=False,
            status="unknown",
            parameters=dict(parameters),
            blocked_reasons=[f"'{operation}' is not a catalogued Session operation."],
            estimated_effects=[],
        )

    if workflow_step is None:
        status = "unknown"
        available = False
        blocked = [f"No workflow resolution for '{operation}'."]
    else:
        status = (
            workflow_step.status.value
            if hasattr(workflow_step.status, "value")
            else str(workflow_step.status)
        )
        available = status in {
            WorkflowStepStatus.AVAILABLE.value,
            WorkflowStepStatus.READY.value,
            WorkflowStepStatus.DONE.value,
            "available",
            "ready",
            "done",
        }
        # Done means it already ran; dry-run still allows re-preview as available-ish.
        if status in {WorkflowStepStatus.DONE.value, "done"}:
            available = True
        blocked = list(getattr(workflow_step, "blockers", ()) or ())
        if not blocked:
            blocked = list(getattr(workflow_step, "reasons", ()) or ())
        if status in {WorkflowStepStatus.BLOCKED.value, "blocked"}:
            available = False

    prereq = [
        f"{item.key}: {item.description}"
        for item in spec.prerequisites
    ]
    effects = list(spec.state_changes) + list(spec.outputs)[:2]
    if parameters:
        effects = [f"Requested parameters: {dict(parameters)}", *effects]
    return DryRunStep(
        operation=operation,
        available=available,
        status=status,
        parameters=dict(parameters),
        blocked_reasons=blocked if not available else [],
        prerequisites=prereq,
        leakage_risks=list(spec.leakage_risks),
        estimated_effects=effects[:6],
        anti_patterns=list(spec.anti_patterns)[:4],
    )


def _priority_order(names: Sequence[str]) -> list[str]:
    priority = (
        "split",
        "inject_split",
        "group_split",
        "time_split",
        "impute",
        "handle_outliers",
        "encode",
        "text_features",
        "bin",
        "scale",
        "reduce_dimensions",
        "select_features",
        "apply_custom_transform",
        "resample",
        "fit",
        "evaluate",
        "calibration",
        "feature_importance",
        "save_pipeline",
        "checkpoint_save",
        "dry_run",
        "summarize_history",
        "walkthrough",
    )
    rank = {name: index for index, name in enumerate(priority)}
    return sorted(names, key=lambda item: (rank.get(item, 1000), item))
