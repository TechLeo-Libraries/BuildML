"""Dry-run previews and history/audit helpers (read-only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.explain.catalog import OPERATION_CATALOG, get_operation
from buildml.explain.history import normalize_history
from buildml.explain.resolver import prerequisite_status, resolve_workflow
from buildml.explain.schemas import WorkflowStepStatus

RiskSeverity = Literal["high", "medium", "low", "info"]


@dataclass(slots=True)
class RankedRisk:
    """One unresolved risk with actionable ranking metadata."""

    rank: int
    severity: RiskSeverity
    message: str
    source: str
    suggested_operation: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "suggested_operation": self.suggested_operation,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class PrerequisiteGraphSummary:
    """Compact prerequisite graph for previewed or suggested operations."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [dict(item) for item in self.nodes],
            "edges": [dict(item) for item in self.edges],
            "missing_required": list(self.missing_required),
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_missing_required": len(self.missing_required),
        }


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
    ranked_risks: list[RankedRisk] = field(default_factory=list)
    prerequisite_graph: PrerequisiteGraphSummary = field(default_factory=PrerequisiteGraphSummary)
    suggested_next_ops: list[dict[str, Any]] = field(default_factory=list)
    would_mutate: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "unresolved_risks": list(self.unresolved_risks),
            "ranked_risks": [item.to_dict() for item in self.ranked_risks],
            "prerequisite_graph": self.prerequisite_graph.to_dict(),
            "suggested_next_ops": [dict(item) for item in self.suggested_next_ops],
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
        if self.ranked_risks:
            print("Ranked unresolved risks:")
            for item in self.ranked_risks[:8]:
                op = f" → {item.suggested_operation}" if item.suggested_operation else ""
                print(f"! [{item.severity}] {item.message}{op}")
        else:
            for risk in self.unresolved_risks[:8]:
                print(f"! {risk}")
        if self.suggested_next_ops:
            print(
                "Suggested next ops:",
                ", ".join(str(item.get("operation")) for item in self.suggested_next_ops[:6]),
            )
        graph = self.prerequisite_graph
        if graph.missing_required:
            print("Missing required prerequisites:", "; ".join(graph.missing_required[:6]))


@dataclass(slots=True)
class HistorySummary:
    """Compact view of operation history and open risks."""

    n_operations: int
    operation_counts: dict[str, int]
    warning_count: int
    recent_operations: list[str]
    unresolved_risks: list[str]
    ranked_risks: list[RankedRisk] = field(default_factory=list)
    prerequisite_graph: PrerequisiteGraphSummary = field(default_factory=PrerequisiteGraphSummary)
    suggested_next_ops: list[dict[str, Any]] = field(default_factory=list)
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
            "ranked_risks": [item.to_dict() for item in self.ranked_risks],
            "prerequisite_graph": self.prerequisite_graph.to_dict(),
            "suggested_next_ops": [dict(item) for item in self.suggested_next_ops],
            "decision_origins": dict(self.decision_origins),
            "has_split": self.has_split,
            "has_fit": self.has_fit,
        }

    def show(self) -> None:
        print(
            f"History summary · {self.n_operations} operation(s), {self.warning_count} warning(s)"
        )
        print("Recent:", ", ".join(self.recent_operations) or "(empty)")
        if self.ranked_risks:
            for item in self.ranked_risks[:10]:
                op = f" → {item.suggested_operation}" if item.suggested_operation else ""
                print(f"! [{item.severity}] {item.message}{op}")
        else:
            for risk in self.unresolved_risks[:10]:
                print(f"! {risk}")
        if self.suggested_next_ops:
            print(
                "Suggested next ops:",
                ", ".join(str(item.get("operation")) for item in self.suggested_next_ops[:6]),
            )


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

    ranked = rank_unresolved_risks(session)
    risks = [item.message for item in ranked]
    graph = build_prerequisite_graph_summary(session, ordered)
    suggestions = suggest_next_operations(session, limit=8)
    notes = [
        "Dry-run does not fit, transform, or append history.",
        "Availability means API prerequisites pass, not that the operation is appropriate.",
        "Ranked risks and suggested next ops are heuristic review cues, not automatic approvals.",
    ]
    if any(not step.available for step in steps):
        notes.append("Blocked steps list missing prerequisites; resolve them before execution.")
    if graph.missing_required:
        notes.append(
            "Prerequisite graph lists missing required capabilities for the previewed operations."
        )
    return DryRunReport(
        steps=steps,
        unresolved_risks=risks,
        ranked_risks=ranked,
        prerequisite_graph=graph,
        suggested_next_ops=suggestions,
        would_mutate=False,
        notes=notes,
    )


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
    recent = [str(record.get("operation_id") or record.get("action")) for record in history[-8:]]
    split = getattr(session, "_split_plan", None) is not None
    fit = getattr(session, "_fit_result", None) is not None
    ranked = rank_unresolved_risks(session)
    suggestions = suggest_next_operations(session, limit=8)
    focus_ops = [str(item.get("operation")) for item in suggestions[:6]]
    if not focus_ops:
        focus_ops = recent[-4:]
    return HistorySummary(
        n_operations=len(history),
        operation_counts=dict(sorted(counts.items())),
        warning_count=warning_count,
        recent_operations=recent,
        unresolved_risks=[item.message for item in ranked],
        ranked_risks=ranked,
        prerequisite_graph=build_prerequisite_graph_summary(session, focus_ops),
        suggested_next_ops=suggestions,
        decision_origins=origins,
        has_split=split,
        has_fit=fit,
    )


def collect_unresolved_risks(session: Any) -> list[str]:
    """Heuristic open risks from state, history warnings, and catalog notes."""
    return [item.message for item in rank_unresolved_risks(session)]


def rank_unresolved_risks(session: Any) -> list[RankedRisk]:
    """Rank open risks by likely workflow impact with suggested follow-ups."""
    raw: list[tuple[int, RiskSeverity, str, str, str | None, str]] = []
    dataset = getattr(session, "_dataset", None)
    split = getattr(session, "_split_plan", None)
    history = normalize_history(getattr(session, "_history", None))
    ops = {str(r.get("operation_id") or r.get("action")) for r in history}

    if (
        dataset is not None
        and split is None
        and any(
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
                "fit_clusters",
                "fit_voting",
                "fit_stacking",
                "fit_blending",
                "run_automl",
                "fit_forecast",
                "analyze_timeseries",
                "fit_anomaly",
                "fit_semisupervised",
                "fit_ssl_pretext",
                "finetune_ssl_head",
                "fit_active_learner",
                "fit_online",
                "partial_fit_online",
                "fit_multitask",
                "fit_metalearning",
                "fit_federated",
                "fit_probabilistic",
                "fit_causal",
                "fit_graph",
                "fit_symbolic",
                "fit_neuro_symbolic",
                "fit_cbr",
                "fit_imitation",
                "fit_rl",
                "fit_tda",
                "fit_recommender",
                "fit_ranker",
                "fit_kg",
                "fit_decision_policy",
                "fit_synthesizer",
                "fit",
            )
        )
    ):
        raw.append(
            (
                0,
                "high",
                (
                    "History records fit-capable operations but no SplitPlan is attached; "
                    "reattach or recreate partitions before trusting scores."
                ),
                "state.split",
                "split",
                "Scores without a stored split cannot be attributed to a holdout policy.",
            )
        )

    if getattr(session, "_fit_result", None) is not None and "evaluate" not in ops:
        raw.append(
            (
                1,
                "medium",
                "An estimator is fitted but evaluate has not been recorded in history yet.",
                "state.fit",
                "evaluate",
                "Record a holdout evaluation before comparing or exporting claims.",
            )
        )

    if getattr(session, "_session_preprocess_applied", lambda: False)():
        if (
            "cv_score" in ops
            or "nested_cv_score" in ops
            or "grid_search" in ops
            or "randomized_search" in ops
            or "optuna_search" in ops
            or "evolutionary_search" in ops
            or "run_automl" in ops
        ):
            raw.append(
                (
                    2,
                    "high",
                    (
                        "Session-global preprocess plans exist alongside CV/search history. "
                        "Default path refuses even when a fold-local PreprocessRecipe is "
                        "passed (recipes do not rebuild from raw rows). Opt in only via "
                        "allow_session_global_preprocess=True, or re-ingest unpoisoned data."
                    ),
                    "history.cv_preprocess_scope",
                    "cv_score",
                    "Re-ingest unpoisoned data, then use fold-local PreprocessRecipe.",
                )
            )

    for record in history:
        for warning in record.get("warnings") or []:
            text = str(warning)
            if "leak" in text.lower() or "lineage" in text.lower():
                op_id = str(record.get("operation_id") or record.get("action") or "history")
                raw.append(
                    (
                        0,
                        "high",
                        f"History warning ({op_id}): {text}",
                        f"history.{op_id}",
                        op_id if op_id in OPERATION_CATALOG else "walkthrough",
                        "Leakage-related warnings stay open until the workflow path is corrected.",
                    )
                )

    last = getattr(session, "_last_preprocess", None)
    if last is not None:
        for finding in getattr(last, "findings", []) or []:
            severity = getattr(finding, "severity", None)
            value = getattr(severity, "value", severity)
            if str(value) in {"high", "critical"}:
                sev: RiskSeverity = "high" if str(value) == "critical" else "medium"
                raw.append(
                    (
                        1 if sev == "high" else 3,
                        sev,
                        (
                            f"Unresolved preprocess finding '{getattr(finding, 'key', finding)}' "
                            f"(severity={value})."
                        ),
                        "preprocess.findings",
                        "summarize_history",
                        "High-severity preprocess findings should be reviewed before fit claims.",
                    )
                )

    if getattr(session, "_fit_result", None) is not None and (
        "tune_threshold" not in ops and "fit_decision_policy" not in ops
    ):
        fit = getattr(session, "_fit_result", None)
        if getattr(fit, "task", None) == "classification":
            raw.append(
                (
                    4,
                    "low",
                    (
                        "A classification fit exists without a recorded threshold / "
                        "decision-policy step; default 0.5 may not match decision costs."
                    ),
                    "state.threshold_policy",
                    "fit_decision_policy",
                    "Prefer fit_decision_policy(method='threshold', partition='validation', "
                    "fp_cost=..., fn_cost=...) or classical tune_threshold on validation.",
                )
            )

    # De-duplicate by message while preserving severity order.
    severity_rank = {"high": 0, "medium": 1, "low": 2, "info": 3}
    seen: set[str] = set()
    unique: list[tuple[int, RiskSeverity, str, str, str | None, str]] = []
    for item in sorted(raw, key=lambda row: (severity_rank[row[1]], row[0], row[2])):
        if item[2] in seen:
            continue
        seen.add(item[2])
        unique.append(item)

    return [
        RankedRisk(
            rank=index + 1,
            severity=severity,
            message=message,
            source=source,
            suggested_operation=suggested,
            rationale=rationale,
        )
        for index, (_priority, severity, message, source, suggested, rationale) in enumerate(unique)
    ]


def suggest_next_operations(session: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    """Suggest concrete next Session operations from the workflow resolver."""
    workflow = resolve_workflow(session)
    suggestions: list[dict[str, Any]] = []
    for step in workflow:
        status = step.status.value if hasattr(step.status, "value") else str(step.status)
        if status not in {
            WorkflowStepStatus.AVAILABLE.value,
            WorkflowStepStatus.READY.value,
            "available",
            "ready",
        }:
            continue
        if step.operation in {
            "dry_run",
            "summarize_history",
            "walkthrough",
            "explain",
            "workflow",
            "metadata",
            "head",
            "list_transforms",
            "resample_strategies",
        }:
            continue
        reason = step.reasons[0] if step.reasons else step.summary
        suggestions.append(
            {
                "operation": step.operation,
                "status": status,
                "reason": reason,
                "api_action": f"Session.explain({step.operation!r}, moment='before')",
                "evidence": f"workflow-{step.operation}",
            }
        )
    ordered = _priority_order([str(item["operation"]) for item in suggestions])
    by_name = {str(item["operation"]): item for item in suggestions}
    return [by_name[name] for name in ordered if name in by_name][:limit]


def build_prerequisite_graph_summary(
    session: Any,
    operations: Sequence[str],
) -> PrerequisiteGraphSummary:
    """Summarize prerequisite nodes/edges for the given operations."""
    workflow = {step.operation: step for step in resolve_workflow(session)}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    missing: list[str] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    for operation in operations:
        try:
            spec = get_operation(operation)
        except Exception:
            continue
        step = workflow.get(operation)
        status = (
            step.status.value
            if step is not None and hasattr(step.status, "value")
            else (str(step.status) if step is not None else "unknown")
        )
        if operation not in seen_nodes:
            nodes.append(
                {
                    "operation": operation,
                    "status": status,
                    "available": status
                    in {
                        WorkflowStepStatus.AVAILABLE.value,
                        WorkflowStepStatus.READY.value,
                        WorkflowStepStatus.DONE.value,
                        "available",
                        "ready",
                        "done",
                    },
                }
            )
            seen_nodes.add(operation)
        prereq_state = prerequisite_status(session, operation)
        for prerequisite in spec.prerequisites:
            key = prerequisite.key
            passed = bool(prereq_state.get(key, False))
            detail = (
                f"Prerequisite '{key}' is satisfied."
                if passed
                else f"Prerequisite '{key}' is not satisfied."
            )
            if key not in seen_nodes:
                nodes.append(
                    {
                        "operation": key,
                        "status": "satisfied" if passed else "missing",
                        "available": passed,
                        "kind": "prerequisite",
                        "detail": detail,
                        "required": prerequisite.status.value == "required",
                    }
                )
                seen_nodes.add(key)
            if not passed and prerequisite.status.value == "required":
                missing.append(f"{operation} requires {key}: {detail}")
            edge = (key, operation)
            if edge not in seen_edges:
                edges.append({"from": key, "to": operation, "via": "prerequisite"})
                seen_edges.add(edge)
            providers = {
                "dataset": ("ingest", "checkpoint_load", "reattach"),
                "roles": ("set_roles",),
                "split": ("split", "inject_split", "group_split", "time_split"),
                "fit": (
                    "fit",
                    "compare_models",
                    "load_model",
                    "load_pipeline",
                    "grid_search",
                    "randomized_search",
                    "optuna_search",
                    "evolutionary_search",
                ),
            }.get(key, ())
            for provider in providers:
                provider_edge = (provider, key)
                if provider_edge not in seen_edges:
                    edges.append({"from": provider, "to": key, "via": "provides"})
                    seen_edges.add(provider_edge)
                if provider not in seen_nodes:
                    provider_step = workflow.get(provider)
                    provider_status = (
                        provider_step.status.value
                        if provider_step is not None and hasattr(provider_step.status, "value")
                        else (str(provider_step.status) if provider_step is not None else "catalog")
                    )
                    nodes.append(
                        {
                            "operation": provider,
                            "status": provider_status,
                            "available": provider_status
                            in {
                                WorkflowStepStatus.AVAILABLE.value,
                                WorkflowStepStatus.READY.value,
                                WorkflowStepStatus.DONE.value,
                                "available",
                                "ready",
                                "done",
                            },
                            "kind": "provider",
                        }
                    )
                    seen_nodes.add(provider)

    # Preserve insertion order; keep missing list unique.
    unique_missing: list[str] = []
    seen_missing: set[str] = set()
    for item in missing:
        if item not in seen_missing:
            unique_missing.append(item)
            seen_missing.add(item)
    return PrerequisiteGraphSummary(
        nodes=nodes,
        edges=edges,
        missing_required=unique_missing,
    )


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

    prereq = [f"{item.key}: {item.description}" for item in spec.prerequisites]
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
        "fit_clusters",
        "evaluate_clusters",
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "evaluate_ensemble",
        "run_automl",
        "evaluate_automl",
        "fit_forecast",
        "generate_forecast",
        "evaluate_forecast",
        "analyze_timeseries",
        "ts_decompose",
        "ts_diagnostics",
        "fit_anomaly",
        "score_anomalies",
        "evaluate_anomaly",
        "fit_semisupervised",
        "predict_semisupervised",
        "evaluate_semisupervised",
        "fit_ssl_pretext",
        "transform_ssl",
        "finetune_ssl_head",
        "evaluate_ssl",
        "fit_active_learner",
        "suggest_query",
        "label_rows",
        "evaluate_active_learning",
        "fit_online",
        "partial_fit_online",
        "predict_online",
        "evaluate_online",
        "fit_multitask",
        "predict_multitask",
        "evaluate_multitask",
        "fit_metalearning",
        "adapt_to_task",
        "evaluate_metalearning",
        "fit_federated",
        "predict_federated",
        "evaluate_federated",
        "fit_probabilistic",
        "predict_probabilistic",
        "predict_interval",
        "evaluate_probabilistic",
        "declare_causal_assumptions",
        "fit_causal",
        "estimate_causal",
        "evaluate_causal",
        "refute_causal",
        "set_graph",
        "fit_graph",
        "predict_graph",
        "evaluate_graph",
        "fit_symbolic",
        "predict_symbolic",
        "evaluate_symbolic",
        "fit_neuro_symbolic",
        "predict_neuro_symbolic",
        "evaluate_neuro_symbolic",
        "fit_cbr",
        "retrieve_cases",
        "predict_cbr",
        "evaluate_cbr",
        "retain_cbr",
        "fit_imitation",
        "predict_imitation_action",
        "evaluate_imitation",
        "fit_rl",
        "act_rl",
        "evaluate_rl",
        "fit_tda",
        "transform_tda",
        "predict_tda",
        "evaluate_tda",
        "fit_recommender",
        "recommend",
        "evaluate_recommender",
        "fit_ranker",
        "rank",
        "evaluate_ranker",
        "fit_kg",
        "score_triples",
        "predict_links",
        "query_kg",
        "evaluate_kg",
        "fit_decision_policy",
        "apply_decisions",
        "evaluate_decisions",
        "fit_synthesizer",
        "sample_synthetic",
        "evaluate_synthetic",
        "fit",
        "evaluate",
        "calibration",
        "tune_threshold",
        "error_slices",
        "feature_importance",
        "save_pipeline",
        "checkpoint_save",
        "dry_run",
        "summarize_history",
        "walkthrough",
    )
    rank = {name: index for index, name in enumerate(priority)}
    return sorted(names, key=lambda item: (rank.get(item, 1000), item))
