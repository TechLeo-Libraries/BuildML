# ruff: noqa: E501
"""Offline workflow walkthroughs built from resolver state and history."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from buildml.explain.catalog import OPERATION_CATALOG, get_operation
from buildml.explain.schemas import WorkflowStep
from buildml.reporting.html import (
    ReportSection,
    element_id,
    escape,
    render_badge,
    render_reading_frame,
    render_report,
    render_table,
)


@dataclass(slots=True)
class WorkflowWalkthroughReport:
    """Serializable walkthrough of live workflow state and observed choices."""

    workflow: tuple[WorkflowStep, ...]
    timeline: list[dict[str, Any]]
    status_counts: dict[str, int]
    unusual_order: list[dict[str, str]]
    unresolved_risks: list[dict[str, str]]
    concept_links: list[dict[str, str]]
    next_actions: list[dict[str, Any]]
    engine_status: dict[str, Any] = field(default_factory=dict)
    warm_start_status: dict[str, Any] = field(default_factory=dict)
    preprocess_scope_status: dict[str, Any] = field(default_factory=dict)
    torch_training_status: dict[str, Any] = field(default_factory=dict)
    audit_summary: dict[str, Any] = field(default_factory=dict)
    html_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": [step.to_dict() for step in self.workflow],
            "timeline": list(self.timeline),
            "status_counts": dict(self.status_counts),
            "unusual_order": list(self.unusual_order),
            "unresolved_risks": list(self.unresolved_risks),
            "concept_links": list(self.concept_links),
            "next_actions": list(self.next_actions),
            "engine_status": dict(self.engine_status),
            "warm_start_status": dict(self.warm_start_status),
            "preprocess_scope_status": dict(self.preprocess_scope_status),
            "torch_training_status": dict(self.torch_training_status),
            "audit_summary": dict(self.audit_summary),
            "html_path": self.html_path,
        }

    def export_html(self, path: str | Path) -> Path:
        """Write the walkthrough with the shared offline report shell."""
        destination = export_walkthrough_html(self.to_dict(), path)
        self.html_path = str(destination)
        return destination


def build_walkthrough(session: Any) -> WorkflowWalkthroughReport:
    """Resolve statuses and join them to the session's versioned history."""
    from buildml.session.audit import summarize_history

    workflow = tuple(session.workflow())
    history = list(session.history)
    timeline = [_timeline_row(record) for record in history]
    counts = Counter(step.status.value for step in workflow)
    status_counts = {
        status: int(counts.get(status, 0)) for status in ("done", "available", "blocked", "skipped")
    }
    unusual = _unusual_order(history)
    risks, concepts = _risks_and_concepts(workflow, history)
    audit = summarize_history(session)
    next_actions = list(audit.suggested_next_ops) or [
        {
            "operation": step.operation,
            "status": step.status.value,
            "reason": step.reasons[0] if step.reasons else step.summary,
            "api_action": f"Session.explain({step.operation!r}, moment='before')",
            "evidence": f"workflow-{step.operation}",
        }
        for step in workflow
        if step.status.value == "available"
    ]
    return WorkflowWalkthroughReport(
        workflow=workflow,
        timeline=timeline,
        status_counts=status_counts,
        unusual_order=unusual,
        unresolved_risks=risks,
        concept_links=concepts,
        next_actions=next_actions,
        engine_status=_engine_status(session),
        warm_start_status=warm_start_studies_status(
            history,
            last_nested_cv=getattr(session, "last_nested_cv", None),
        ),
        preprocess_scope_status=preprocess_scope_status(
            history,
            session=session,
            last_cv=getattr(session, "last_cv", None),
            last_nested_cv=getattr(session, "last_nested_cv", None),
        ),
        torch_training_status=torch_training_status_for_walkthrough(session),
        audit_summary={
            "n_operations": audit.n_operations,
            "warning_count": audit.warning_count,
            "ranked_risks": [item.to_dict() for item in audit.ranked_risks],
            "prerequisite_graph": audit.prerequisite_graph.to_dict(),
            "suggested_next_ops": list(audit.suggested_next_ops),
            "has_split": audit.has_split,
            "has_fit": audit.has_fit,
        },
    )


def torch_training_status_for_walkthrough(session: Any) -> dict[str, Any]:
    """Factual Torch training-curve / early-stop / device disclosure."""
    from buildml.dl.curves import torch_training_status

    return torch_training_status(
        train_result=getattr(session, "dl_train_result", None),
        history=list(getattr(session, "history", []) or []),
    )


def warm_start_studies_status(
    history: list[dict[str, Any]] | None = None,
    *,
    last_nested_cv: Any | None = None,
) -> dict[str, Any]:
    """Factual disclosure when nested CV used ``warm_start_studies``.

    Inspects Session history and optional ``last_nested_cv``. When the flag was
    never enabled, returns ``enabled=False`` with an empty disclosure list.
    """
    records = list(history or [])
    enabled = False
    search_method: str | None = None
    n_outer: int | None = None
    sequences: list[int] = []
    held_out: list[str] = []
    limitations: list[str] = []

    for record in records:
        operation = str(record.get("operation_id") or record.get("action") or "")
        if operation != "nested_cv_score":
            continue
        params = record.get("parameters") or {}
        summary = record.get("result_summary") or {}
        flag = params.get("warm_start_studies")
        if flag is None:
            flag = summary.get("warm_start_studies")
        if not flag:
            continue
        enabled = True
        seq = record.get("sequence")
        if seq is not None:
            sequences.append(int(seq))
        search_method = (
            str(summary.get("search_method") or params.get("search_method") or search_method or "")
            or None
        )
        if summary.get("n_outer_splits") is not None:
            n_outer = int(summary["n_outer_splits"])

    if last_nested_cv is not None and bool(getattr(last_nested_cv, "warm_start_studies", False)):
        enabled = True
        search_method = str(getattr(last_nested_cv, "search_method", None) or search_method or "") or None
        n_outer = int(getattr(last_nested_cv, "n_outer_splits", n_outer or 0) or n_outer or 0) or n_outer
        held_out = [str(x) for x in (getattr(last_nested_cv, "held_out_partitions", None) or [])]
        limitations = [
            str(x)
            for x in (getattr(last_nested_cv, "limitations", None) or [])
            if "warm_start" in str(x).lower()
        ]

    if not enabled:
        return {
            "enabled": False,
            "present_in_history": False,
            "search_method": None,
            "n_outer_splits": None,
            "history_sequences": [],
            "held_out_partitions": [],
            "shared": None,
            "disclosures": [],
        }

    disclosures = [
        "warm_start_studies=True was recorded for nested_cv_score.",
        "Policy: one Optuna study was shared across outer folds so later folds "
        "could reuse prior inner-CV trial history as priors.",
        "What was shared: Optuna trial history / study state only — not outer-eval "
        "rows, Session test rows, or Session validation rows.",
        "Risk: shared priors couple outer folds; the outer mean±std is still the "
        "post-selection estimate, but fold independence of the inner search is reduced.",
        "Session test/validation partitions stay outside both loops when warm start is on.",
    ]
    if search_method:
        disclosures.append(f"Inner search_method={search_method}.")
    if n_outer is not None:
        disclosures.append(f"Outer folds observed: n_outer_splits={n_outer}.")
    if held_out:
        disclosures.append(f"Held-out Session partitions: {', '.join(held_out)}.")
    disclosures.extend(limitations)

    return {
        "enabled": True,
        "present_in_history": bool(sequences) or bool(last_nested_cv),
        "search_method": search_method,
        "n_outer_splits": n_outer,
        "history_sequences": sequences,
        "held_out_partitions": held_out,
        "shared": "optuna_study_trial_history",
        "disclosures": disclosures,
    }


_CV_OPS_WITH_FOLD_RECIPE = frozenset(
    {
        "cv_score",
        "nested_cv_score",
        "grid_search",
        "random_search",
        "optuna_search",
    }
)
_SESSION_GLOBAL_PREPROCESS_OPS = frozenset(
    {
        "text_features",
        "reduce_dimensions",
        "apply_custom_transform",
        "resample",
    }
)


def preprocess_scope_status(
    history: list[dict[str, Any]] | None = None,
    *,
    session: Any | None = None,
    last_cv: Any | None = None,
    last_nested_cv: Any | None = None,
) -> dict[str, Any]:
    """Factual fold-local vs Session-global preprocess disclosure.

    Surfaces ``PreprocessRecipe`` text/PCA when recorded in CV history or live
    results, and clarifies that custom transforms and resample stay
    Session-global (not fold-local).
    """
    from buildml.preprocess.fold import SESSION_GLOBAL_ONLY_STEPS

    records = list(history or [])
    fold_text: str | None = None
    fold_reduce: str | None = None
    fold_sequences: list[int] = []
    fold_ops: list[str] = []
    session_ops: dict[str, list[int]] = {name: [] for name in _SESSION_GLOBAL_PREPROCESS_OPS}

    for record in records:
        operation = str(record.get("operation_id") or record.get("action") or "")
        params = record.get("parameters") or {}
        seq = record.get("sequence")
        seq_i = int(seq) if seq is not None else None

        if operation in _CV_OPS_WITH_FOLD_RECIPE:
            recipe = params.get("fold_preprocess") or {}
            if not isinstance(recipe, dict):
                continue
            text = recipe.get("text")
            reduce = recipe.get("reduce")
            if text is None and reduce is None:
                continue
            if text is not None:
                fold_text = str(text)
            if reduce is not None:
                fold_reduce = str(reduce)
            if seq_i is not None:
                fold_sequences.append(seq_i)
            if operation not in fold_ops:
                fold_ops.append(operation)

        if operation in _SESSION_GLOBAL_PREPROCESS_OPS and seq_i is not None:
            session_ops[operation].append(seq_i)

    for result in (last_cv, last_nested_cv):
        if result is None:
            continue
        recipe = getattr(result, "fold_preprocess", None) or {}
        if not isinstance(recipe, dict):
            continue
        if recipe.get("text") is not None:
            fold_text = str(recipe["text"])
        if recipe.get("reduce") is not None:
            fold_reduce = str(recipe["reduce"])

    live_session: dict[str, bool] = {
        "text_features": False,
        "reduce_dimensions": False,
        "apply_custom_transform": False,
        "resample": False,
    }
    if session is not None:
        live_session["text_features"] = getattr(session, "_text_plan", None) is not None
        live_session["reduce_dimensions"] = getattr(session, "_reduce_plan", None) is not None
        live_session["apply_custom_transform"] = getattr(session, "_custom_plan", None) is not None
        live_session["resample"] = getattr(session, "_resample_plan", None) is not None

    fold_present = fold_text is not None or fold_reduce is not None
    session_present = any(session_ops[name] or live_session[name] for name in session_ops)
    present = fold_present or session_present

    if not present:
        return {
            "enabled": False,
            "present": False,
            "fold_local": {"text": None, "reduce": None, "operations": [], "history_sequences": []},
            "session_global": {
                "text_features": False,
                "reduce_dimensions": False,
                "apply_custom_transform": False,
                "resample": False,
                "history_sequences": {},
            },
            "session_global_only": list(SESSION_GLOBAL_ONLY_STEPS),
            "disclosures": [],
        }

    disclosures: list[str] = []
    if fold_text is not None:
        disclosures.append(
            f"Fold-local PreprocessRecipe text={fold_text!r}: vectorizer vocabulary/IDF "
            "fit on fold-train documents only; fold-eval rows use the frozen mapping."
        )
    if fold_reduce is not None:
        disclosures.append(
            f"Fold-local PreprocessRecipe reduce={fold_reduce!r}: PCA fits the rotation "
            "on fold-train numeric columns only; fold-eval rows use the frozen components."
        )
    if live_session["text_features"] or session_ops["text_features"]:
        disclosures.append(
            "Session.text_features fitted a Session-global text plan on the train "
            "partition. Prefer PreprocessRecipe(text=...) inside CV when selection "
            "honesty requires fold-local vocabulary/IDF."
        )
    if live_session["reduce_dimensions"] or session_ops["reduce_dimensions"]:
        disclosures.append(
            "Session.reduce_dimensions fitted a Session-global PCA plan on the train "
            "partition. Prefer PreprocessRecipe(reduce='pca') inside CV when selection "
            "honesty requires fold-local components."
        )
    if live_session["apply_custom_transform"] or session_ops["apply_custom_transform"]:
        disclosures.append(
            "Session.apply_custom_transform stays Session-global: registered callables "
            "are not part of PreprocessRecipe and are not refit per CV fold."
        )
    if live_session["resample"] or session_ops["resample"]:
        disclosures.append(
            "Session.resample stays Session-global: train-row rewrite is not applied "
            "inside CV folds (lineage-only at score time)."
        )
    if not fold_present and session_present:
        disclosures.append(
            "No fold-local PreprocessRecipe text/reduce was recorded on recent CV/search "
            "calls; Session-global plans above apply to the fitted Session path."
        )
    disclosures.append(
        "Session-global-only steps (not fold-local via PreprocessRecipe): "
        + "; ".join(SESSION_GLOBAL_ONLY_STEPS)
        + "."
    )

    return {
        "enabled": True,
        "present": True,
        "fold_local": {
            "text": fold_text,
            "reduce": fold_reduce,
            "operations": fold_ops,
            "history_sequences": fold_sequences,
        },
        "session_global": {
            "text_features": bool(live_session["text_features"] or session_ops["text_features"]),
            "reduce_dimensions": bool(
                live_session["reduce_dimensions"] or session_ops["reduce_dimensions"]
            ),
            "apply_custom_transform": bool(
                live_session["apply_custom_transform"] or session_ops["apply_custom_transform"]
            ),
            "resample": bool(live_session["resample"] or session_ops["resample"]),
            "history_sequences": {k: v for k, v in session_ops.items() if v},
        },
        "session_global_only": list(SESSION_GLOBAL_ONLY_STEPS),
        "disclosures": disclosures,
    }


def export_walkthrough_html(report: dict[str, Any], path: str | Path) -> Path:
    """Export one escaped, accessible, network-free workflow walkthrough."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        _orientation(report),
        _timeline(report),
        _workflow(report),
        _choices(report),
        _audit(report),
        _risks(report),
        _concepts(report),
        _next_actions(report),
        _methods(report),
    ]
    document = render_report(
        "BuildML Session Workflow Walkthrough",
        sections,
        subtitle="Recorded operations, current availability, blockers, and stated reasons.",
        metadata={
            "Recorded operations": len(report.get("timeline") or []),
            "Done": (report.get("status_counts") or {}).get("done", 0),
            "Available": (report.get("status_counts") or {}).get("available", 0),
            "Blocked": (report.get("status_counts") or {}).get("blocked", 0),
            "Skipped": (report.get("status_counts") or {}).get("skipped", 0),
        },
    )
    destination.write_text(document, encoding="utf-8")
    return destination


def _timeline_row(record: dict[str, Any]) -> dict[str, Any]:
    transition = record.get("state_transition") or {}
    return {
        "sequence": record.get("sequence"),
        "timestamp": record.get("timestamp"),
        "operation": record.get("operation_id") or record.get("action"),
        "choice_origin": record.get("decision_origin", "explicit"),
        "parameters": record.get("parameters") or {},
        "state_changes": transition.get("changes") or [],
        "warnings": record.get("warnings") or [],
    }


def _unusual_order(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    state_keys = {
        "dataset": "has_dataset",
        "split": "has_split",
        "fit": "has_fit",
    }
    rows: list[dict[str, str]] = []
    for record in history:
        operation = str(record.get("operation_id") or record.get("action"))
        before = (record.get("state_transition") or {}).get("before") or {}
        spec = OPERATION_CATALOG.get(operation)
        missing: list[str] = []
        if spec is not None:
            for prerequisite in spec.prerequisites:
                if prerequisite.status.value != "required":
                    continue
                if prerequisite.key == "roles":
                    roles = set((before.get("roles") or {}).values())
                    if not {"feature", "target"} <= roles:
                        missing.append("feature and target roles")
                    continue
                state_key = state_keys.get(prerequisite.key)
                if state_key is not None and not before.get(state_key):
                    missing.append(state_key)
        parameters = record.get("parameters") or {}
        if operation == "split" and parameters.get("stratify"):
            roles = set((before.get("roles") or {}).values())
            if "target" not in roles:
                missing.append("target role required by stratify=True")
        if missing:
            rows.append(
                {
                    "operation": operation,
                    "sequence": str(record.get("sequence")),
                    "reason": f"Recorded before expected state: {', '.join(missing)}.",
                }
            )
    return rows


def _risks_and_concepts(
    workflow: tuple[WorkflowStep, ...],
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    completed = {str(record.get("operation_id") or record.get("action")) for record in history}
    relevant = completed | {step.operation for step in workflow if step.status.value == "available"}
    risks: list[dict[str, str]] = []
    concepts: dict[str, dict[str, str]] = {}
    seen_risks: set[tuple[str, str]] = set()
    for operation in sorted(relevant):
        try:
            spec = get_operation(operation)
        except KeyError:
            continue
        for risk in (*spec.leakage_risks, *spec.assumptions):
            marker = (operation, risk)
            if marker not in seen_risks:
                risks.append(
                    {
                        "operation": operation,
                        "risk": risk,
                        "evidence": f"workflow-{operation}",
                    }
                )
                seen_risks.add(marker)
        for key in spec.concept_links:
            concepts[key] = {
                "concept": key,
                "operations": ", ".join(
                    sorted(name for name in relevant if key in get_operation(name).concept_links)
                ),
                "link": f"#concept-{element_id(key)}",
            }
    return risks, list(concepts.values())


def _engine_status(session: Any) -> dict[str, Any]:
    """Factual engine / lazy-native disclosure for orientation surfaces."""
    dataset = getattr(session, "dataset", None)
    if dataset is None:
        return {
            "engine": None,
            "mode": None,
            "has_native": False,
            "has_lazy_native": False,
            "pandas_stale": False,
            "disclosures": [
                "No dataset is attached; engine and lazy-native status are unavailable."
            ],
        }
    has_lazy = bool(getattr(dataset, "has_lazy_native", False))
    has_native = bool(getattr(dataset, "has_native", False))
    engine = getattr(getattr(dataset, "engine", None), "value", None) or str(
        getattr(dataset, "engine", None)
    )
    mode = getattr(getattr(dataset, "mode", None), "value", None) or str(
        getattr(dataset, "mode", None)
    )
    disclosures = [
        f"Engine={engine}; mode={mode}; has_native={has_native}; has_lazy_native={has_lazy}."
    ]
    if has_lazy:
        disclosures.append(
            "A Polars LazyFrame is attached. Projection can stay lazy; row counts, "
            "mask filters, samples, and Dataset.to_pandas() collect. This is "
            "collect-on-promote, not out-of-core sklearn training."
        )
    elif has_native:
        disclosures.append(
            "A native Polars/DuckDB handle is attached for project/filter/sample "
            "before Pandas materialization. Estimators still require an in-memory "
            "design matrix at fit."
        )
    else:
        disclosures.append(
            "No engine-native handle is attached; the Session is Pandas-backed for "
            "tabular ops and sklearn materialization."
        )
    return {
        "engine": engine,
        "mode": mode,
        "has_native": has_native,
        "has_lazy_native": has_lazy,
        "pandas_stale": bool(getattr(dataset, "pandas_stale", False)),
        "disclosures": disclosures,
    }


def _orientation(report: dict[str, Any]) -> ReportSection:
    counts = report.get("status_counts") or {}
    engine = report.get("engine_status") or {}
    warm = report.get("warm_start_status") or {}
    scope = report.get("preprocess_scope_status") or {}
    torch_status = report.get("torch_training_status") or {}
    body = _frame(
        "Resolver statuses, engine/lazy-native status, nested-CV warm-start disclosure, "
        "fold-local vs Session-global preprocess scope, Torch training-curve disclosure "
        "when a trainer exists, and the complete versioned Session history.",
        ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        "The resolver separates operations already done from valid, blocked, and intentionally skipped paths.",
        "Availability proves API prerequisites only; it does not prove domain appropriateness.",
        "Read the timeline first, then inspect blocked and available operations.",
    )
    body += render_table(
        [{"status": key, "operations": value} for key, value in sorted(counts.items())],
        caption="Workflow status counts",
    )
    if engine:
        body += render_table(
            [
                {
                    "engine": engine.get("engine"),
                    "mode": engine.get("mode"),
                    "has_native": engine.get("has_native"),
                    "has_lazy_native": engine.get("has_lazy_native"),
                    "pandas_stale": engine.get("pandas_stale"),
                }
            ],
            caption="Engine and lazy-native status",
        )
        for note in engine.get("disclosures") or []:
            body += f"<p>{escape(note)}</p>"
    if warm.get("enabled"):
        body += render_table(
            [
                {
                    "warm_start_studies": True,
                    "search_method": warm.get("search_method"),
                    "n_outer_splits": warm.get("n_outer_splits"),
                    "shared": warm.get("shared"),
                    "held_out_partitions": ", ".join(warm.get("held_out_partitions") or []),
                    "history_sequences": ", ".join(
                        str(s) for s in (warm.get("history_sequences") or [])
                    ),
                }
            ],
            caption="Nested CV warm_start_studies",
        )
        for note in warm.get("disclosures") or []:
            body += f"<p>{escape(note)}</p>"
    if scope.get("enabled"):
        fold = scope.get("fold_local") or {}
        session_g = scope.get("session_global") or {}
        body += render_table(
            [
                {
                    "fold_text": fold.get("text"),
                    "fold_reduce": fold.get("reduce"),
                    "session_text_features": session_g.get("text_features"),
                    "session_reduce_dimensions": session_g.get("reduce_dimensions"),
                    "session_custom_transform": session_g.get("apply_custom_transform"),
                    "session_resample": session_g.get("resample"),
                }
            ],
            caption="Preprocess scope (fold-local vs Session-global)",
        )
        for note in scope.get("disclosures") or []:
            body += f"<p>{escape(note)}</p>"
    if torch_status.get("enabled"):
        early = torch_status.get("early_stop") or {}
        body += render_table(
            [
                {
                    "n_epochs_ran": torch_status.get("n_epochs_ran"),
                    "scheduler": torch_status.get("scheduler_name"),
                    "device": (torch_status.get("device") or {}).get("resolved"),
                    "early_stop_triggered": early.get("triggered"),
                    "early_stop_monitor": early.get("monitor"),
                    "early_stop_partition": early.get("partition"),
                    "resumed_from_epochs": torch_status.get("resumed_from_epochs"),
                }
            ],
            caption="Torch training (curves / early stop / device)",
        )
        for note in torch_status.get("disclosures") or []:
            body += f"<p>{escape(note)}</p>"
        for note in (torch_status.get("limitations") or [])[:3]:
            body += f"<p>{escape(note)}</p>"
    return ReportSection("orientation", "Workflow orientation", body)


def _timeline(report: dict[str, Any]) -> ReportSection:
    rows = report.get("timeline") or []
    body = _frame(
        "Recorded operations in execution order, including parameters, origin, warnings, and state transitions.",
        f"{len(rows)} operations were recorded.",
        "The timeline distinguishes explicit user choices from automatic defaults and shows what state actually changed.",
        "History records public calls, not semantic correctness or external data provenance.",
        "Follow each state change and investigate warnings or unexpected no-op transitions.",
    )
    body += render_table(rows, caption="Operation timeline")
    return ReportSection("timeline", "Timeline and state transitions", body)


def _workflow(report: dict[str, Any]) -> ReportSection:
    rows = report.get("workflow") or []
    rendered = []
    for row in rows:
        operation = str(row.get("operation"))
        rendered.append(
            {
                "operation": operation,
                "status": row.get("status"),
                "origin": row.get("origin"),
                "reason": "; ".join(row.get("reasons") or []),
                "blockers": row.get("blockers"),
                "repeatable": row.get("repeatable"),
                "evidence_id": f"workflow-{operation}",
            }
        )
    body = _frame(
        "Every catalog operation against current Session state.",
        f"{len(rendered)} operation statuses were resolved.",
        "Blocked and skipped states explain absent workflow paths rather than silently hiding them.",
        "Optional dependencies and task-specific constraints may add runtime limits.",
        "Use the exact blocker or reason before attempting an operation.",
    )
    body += render_table(rendered, caption="Done, available, blocked, and skipped operations")
    return ReportSection("workflow", "Resolved workflow statuses", body)


def _choices(report: dict[str, Any]) -> ReportSection:
    rows = report.get("timeline") or []
    origins = Counter(str(row.get("choice_origin", "explicit")) for row in rows)
    unusual = report.get("unusual_order") or []
    body = _frame(
        "Decision origins and operation ordering against expected state prerequisites.",
        f"Origins: {dict(origins)}; unusual-order flags: {len(unusual)}.",
        "Automatic defaults and explicit choices have different audit implications.",
        "An unusual order flag is structural evidence, not proof that the scientific workflow is invalid.",
        "Review every automatic choice and every unusual-order flag.",
    )
    body += render_table(
        [{"choice_origin": key, "count": value} for key, value in sorted(origins.items())],
        caption="Explicit, recommended, and automatic choices",
    )
    body += render_table(unusual, caption="Unusual operation order")
    return ReportSection("choices", "Choices and unusual order", body)


def _audit(report: dict[str, Any]) -> ReportSection:
    audit = report.get("audit_summary") or {}
    ranked = list(audit.get("ranked_risks") or [])
    suggestions = list(audit.get("suggested_next_ops") or [])
    graph = dict(audit.get("prerequisite_graph") or {})
    body = _frame(
        "History counts, ranked unresolved risks, prerequisite gaps, and suggested next operations.",
        (
            f"{audit.get('n_operations', 0)} recorded operation(s); "
            f"{audit.get('warning_count', 0)} warning(s); "
            f"{len(ranked)} ranked risk(s); "
            f"{len(suggestions)} suggested next op(s)."
        ),
        "The audit summary ranks review cues and links them to concrete Session follow-ups.",
        "Ranked risks are heuristic; they are not proof of leakage, fairness failure, or invalid scores.",
        "Explain a suggested operation before executing it.",
    )
    body += render_table(
        [
            {
                "rank": item.get("rank"),
                "severity": item.get("severity"),
                "message": item.get("message"),
                "suggested_operation": item.get("suggested_operation"),
                "source": item.get("source"),
            }
            for item in ranked
        ],
        caption="Ranked unresolved risks",
    )
    body += render_table(
        [
            {
                "operation": item.get("operation"),
                "status": item.get("status"),
                "reason": item.get("reason"),
                "api_action": item.get("api_action"),
            }
            for item in suggestions
        ],
        caption="Suggested next operations",
    )
    missing = list(graph.get("missing_required") or [])
    if missing:
        body += render_table(
            [{"missing_required": item} for item in missing],
            caption="Missing required prerequisites for suggested operations",
        )
    return ReportSection("audit", "Audit summary", body)


def _risks(report: dict[str, Any]) -> ReportSection:
    rows = report.get("unresolved_risks") or []
    body = _frame(
        "Catalog assumptions and leakage risks for completed or currently available operations.",
        f"{len(rows)} unresolved review prompts remain.",
        "These risks identify conditions the API cannot verify from runtime state.",
        "The list is conservative and may include risks already controlled outside BuildML.",
        "Resolve or document each material risk before relying on model results.",
    )
    body += render_table(rows, caption="Unresolved risks linked to workflow evidence")
    return ReportSection("risks", "Unresolved risks", body)


def _concepts(report: dict[str, Any]) -> ReportSection:
    rows = report.get("concept_links") or []
    cards = "".join(
        f'<article class="bml-card" id="concept-{escape(element_id(row.get("concept")))}">'
        f"<h3>{escape(row.get('concept'))}</h3><p>Relevant operations: {escape(row.get('operations'))}</p></article>"
        for row in rows
    )
    body = (
        _frame(
            "Shared technical concepts referenced by relevant operation specifications.",
            f"{len(rows)} concept links are available.",
            "Concepts connect individual calls to reusable modeling principles.",
            "Short concept links are orientation, not a substitute for project-specific methodology.",
            "Use Session.explain(operation) for the complete linked concept notes.",
        )
        + cards
    )
    return ReportSection("concepts", "Concept links", body)


def _next_actions(report: dict[str, Any]) -> ReportSection:
    rows = report.get("next_actions") or []
    items = "".join(
        "<li>"
        f"<strong>{escape(row.get('operation'))}</strong> {render_badge(row.get('status'), tone='good')}"
        f"<p>{escape(row.get('reason'))}</p>"
        f'<p>Evidence: <a href="#workflow">{escape(row.get("evidence"))}</a></p>'
        f"<p>API action: <code>{escape(row.get('api_action'))}</code></p>"
        "</li>"
        for row in rows
    )
    body = (
        _frame(
            "Currently available operations and the API call that explains each before execution.",
            f"{len(rows)} next actions satisfy current API prerequisites.",
            "Every suggestion links to resolver evidence and an explicit public action.",
            "Available does not mean recommended; read assumptions, alternatives, and leakage risks.",
            "Call the listed Session.explain action before choosing.",
        )
        + f"<ol>{items}</ol>"
    )
    return ReportSection("next-actions", "Available next actions", body)


def _methods(report: dict[str, Any]) -> ReportSection:
    body = _frame(
        "Workflow resolver output, normalized history records, operation catalog risks, and concept links.",
        "No state was mutated while building this walkthrough.",
        "Joining live state to observed history distinguishes possible actions from completed actions.",
        "External notebooks, manual data changes, and unrecorded estimator work are outside the Session history.",
        "Retain the walkthrough with checkpoint and model artifacts when auditability matters.",
    )
    body += f'<details><summary>Serializable walkthrough payload</summary><pre class="bml-json">{escape(json.dumps(report, indent=2, default=str))}</pre></details>'
    return ReportSection("methods", "Methods and limitations", body)


def _frame(examined: str, observed: str, why: str, limits: str, next_step: str) -> str:
    return render_reading_frame(
        examined=examined,
        observed=observed,
        why=why,
        limits=limits,
        next_step=next_step,
    )
