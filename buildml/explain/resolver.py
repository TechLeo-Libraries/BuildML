"""Read-only workflow and operation explanation resolver."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Literal

from buildml.core.types import ColumnRole
from buildml.explain.catalog import OPERATION_CATALOG, get_operation
from buildml.explain.concepts import get_concept
from buildml.explain.schemas import (
    AfterOperationExplanation,
    BeforeOperationExplanation,
    DecisionOrigin,
    WorkflowStep,
    WorkflowStepStatus,
)

_PROVIDERS: dict[str, tuple[str, ...]] = {
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
    ),
    "fit_torch": ("fit_torch", "load_torch_bundle"),
    "torch-extra": (),
    "viz-extra": (),
}

_REPEATABLE = {
    "apply_custom_transform",
    "apply_preprocess_plans",
    "assert_can_fit",
    "bin",
    "calibration",
    "checkpoint_save",
    "cv_score",
    "drop_columns",
    "dry_run",
    "eda",
    "encode",
    "error_slices",
    "eval_plots",
    "evaluate",
    "evaluate_torch",
    "explain",
    "extract_dates",
    "feature_importance",
    "fit_torch",
    "grid_search",
    "handle_outliers",
    "head",
    "impute",
    "learning_curve",
    "list_transforms",
    "load_torch_bundle",
    "make_torch_loaders",
    "metadata",
    "nested_cv_score",
    "partition",
    "predict",
    "predict_from_pipeline",
    "prepare_design_matrix",
    "randomized_search",
    "reduce_dimensions",
    "register_transform",
    "resample",
    "resample_strategies",
    "save_model",
    "save_pipeline",
    "save_torch_bundle",
    "scale",
    "select_features",
    "summarize_history",
    "text_features",
    "to_engine",
    "to_pandas",
    "to_parquet",
    "tune_threshold",
    "workflow",
    "with_engine",
    "with_mode",
    "walkthrough",
}


def _history(session: Any) -> list[dict[str, Any]]:
    return list(getattr(session, "_history", ()))


def _operation_ids(session: Any) -> list[str]:
    return [
        str(record.get("operation_id") or record.get("action"))
        for record in _history(session)
    ]


def _prerequisite_state(session: Any, key: str) -> tuple[bool, str]:
    dataset = getattr(session, "_dataset", None)
    if key == "dataset":
        return (
            dataset is not None,
            "A materialized dataset is attached."
            if dataset is not None
            else "No materialized dataset is attached.",
        )
    if key == "roles":
        roles = {} if dataset is None else dataset.roles
        has_feature = ColumnRole.FEATURE in roles.values()
        has_target = ColumnRole.TARGET in roles.values()
        missing = [
            label
            for label, present in (("feature role", has_feature), ("target role", has_target))
            if not present
        ]
        return (
            has_feature and has_target,
            "Feature and target roles are assigned."
            if not missing
            else f"Missing {', '.join(missing)}.",
        )
    if key == "split":
        present = getattr(session, "_split_plan", None) is not None
        return (
            present,
            "A train/evaluation split exists."
            if present
            else "No train/evaluation split exists.",
        )
    if key == "fit":
        present = getattr(session, "_fit_result", None) is not None
        return (
            present,
            "An active fitted estimator exists."
            if present
            else "No active fitted estimator exists.",
        )
    if key == "fit_torch":
        present = getattr(session, "_dl_train_result", None) is not None
        return (
            present,
            "An active Torch trainer exists."
            if present
            else "No active Torch trainer exists.",
        )
    if key == "torch-extra":
        present = find_spec("torch") is not None
        return (
            present,
            "Torch dependencies are installed."
            if present
            else "Torch is not installed; install buildml[torch] for DL methods.",
        )
    if key == "viz-extra":
        present = find_spec("matplotlib") is not None
        return (
            present,
            "Visualization dependencies are installed."
            if present
            else "Visualization dependencies are not installed; non-plot paths remain usable.",
        )
    return False, f"Unknown prerequisite '{key}'."


def prerequisite_status(session: Any, operation: str) -> dict[str, bool]:
    """Return the live pass/fail state for an operation's prerequisites."""
    spec = get_operation(operation)
    return {item.key: _prerequisite_state(session, item.key)[0] for item in spec.prerequisites}


def _chains(operation: str, keys: list[str]) -> tuple[str, ...]:
    chains: list[str] = []

    def visit(consumer: str, prerequisite_keys: list[str], visited: set[str]) -> None:
        if consumer in visited:
            return
        visited.add(consumer)
        for key in prerequisite_keys:
            providers = _PROVIDERS.get(key, ())
            if providers:
                statement = f"{consumer} requires {key} via {' or '.join(providers)}"
                if statement not in chains:
                    chains.append(statement)
                for provider in providers:
                    provider_spec = OPERATION_CATALOG.get(provider)
                    if provider_spec is not None:
                        visit(
                            provider,
                            [item.key for item in provider_spec.prerequisites],
                            visited,
                        )
            else:
                statement = f"{consumer} requires external capability {key}"
                if statement not in chains:
                    chains.append(statement)

    visit(operation, keys, set())
    return tuple(chains)


def resolve_workflow(session: Any) -> tuple[WorkflowStep, ...]:
    """Resolve every catalog operation against current Session state."""
    completed = _operation_ids(session)
    completed_set = set(completed)
    steps: list[WorkflowStep] = []
    for operation, spec in OPERATION_CATALOG.items():
        failed: list[str] = []
        optional_notes: list[str] = []
        prerequisite_keys: list[str] = []
        for prerequisite in spec.prerequisites:
            passed, reason = _prerequisite_state(session, prerequisite.key)
            prerequisite_keys.append(prerequisite.key)
            if not passed and prerequisite.status.value == "required":
                providers = _PROVIDERS.get(prerequisite.key, ())
                remedy = f" Run {' or '.join(providers)} first." if providers else ""
                failed.append(f"{reason}{remedy}")
            elif not passed:
                optional_notes.append(reason)

        reasons: list[str] = []
        alternate: str | None = None
        split_providers = _PROVIDERS.get("split", ())
        if operation in split_providers and operation not in completed_set:
            for provider in split_providers:
                if provider != operation and provider in completed_set:
                    alternate = provider
                    break

        if operation in completed_set:
            status = WorkflowStepStatus.DONE
            operation_records = (
                record
                for record in _history(session)
                if (record.get("operation_id") or record.get("action")) == operation
            )
            latest_sequence = max(record["sequence"] for record in operation_records)
            reasons.append(f"Completed at sequence {latest_sequence}.")
            if operation in _REPEATABLE:
                reasons.append("The operation is repeatable when its prerequisites still hold.")
        elif alternate:
            status = WorkflowStepStatus.SKIPPED
            reasons.append(
                f"Skipped because alternative operation '{alternate}' established this state."
            )
        elif failed:
            status = WorkflowStepStatus.BLOCKED
            reasons.extend(failed)
        else:
            status = WorkflowStepStatus.AVAILABLE
            reasons.append("All required prerequisites are satisfied.")
        reasons.extend(optional_notes)

        origin = DecisionOrigin.EXPLICIT
        records = [
            record
            for record in _history(session)
            if (record.get("operation_id") or record.get("action")) == operation
        ]
        if records:
            try:
                origin = DecisionOrigin(records[-1].get("decision_origin", "explicit"))
            except ValueError:
                origin = DecisionOrigin.EXPLICIT
        steps.append(
            WorkflowStep(
                operation=operation,
                status=status,
                origin=origin,
                summary=f"{spec.purpose} {' '.join(reasons)}",
                blockers=tuple(failed),
                prerequisite_chain=_chains(operation, prerequisite_keys),
                reasons=tuple(reasons),
                repeatable=operation in _REPEATABLE,
            )
        )
    return tuple(steps)


def explain_before(session: Any, operation: str) -> BeforeOperationExplanation:
    """Explain whether and why an operation fits the current workflow moment."""
    spec = get_operation(operation)
    step = next(item for item in resolve_workflow(session) if item.operation == operation)
    appropriateness = list(step.reasons)
    appropriateness.extend(spec.usual_ordering)
    return BeforeOperationExplanation(
        operation=operation,
        purpose=spec.purpose,
        pipeline_role=spec.pipeline_role,
        status=step.status,
        prerequisite_status=prerequisite_status(session, operation),
        prerequisite_chain=step.prerequisite_chain,
        appropriateness=tuple(appropriateness),
        alternatives=spec.alternatives,
        risks=spec.leakage_risks + spec.failure_modes + spec.anti_patterns,
        likely_state_changes=spec.state_changes,
        concept_notes=tuple(get_concept(key) for key in spec.concept_links),
    )


def explain_after(session: Any, operation: str) -> AfterOperationExplanation:
    """Explain the latest observed run, or explicitly describe its absence."""
    spec = get_operation(operation)
    records = [
        record
        for record in _history(session)
        if (record.get("operation_id") or record.get("action")) == operation
    ]
    record = records[-1] if records else None
    choices = tuple(
        step.operation
        for step in resolve_workflow(session)
        if step.status == WorkflowStepStatus.AVAILABLE
    )
    if record is None:
        return AfterOperationExplanation(
            operation=operation,
            sequence=None,
            parameters={},
            result_summary={},
            decision_origin=DecisionOrigin.EXPLICIT,
            why_applied=(
                "No observed run exists; use the before explanation to assess this operation.",
            ),
            state_changes=(),
            interpretation=("There is no result to interpret.",),
            limitations=("After-operation evidence is unavailable until the operation runs.",),
            next_valid_choices=choices,
            concept_notes=tuple(get_concept(key) for key in spec.concept_links),
        )
    try:
        origin = DecisionOrigin(record.get("decision_origin", "explicit"))
    except ValueError:
        origin = DecisionOrigin.EXPLICIT
    transition = record.get("state_transition", {})
    changes = transition.get("changes", ()) if isinstance(transition, dict) else ()
    warnings = tuple(str(item) for item in record.get("warnings", ()))
    return AfterOperationExplanation(
        operation=operation,
        sequence=int(record["sequence"]),
        parameters=record.get("parameters", {}),
        result_summary=record.get("result_summary", {}),
        decision_origin=origin,
        why_applied=spec.selection_rationale
        + (f"Recorded decision origin: {origin.value}.",),
        state_changes=tuple(str(item) for item in changes),
        interpretation=spec.result_reading,
        limitations=spec.assumptions + spec.failure_modes + warnings,
        next_valid_choices=choices,
        concept_notes=tuple(get_concept(key) for key in spec.concept_links),
    )


def explain(
    session: Any,
    operation: str | None = None,
    *,
    moment: Literal["before", "after"] = "before",
) -> BeforeOperationExplanation | AfterOperationExplanation | tuple[WorkflowStep, ...]:
    """Standalone explanation API used by the thin Session facade."""
    if operation is None:
        return resolve_workflow(session)
    if moment == "before":
        return explain_before(session, operation)
    if moment == "after":
        return explain_after(session, operation)
    raise ValueError("moment must be 'before' or 'after'")
