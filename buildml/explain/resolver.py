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
        "optuna_search",
        "evolutionary_search",
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "load_ensemble_bundle",
        "run_automl",
        "load_automl_bundle",
    ),
    "ensemble-plan": (
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "load_ensemble_bundle",
    ),
    "automl-plan": (
        "run_automl",
        "load_automl_bundle",
    ),
    "forecast-plan": (
        "fit_forecast",
        "load_forecast_bundle",
    ),
    "anomaly-plan": (
        "fit_anomaly",
        "load_anomaly_bundle",
    ),
    "semisupervised-plan": (
        "fit_semisupervised",
        "load_semisupervised_bundle",
    ),
    "ssl-plan": (
        "fit_ssl_pretext",
        "load_ssl_bundle",
    ),
    "ssl-head": (
        "finetune_ssl_head",
    ),
    "activelearning-plan": (
        "fit_active_learner",
        "load_active_learning_bundle",
    ),
    "online-plan": (
        "fit_online",
        "load_online_bundle",
    ),
    "multitask-plan": (
        "fit_multitask",
        "load_multitask_bundle",
    ),
    "metalearning-plan": (
        "fit_metalearning",
        "load_metalearning_bundle",
    ),
    "federated-plan": (
        "fit_federated",
        "load_federated_bundle",
    ),
    "probabilistic-plan": (
        "fit_probabilistic",
        "load_probabilistic_bundle",
    ),
    "causal-assumptions": (
        "declare_causal_assumptions",
        "fit_causal",
        "load_causal_bundle",
    ),
    "causal-plan": (
        "fit_causal",
        "load_causal_bundle",
    ),
    "graph-spec": (
        "set_graph",
        "load_graph_bundle",
    ),
    "graph-plan": (
        "fit_graph",
        "load_graph_bundle",
    ),
    "symbolic-plan": (
        "fit_symbolic",
        "load_symbolic_bundle",
    ),
    "neuro-symbolic-plan": (
        "fit_neuro_symbolic",
        "load_symbolic_bundle",
    ),
    "cbr-plan": (
        "fit_cbr",
        "load_cbr_bundle",
    ),
    "imitation-plan": (
        "fit_imitation",
        "load_imitation_bundle",
    ),
    "rl-plan": (
        "fit_rl",
        "load_rl_bundle",
    ),
    "fit_torch": ("fit_torch", "load_torch_bundle", "fit_torch_ddp"),
    "torch-extra": (),
    "rag-corpus": ("rag_ingest_corpus",),
    "rag-index": ("rag_embed_and_index", "load_rag_bundle", "rag_upsert"),
    "rag-extra": (),
    "cluster-plan": ("fit_clusters", "load_unsupervised_bundle"),
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
    "evaluate_ensemble",
    "evaluate_automl",
    "evaluate_forecast",
    "evaluate_anomaly",
    "evaluate_semisupervised",
    "evaluate_ssl",
    "evaluate_active_learning",
    "evaluate_online",
    "evaluate_multitask",
    "evaluate_metalearning",
    "evaluate_federated",
    "evaluate_probabilistic",
    "evaluate_causal",
    "estimate_causal",
    "refute_causal",
    "evaluate_graph",
    "evaluate_symbolic",
    "evaluate_neuro_symbolic",
    "evaluate_cbr",
    "evaluate_imitation",
    "evaluate_rl",
    "evaluate_torch",
    "explain",
    "fit_voting",
    "fit_stacking",
    "fit_blending",
    "fit_forecast",
    "fit_anomaly",
    "fit_semisupervised",
    "predict_semisupervised",
    "fit_ssl_pretext",
    "transform_ssl",
    "finetune_ssl_head",
    "fit_active_learner",
    "suggest_query",
    "label_rows",
    "fit_online",
    "partial_fit_online",
    "predict_online",
    "fit_multitask",
    "predict_multitask",
    "fit_metalearning",
    "adapt_to_task",
    "fit_federated",
    "predict_federated",
    "fit_probabilistic",
    "predict_probabilistic",
    "predict_interval",
    "declare_causal_assumptions",
    "fit_causal",
    "set_graph",
    "fit_graph",
    "predict_graph",
    "fit_symbolic",
    "predict_symbolic",
    "fit_neuro_symbolic",
    "predict_neuro_symbolic",
    "fit_cbr",
    "retrieve_cases",
    "predict_cbr",
    "retain_cbr",
    "fit_imitation",
    "predict_imitation_action",
    "fit_rl",
    "act_rl",
    "generate_forecast",
    "score_anomalies",
    "load_ensemble_bundle",
    "save_ensemble_bundle",
    "run_automl",
    "load_automl_bundle",
    "save_automl_bundle",
    "load_forecast_bundle",
    "save_forecast_bundle",
    "load_anomaly_bundle",
    "save_anomaly_bundle",
    "load_semisupervised_bundle",
    "save_semisupervised_bundle",
    "load_ssl_bundle",
    "save_ssl_bundle",
    "load_active_learning_bundle",
    "save_active_learning_bundle",
    "load_online_bundle",
    "save_online_bundle",
    "load_multitask_bundle",
    "save_multitask_bundle",
    "load_metalearning_bundle",
    "save_metalearning_bundle",
    "load_federated_bundle",
    "save_federated_bundle",
    "load_probabilistic_bundle",
    "save_probabilistic_bundle",
    "load_causal_bundle",
    "save_causal_bundle",
    "load_graph_bundle",
    "save_graph_bundle",
    "load_symbolic_bundle",
    "save_symbolic_bundle",
    "load_cbr_bundle",
    "save_cbr_bundle",
    "load_imitation_bundle",
    "save_imitation_bundle",
    "load_rl_bundle",
    "save_rl_bundle",
    "extract_dates",
    "feature_importance",
    "fit_torch",
    "cross_validate_torch",
    "search_torch",
    "nested_cv_torch",
    "make_multimodal_torch_loaders",
    "make_image_multimodal_torch_loaders",
    "make_audio_multimodal_torch_loaders",
    "export_torch",
    "fit_torch_ddp",
    "torch_training_curve",
    "ai_run_autonomous",
    "assign_clusters",
    "evaluate_clusters",
    "fit_clusters",
    "evolutionary_search",
    "grid_search",
    "handle_outliers",
    "head",
    "impute",
    "learning_curve",
    "list_transforms",
    "load_torch_bundle",
    "load_unsupervised_bundle",
    "make_torch_loaders",
    "make_text_torch_loaders",
    "metadata",
    "nested_cv_score",
    "partition",
    "predict",
    "predict_from_pipeline",
    "prepare_design_matrix",
    "rag_chunk",
    "rag_delete",
    "rag_embed_and_index",
    "rag_evaluate",
    "rag_generate",
    "rag_ingest_corpus",
    "rag_retrieve",
    "rag_upsert",
    "optuna_search",
    "randomized_search",
    "reduce_dimensions",
    "register_transform",
    "resample",
    "resample_strategies",
    "save_model",
    "save_pipeline",
    "save_rag_bundle",
    "save_torch_bundle",
    "save_unsupervised_bundle",
    "scale",
    "load_rag_bundle",
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
    if key == "rag-corpus":
        present = getattr(session, "_rag_corpus", None) is not None
        return (
            present,
            "A RAG corpus is attached."
            if present
            else "No RAG corpus is attached.",
        )
    if key == "rag-index":
        present = getattr(session, "_rag_index_result", None) is not None
        return (
            present,
            "An active RAG index exists."
            if present
            else "No active RAG index exists.",
        )
    if key == "rag-extra":
        present = find_spec("sentence_transformers") is not None
        return (
            present,
            "RAG dependencies are installed."
            if present
            else "RAG is not installed; install buildml[rag] for retrieval methods.",
        )
    if key == "cluster-plan":
        present = getattr(session, "_cluster_plan", None) is not None
        return (
            present,
            "A train-fitted ClusterPlan is attached."
            if present
            else "No ClusterPlan is attached; call fit_clusters or load_unsupervised_bundle.",
        )
    if key == "ensemble-plan":
        present = getattr(session, "_ensemble_plan", None) is not None
        return (
            present,
            "A train-fitted EnsemblePlan is attached."
            if present
            else (
                "No EnsemblePlan is attached; call fit_voting / fit_stacking / "
                "fit_blending or load_ensemble_bundle."
            ),
        )
    if key == "automl-plan":
        present = getattr(session, "_automl_plan", None) is not None
        return (
            present,
            "A train-selected AutoMLPlan is attached."
            if present
            else "No AutoMLPlan is attached; call run_automl or load_automl_bundle.",
        )
    if key == "forecast-plan":
        present = getattr(session, "_forecast_plan", None) is not None
        return (
            present,
            "A train-fitted ForecastPlan is attached."
            if present
            else "No ForecastPlan is attached; call fit_forecast or load_forecast_bundle.",
        )
    if key == "anomaly-plan":
        present = getattr(session, "_anomaly_plan", None) is not None
        return (
            present,
            "A train-fitted AnomalyPlan is attached."
            if present
            else "No AnomalyPlan is attached; call fit_anomaly or load_anomaly_bundle.",
        )
    if key == "semisupervised-plan":
        present = getattr(session, "_semisupervised_plan", None) is not None
        return (
            present,
            "A train-fitted SemiSupervisedPlan is attached."
            if present
            else "No SemiSupervisedPlan is attached; call fit_semisupervised or "
            "load_semisupervised_bundle.",
        )
    if key == "ssl-plan":
        present = getattr(session, "_ssl_plan", None) is not None
        return (
            present,
            "A train-fitted SelfSupervisedPlan is attached."
            if present
            else "No SelfSupervisedPlan is attached; call fit_ssl_pretext or load_ssl_bundle.",
        )
    if key == "ssl-head":
        present = getattr(session, "_ssl_head_plan", None) is not None
        return (
            present,
            "An SSLHeadPlan is attached."
            if present
            else "No SSLHeadPlan is attached; call finetune_ssl_head.",
        )
    if key == "activelearning-plan":
        present = getattr(session, "_activelearning_plan", None) is not None
        return (
            present,
            "A train-fitted ActiveLearningPlan is attached."
            if present
            else "No ActiveLearningPlan is attached; call fit_active_learner or "
            "load_active_learning_bundle.",
        )
    if key == "online-plan":
        present = getattr(session, "_online_plan", None) is not None
        return (
            present,
            "A warm-started OnlinePlan is attached."
            if present
            else "No OnlinePlan is attached; call fit_online or load_online_bundle.",
        )
    if key == "multitask-plan":
        present = getattr(session, "_multitask_plan", None) is not None
        return (
            present,
            "A train-fitted MultiTaskPlan is attached."
            if present
            else "No MultiTaskPlan is attached; call fit_multitask or "
            "load_multitask_bundle.",
        )
    if key == "metalearning-plan":
        present = getattr(session, "_metalearning_plan", None) is not None
        return (
            present,
            "A train-fitted MetaLearningPlan is attached."
            if present
            else "No MetaLearningPlan is attached; call fit_metalearning or "
            "load_metalearning_bundle.",
        )
    if key == "federated-plan":
        present = getattr(session, "_federated_plan", None) is not None
        return (
            present,
            "A train-fitted FederatedPlan is attached."
            if present
            else "No FederatedPlan is attached; call fit_federated or "
            "load_federated_bundle.",
        )
    if key == "probabilistic-plan":
        present = getattr(session, "_probabilistic_plan", None) is not None
        return (
            present,
            "A train-fitted ProbabilisticPlan is attached."
            if present
            else "No ProbabilisticPlan is attached; call fit_probabilistic or "
            "load_probabilistic_bundle.",
        )
    if key == "causal-assumptions":
        present = (
            getattr(session, "_causal_assumptions", None) is not None
            or getattr(session, "_causal_plan", None) is not None
        )
        return (
            present,
            "Validated CausalAssumptions are attached."
            if present
            else "No CausalAssumptions declared; call declare_causal_assumptions "
            "or pass assumptions= into fit_causal (EDA is not a substitute).",
        )
    if key == "causal-plan":
        present = getattr(session, "_causal_plan", None) is not None
        return (
            present,
            "A train-fitted CausalPlan is attached."
            if present
            else "No CausalPlan is attached; call fit_causal or "
            "load_causal_bundle.",
        )
    if key == "graph-spec":
        present = (
            getattr(session, "_graph_spec", None) is not None
            or getattr(session, "_graph_plan", None) is not None
        )
        return (
            present,
            "A GraphSpec edge list is attached."
            if present
            else "No GraphSpec attached; call set_graph(edges, node_id_col=...) "
            "before fit_graph.",
        )
    if key == "graph-plan":
        present = getattr(session, "_graph_plan", None) is not None
        return (
            present,
            "A train-fitted GraphPlan is attached."
            if present
            else "No GraphPlan is attached; call fit_graph or load_graph_bundle.",
        )
    if key == "symbolic-plan":
        present = getattr(session, "_symbolic_plan", None) is not None
        return (
            present,
            "A train-fitted SymbolicPlan is attached."
            if present
            else "No SymbolicPlan is attached; call fit_symbolic or "
            "load_symbolic_bundle.",
        )
    if key == "neuro-symbolic-plan":
        present = getattr(session, "_neuro_symbolic_plan", None) is not None
        return (
            present,
            "A train-fitted NeuroSymbolicPlan is attached."
            if present
            else "No NeuroSymbolicPlan is attached; call fit_neuro_symbolic or "
            "load_symbolic_bundle.",
        )
    if key == "cbr-plan":
        present = getattr(session, "_cbr_plan", None) is not None
        return (
            present,
            "A train-fitted CbrPlan (case memory) is attached."
            if present
            else "No CbrPlan is attached; call fit_cbr or load_cbr_bundle.",
        )
    if key == "imitation-plan":
        present = getattr(session, "_imitation_plan", None) is not None
        return (
            present,
            "A train-fitted ImitationPlan (behavioral cloning) is attached."
            if present
            else "No ImitationPlan is attached; call fit_imitation or "
            "load_imitation_bundle.",
        )
    if key == "rl-plan":
        present = getattr(session, "_rl_plan", None) is not None
        return (
            present,
            "A fitted RlPlan (bandit or gym policy) is attached."
            if present
            else "No RlPlan is attached; call fit_rl or load_rl_bundle.",
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
