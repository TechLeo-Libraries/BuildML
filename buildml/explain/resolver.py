"""Read-only workflow and operation explanation resolver."""

from __future__ import annotations

from typing import Any, Literal

from buildml.explain.catalog import OPERATION_CATALOG, get_operation
from buildml.explain.capability_status import CAPABILITY_MATRIX_OPERATIONS
from buildml.explain.concepts import get_concept
from buildml.explain.pedagogy import primer_for
from buildml.explain.prerequisites import PROVIDERS, probe
from buildml.explain.schemas import (
    AfterOperationExplanation,
    BeforeOperationExplanation,
    DecisionOrigin,
    LearningLevel,
    WorkflowStep,
    WorkflowStepStatus,
)

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


def prerequisite_status(session: Any, operation: str) -> dict[str, bool]:
    """Check each precondition of one operation against the live session.

    "Blocked" is not a useful answer on its own. This reports the preconditions
    individually so a caller can say which one is missing and, through
    :func:`~buildml.explain.prerequisites.providers_for`, which call would
    supply it.

    Parameters
    ----------
    session:
        The session to inspect. Nothing on it is modified.
    operation:
        A catalog operation name, such as ``'fit'``, or a facade form
        (``'classical.fit'`` / ``'session.classical.fit'``).

    Returns
    -------
    dict of str to bool
        Prerequisite key mapped to whether it currently holds. Optional and
        recommended prerequisites are included; read
        :attr:`~buildml.explain.schemas.Prerequisite.status` on the catalog
        entry to tell them apart from required ones.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No catalog operation has that name.
    """
    from buildml.session.facade_registry import resolve_operation_name

    operation = resolve_operation_name(operation)
    spec = get_operation(operation)
    return {item.key: probe(session, item.key)[0] for item in spec.prerequisites}


def _chains(operation: str, keys: list[str]) -> tuple[str, ...]:
    chains: list[str] = []

    def visit(consumer: str, prerequisite_keys: list[str], visited: set[str]) -> None:
        if consumer in visited:
            return
        visited.add(consumer)
        for key in prerequisite_keys:
            providers = PROVIDERS.get(key, ())
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
    """Score every operation in the catalog against the current session.

    This is what turns a list of 288 methods into a next step. Each operation is
    reported as available, done, or blocked, with the reason attached, so a
    caller can answer "what can I do now" without knowing the dependency graph.

    Availability is a statement about preconditions, not about judgement: an
    operation can be available and still be the wrong thing to run.

    Parameters
    ----------
    session:
        The session to resolve against. Nothing on it is modified.

    Returns
    -------
    tuple of ~buildml.explain.schemas.WorkflowStep
        One step per catalog operation, in catalog order, each carrying its
        status, blockers, prerequisite chain, and the reasons behind them.
    """
    completed = _operation_ids(session)
    completed_set = set(completed)
    steps: list[WorkflowStep] = []
    for operation, spec in OPERATION_CATALOG.items():
        if operation in CAPABILITY_MATRIX_OPERATIONS:
            steps.append(
                WorkflowStep(
                    operation=operation,
                    status=WorkflowStepStatus.AVAILABLE,
                    origin=DecisionOrigin.EXPLICIT,
                    summary=(
                        f"{spec.purpose} Read-only capability introspection; "
                        "always available regardless of history."
                    ),
                    blockers=(),
                    prerequisite_chain=(),
                    reasons=(
                        "Read-only backend/method availability matrix for this domain.",
                        "Safe to call before choosing a fit backend or method.",
                    ),
                    repeatable=True,
                )
            )
            continue
        failed: list[str] = []
        optional_notes: list[str] = []
        prerequisite_keys: list[str] = []
        for prerequisite in spec.prerequisites:
            passed, reason = probe(session, prerequisite.key)
            prerequisite_keys.append(prerequisite.key)
            if not passed and prerequisite.status.value == "required":
                providers = PROVIDERS.get(prerequisite.key, ())
                remedy = f" Run {' or '.join(providers)} first." if providers else ""
                failed.append(f"{reason}{remedy}")
            elif not passed:
                optional_notes.append(reason)

        reasons: list[str] = []
        alternate: str | None = None
        split_providers = PROVIDERS.get("split", ())
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


def explain_before(
    session: Any,
    operation: str,
    *,
    level: LearningLevel | str | None = LearningLevel.BEGINNER,
) -> BeforeOperationExplanation:
    """Brief a caller before they run something, so the choice is informed.

    The expensive mistakes in a machine-learning workflow are made before a call
    runs, not after: preprocessing fitted on the wrong rows, a split chosen after
    the fact, a metric picked because it looked good. This assembles what is
    knowable in advance: whether the preconditions hold, what the operation is
    for, what it will change, what could go wrong, and what else could be used
    instead: and fronts all of it with a beginner primer.

    Parameters
    ----------
    session:
        The session the operation would run against. Nothing is modified.
    operation:
        A catalog operation name, such as ``'split'``, or a facade form
        (``'data.split'`` / ``'session.data.split'``).
    level:
        ``'beginner'`` (the default), ``'intermediate'``, or ``'advanced'``.
        Controls how much scaffolding the primer renders, never which facts the
        explanation states.

    Returns
    -------
    ~buildml.explain.schemas.BeforeOperationExplanation
        Purpose, pipeline role, resolved status, prerequisite state and chain,
        appropriateness notes, alternatives, risks, likely state changes, the
        linked concept notes, and the beginner primer. ``operation`` is always
        the canonical flat catalog key.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No catalog operation has that name.
    ValueError
        ``level`` is not one of the three reading levels.

    See Also
    --------
    explain_after : The same operation, once it has run.
    buildml.explain.learn : The concept behind the call, rather than the call.
    """
    from buildml.session.facade_registry import resolve_operation_name

    operation = resolve_operation_name(operation)
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
        beginner=primer_for(operation, level=LearningLevel.coerce(level)),
    )


def explain_after(
    session: Any,
    operation: str,
    *,
    level: LearningLevel | str | None = LearningLevel.BEGINNER,
) -> AfterOperationExplanation:
    """Interpret the most recent run of an operation against its contract.

    A result object tells you what came back; it does not tell you what the
    numbers mean, what the run assumed, or what is now safe to do next. This
    reads the recorded history entry and interprets it against the catalog
    entry, so the interpretation cannot quietly disagree with the operation's
    documented behaviour.

    When the operation has not run, that is stated rather than guessed at: the
    explanation returns with an empty result summary and says so.

    Parameters
    ----------
    session:
        The session whose history is read. Nothing is modified.
    operation:
        A catalog operation name, such as ``'evaluate'``, or a facade form
        (``'classical.evaluate'`` / ``'session.classical.evaluate'``).
    level:
        ``'beginner'`` (the default), ``'intermediate'``, or ``'advanced'``.

    Returns
    -------
    ~buildml.explain.schemas.AfterOperationExplanation
        The recorded parameters and result summary, why the operation was
        applied, what changed, how to read the outcome, the limitations that
        still apply, the operations now available, and the beginner primer.
        ``operation`` is always the canonical flat catalog key.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No catalog operation has that name.
    ValueError
        ``level`` is not one of the three reading levels.

    See Also
    --------
    explain_before : The same operation, before it runs.
    """
    from buildml.session.facade_registry import resolve_operation_name

    operation = resolve_operation_name(operation)
    spec = get_operation(operation)
    primer = primer_for(operation, level=LearningLevel.coerce(level))
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
            beginner=primer,
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
        beginner=primer,
    )


def explain(
    session: Any,
    operation: str | None = None,
    *,
    moment: Literal["before", "after"] = "before",
    level: LearningLevel | str | None = LearningLevel.BEGINNER,
) -> BeforeOperationExplanation | AfterOperationExplanation | tuple[WorkflowStep, ...]:
    """Answer "what should I do", "should I run this", or "what just happened".

    One entry point covers all three because a caller rarely knows in advance
    which question they have. Omitting ``operation`` surveys the whole workflow;
    naming one explains it before or after the fact.

    Parameters
    ----------
    session:
        The session to explain. Nothing on it is modified.
    operation:
        A catalog operation name (flat or facade form
        ``domain.method`` / ``session.domain.method``), or ``None`` to resolve
        the whole workflow.
    moment:
        ``'before'`` to assess a choice not yet made, ``'after'`` to interpret a
        run that already happened. Ignored when ``operation`` is ``None``.
    level:
        ``'beginner'`` (the default), ``'intermediate'``, or ``'advanced'``.

    Returns
    -------
    BeforeOperationExplanation or AfterOperationExplanation or tuple of WorkflowStep
        The before or after explanation, or the resolved workflow when no
        operation was named. Named explanations always emit the canonical flat
        operation key.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No catalog operation has that name.
    ValueError
        ``moment`` is neither ``'before'`` nor ``'after'``, or ``level`` is not
        one of the three reading levels.
    """
    if operation is None:
        return resolve_workflow(session)
    if moment == "before":
        return explain_before(session, operation, level=level)
    if moment == "after":
        return explain_after(session, operation, level=level)
    raise ValueError("moment must be 'before' or 'after'")
