# ruff: noqa: E501
"""Shared catalog helpers and prerequisites for teaching overlays."""

from __future__ import annotations

from buildml.explain.concepts import CONCEPT_NOTES
from buildml.explain.schemas import (
    OperationKind,
    OperationSpec,
    ParameterSpec,
    Prerequisite,
    PrerequisiteStatus,
)


def _p(
    name: str,
    type_name: str,
    description: str,
    default: object = None,
    *,
    required: bool = False,
    choices: tuple[str, ...] = (),
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        type_name=type_name,
        description=description,
        required=required,
        default=default,  # type: ignore[arg-type]
        choices=choices,
    )


DATASET = Prerequisite(
    "dataset",
    "A materialized dataset is attached to the session.",
    check_hint="Session.metadata()['has_dataset'] is true.",
)
ROLES = Prerequisite(
    "roles",
    "Feature and target roles required by target-aware operations have been assigned and reviewed.",
    check_hint="Inspect Session.dataset.roles.",
)
SPLIT = Prerequisite(
    "split",
    "A train/evaluation split exists.",
    check_hint="Session.split_plan is not None.",
)
FIT = Prerequisite(
    "fit",
    "A compatible estimator has been fitted or loaded.",
    check_hint="Session.fit_result is not None.",
)
FIT_TORCH = Prerequisite(
    "fit_torch",
    "A Torch trainer has been fitted or a trainer bundle has been loaded.",
    check_hint="session.dl.train_result is not None.",
)
TORCH = Prerequisite(
    "torch-extra",
    "The optional Torch dependencies are installed for deep-learning methods.",
    status=PrerequisiteStatus.OPTIONAL,
    check_hint="Install buildml[torch] (or buildml[dl]) before Torch Session methods.",
)
RAG_CORPUS = Prerequisite(
    "rag-corpus",
    "A RAG corpus has been ingested on the Session.",
    check_hint="Session history includes session.rag.ingest_corpus or an equivalent corpus handle.",
)
RAG_INDEX = Prerequisite(
    "rag-index",
    "A RAG index has been built or a RAG bundle has been loaded.",
    check_hint="session.rag.index_result is not None.",
)
RAG = Prerequisite(
    "rag-extra",
    "The optional RAG dependencies are installed for retrieval methods.",
    status=PrerequisiteStatus.OPTIONAL,
    check_hint="Install buildml[rag] before RAG Session methods.",
)
VIZ = Prerequisite(
    "viz-extra",
    "The optional visualization dependencies are installed for rendered figures.",
    status=PrerequisiteStatus.OPTIONAL,
    check_hint="Install buildml[viz] only when plots are requested.",
)
DASHBOARD = Prerequisite(
    "dashboard-extra",
    "The optional dashboard dependencies are installed for the local Industry EDA App.",
    status=PrerequisiteStatus.OPTIONAL,
    check_hint="Install buildml[dashboard] before calling Session.eda_app(...).",
)
AI_PROVIDER = Prerequisite(
    "ai-provider",
    "An AI provider has been configured on the Session.",
    check_hint="Session history includes session.ai.configure or Session._ai_provider is not None.",
)
AI = Prerequisite(
    "ai-extra",
    "The optional AI dependencies are installed for LLM operator methods.",
    status=PrerequisiteStatus.OPTIONAL,
    check_hint="Install buildml[ai] before AI Session methods.",
)


def _operation(
    name: str,
    kind: OperationKind,
    definition: str,
    purpose: str,
    role: str,
    mechanism: tuple[str, ...],
    *,
    parameters: tuple[ParameterSpec, ...] = (),
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    prerequisites: tuple[Prerequisite, ...] = (),
    ordering: tuple[str, ...],
    alternatives: tuple[str, ...],
    rationale: tuple[str, ...],
    assumptions: tuple[str, ...],
    failures: tuple[str, ...],
    leakage: tuple[str, ...],
    anti_patterns: tuple[str, ...],
    state_changes: tuple[str, ...],
    result_reading: tuple[str, ...],
    next_steps: tuple[str, ...],
    concepts: tuple[str, ...],
    plain: str = "",
    analogy: str = "",
    beginner_steps: tuple[str, ...] = (),
    when_to_use: tuple[str, ...] = (),
    when_not_to_use: tuple[str, ...] = (),
    mini_example: tuple[str, ...] = (),
) -> OperationSpec:
    unknown = set(concepts) - CONCEPT_NOTES.keys()
    if unknown:
        raise ValueError(f"{name} links unknown concepts: {sorted(unknown)}")
    return OperationSpec(
        name=name,
        kind=kind,
        definition=definition,
        purpose=purpose,
        pipeline_role=role,
        mechanism=mechanism,
        parameters=parameters,
        inputs=inputs,
        outputs=outputs,
        prerequisites=prerequisites,
        usual_ordering=ordering,
        alternatives=alternatives,
        selection_rationale=rationale,
        assumptions=assumptions,
        failure_modes=failures,
        leakage_risks=leakage,
        anti_patterns=anti_patterns,
        state_changes=state_changes,
        result_reading=result_reading,
        next_considerations=next_steps,
        concept_links=concepts,
        plain_summary=plain,
        analogy=analogy,
        beginner_steps=beginner_steps,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        mini_example=mini_example,
    )
