"""Typed explanations and the public Session operation catalog.

The package has two halves that meet at :class:`OperationSpec`. The *reference*
half: :mod:`~buildml.explain.catalog`, :mod:`~buildml.explain.resolver`,
:mod:`~buildml.explain.history`: answers what an operation does and whether it
can run here. The *teaching* half: :mod:`~buildml.explain.concepts`,
:mod:`~buildml.explain.glossary`, :mod:`~buildml.explain.pedagogy`,
:mod:`~buildml.explain.academy`: answers what the idea behind it is, in plain
language, at whichever reading level the caller asks for.
"""

from buildml.explain.academy import LearningBrief, learn, starting_points
from buildml.explain.catalog import OPERATION_CATALOG, get_operation, list_operations
from buildml.explain.concepts import (
    CONCEPT_NOTES,
    concepts_at,
    get_concept,
    learning_path,
    list_concepts,
)
from buildml.explain.glossary import GLOSSARY, all_terms, detect_terms, lookup, require
from buildml.explain.history import HISTORY_SCHEMA_VERSION, normalize_history
from buildml.explain.pedagogy import derive_primer, primer_for
from buildml.explain.prerequisites import plain_prerequisite
from buildml.explain.resolver import (
    explain,
    explain_after,
    explain_before,
    prerequisite_status,
    resolve_workflow,
)
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    AfterOperationExplanation,
    BeforeAfterExplanation,
    BeforeOperationExplanation,
    ConceptDifficulty,
    ConceptNote,
    DecisionOrigin,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    GlossaryTerm,
    LearningLevel,
    Misconception,
    OperationKind,
    OperationPrimer,
    OperationSpec,
    ParameterMeaning,
    ParameterSpec,
    Prerequisite,
    PrerequisiteStatus,
    Recommendation,
    WorkflowStep,
    WorkflowStepStatus,
)

__all__ = [
    "CONCEPT_NOTES",
    "GLOSSARY",
    "HISTORY_SCHEMA_VERSION",
    "OPERATION_CATALOG",
    "Action",
    "ActionPriority",
    "AfterOperationExplanation",
    "BeforeAfterExplanation",
    "BeforeOperationExplanation",
    "ConceptDifficulty",
    "ConceptNote",
    "DecisionOrigin",
    "Evidence",
    "EvidenceKind",
    "Finding",
    "FindingSeverity",
    "GlossaryTerm",
    "LearningBrief",
    "LearningLevel",
    "Misconception",
    "OperationKind",
    "OperationPrimer",
    "OperationSpec",
    "ParameterMeaning",
    "ParameterSpec",
    "Prerequisite",
    "PrerequisiteStatus",
    "Recommendation",
    "WorkflowStep",
    "WorkflowStepStatus",
    "all_terms",
    "concepts_at",
    "derive_primer",
    "detect_terms",
    "explain",
    "explain_after",
    "explain_before",
    "get_concept",
    "get_operation",
    "learn",
    "learning_path",
    "list_concepts",
    "list_operations",
    "lookup",
    "normalize_history",
    "plain_prerequisite",
    "prerequisite_status",
    "primer_for",
    "require",
    "resolve_workflow",
    "starting_points",
]
