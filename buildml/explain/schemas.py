"""Typed, dependency-free schemas used by explanations and reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


def _json_value(value: Any) -> JsonValue:
    """Convert supported schema values into JSON-compatible Python values."""
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class DecisionOrigin(str, Enum):
    """Where a workflow choice came from."""

    AUTOMATIC = "automatic"
    RECOMMENDED = "recommended"
    EXPLICIT = "explicit"


class WorkflowStepStatus(str, Enum):
    """Lifecycle state of an operation in a workflow."""

    AVAILABLE = "available"
    DONE = "done"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    # Backward-compatible names used by the first explanation schemas.
    READY = "available"
    COMPLETED = "done"


class OperationKind(str, Enum):
    """Broad role an operation plays in a BuildML workflow."""

    INGEST = "ingest"
    CONFIGURE = "configure"
    INSPECT = "inspect"
    SPLIT = "split"
    TRANSFORM = "transform"
    MODEL = "model"
    DIAGNOSTIC = "diagnostic"
    PERSIST = "persist"
    EXPORT = "export"


class PrerequisiteStatus(str, Enum):
    """Whether a prerequisite has been established."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class EvidenceKind(str, Enum):
    """Source category for evidence supporting a finding."""

    OBSERVATION = "observation"
    METRIC = "metric"
    TEST = "test"
    ARTIFACT = "artifact"
    CONFIGURATION = "configuration"


class FindingSeverity(str, Enum):
    """Editorial severity, ordered by likely workflow impact."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionPriority(str, Enum):
    """Suggested urgency of a recommendation."""

    OPTIONAL = "optional"
    NEXT = "next"
    BEFORE_MODELING = "before_modeling"
    BEFORE_RELEASE = "before_release"


@dataclass(frozen=True, slots=True)
class SerializableSchema:
    """Mixin providing a strict JSON-compatible representation."""

    def to_dict(self) -> dict[str, JsonValue]:
        return {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class ParameterSpec(SerializableSchema):
    """User-facing description of one operation parameter."""

    name: str
    type_name: str
    description: str
    required: bool = False
    default: JsonValue = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Prerequisite(SerializableSchema):
    """A condition that should hold before an operation is run."""

    key: str
    description: str
    status: PrerequisiteStatus = PrerequisiteStatus.REQUIRED
    check_hint: str | None = None


@dataclass(frozen=True, slots=True)
class Evidence(SerializableSchema):
    """A traceable observation used to support a finding."""

    key: str
    kind: EvidenceKind
    summary: str
    value: JsonValue = None
    source: str | None = None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Finding(SerializableSchema):
    """An interpreted result tied to explicit evidence."""

    key: str
    title: str
    detail: str
    severity: FindingSeverity = FindingSeverity.INFO
    evidence: tuple[Evidence, ...] = ()
    affected_columns: tuple[str, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Action(SerializableSchema):
    """A concrete step a user can take."""

    key: str
    label: str
    operation: str | None = None
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Recommendation(SerializableSchema):
    """Advice with rationale, priority, and optional executable action."""

    key: str
    title: str
    rationale: str
    priority: ActionPriority
    action: Action | None = None
    based_on: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConceptNote(SerializableSchema):
    """Reusable technical note linked from operation specifications.

    ``details`` remains the searchable flat paragraph list. Optional section
    fields power Concept Academy long-form teaching when present.
    """

    key: str
    title: str
    summary: str
    details: tuple[str, ...]
    related_concepts: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    definition: str = ""
    intuition: str = ""
    formal_idea: str = ""
    why_it_matters: tuple[str, ...] = ()
    how_buildml_uses: tuple[str, ...] = ()
    interpretation_rules: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    anti_patterns: tuple[str, ...] = ()
    worked_example_pattern: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkflowStep(SerializableSchema):
    """Serializable status record for a workflow operation."""

    operation: str
    status: WorkflowStepStatus
    origin: DecisionOrigin
    summary: str
    blockers: tuple[str, ...] = ()
    prerequisite_chain: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    repeatable: bool = False
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True, slots=True)
class BeforeAfterExplanation(SerializableSchema):
    """Explain a state transition without storing opaque runtime objects."""

    operation: str
    before: Mapping[str, JsonValue]
    after: Mapping[str, JsonValue]
    changes: tuple[str, ...]
    unchanged: tuple[str, ...] = ()
    origin: DecisionOrigin = DecisionOrigin.EXPLICIT
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BeforeOperationExplanation(SerializableSchema):
    """Context needed to make an informed operation choice."""

    operation: str
    purpose: str
    pipeline_role: str
    status: WorkflowStepStatus
    prerequisite_status: Mapping[str, bool]
    prerequisite_chain: tuple[str, ...]
    appropriateness: tuple[str, ...]
    alternatives: tuple[str, ...]
    risks: tuple[str, ...]
    likely_state_changes: tuple[str, ...]
    concept_notes: tuple[ConceptNote, ...]


@dataclass(frozen=True, slots=True)
class AfterOperationExplanation(SerializableSchema):
    """Observed operation outcome interpreted against its catalog contract."""

    operation: str
    sequence: int | None
    parameters: Mapping[str, JsonValue]
    result_summary: Mapping[str, JsonValue]
    decision_origin: DecisionOrigin
    why_applied: tuple[str, ...]
    state_changes: tuple[str, ...]
    interpretation: tuple[str, ...]
    limitations: tuple[str, ...]
    next_valid_choices: tuple[str, ...]
    concept_notes: tuple[ConceptNote, ...]


@dataclass(frozen=True, slots=True)
class OperationSpec(SerializableSchema):
    """Editorial contract for one public :class:`buildml.Session` operation."""

    name: str
    kind: OperationKind
    definition: str
    purpose: str
    pipeline_role: str
    mechanism: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    prerequisites: tuple[Prerequisite, ...]
    usual_ordering: tuple[str, ...]
    alternatives: tuple[str, ...]
    selection_rationale: tuple[str, ...]
    assumptions: tuple[str, ...]
    failure_modes: tuple[str, ...]
    leakage_risks: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    state_changes: tuple[str, ...]
    result_reading: tuple[str, ...]
    next_considerations: tuple[str, ...]
    concept_links: tuple[str, ...]

