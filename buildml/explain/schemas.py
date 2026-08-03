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


def _normalize_prose_tuples(instance: Any, names: frozenset[str]) -> None:
    """Wrap a lone authored sentence into the one-item tuple it was meant to be.

    Catalog and concept entries are hand-written literals, and a single-item
    tuple needs a trailing comma that is easy to forget. Left alone, the string
    survives every type checker that only sees the annotation and then renders
    one bullet per character the first time a reader asks for an explanation.
    Normalizing at construction keeps that authoring slip out of the output.
    """
    for name in names:
        value = getattr(instance, name)
        if isinstance(value, str):
            object.__setattr__(instance, name, (value,) if value else ())
        elif not isinstance(value, tuple):
            object.__setattr__(instance, name, tuple(value))


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


class LearningLevel(str, Enum):
    """Reader experience tier a rendered explanation is written for.

    BuildML explanations are layered rather than duplicated: one note carries
    beginner, intermediate, and advanced material, and the renderer chooses how
    much to show. ``BEGINNER`` never assumes prior machine-learning vocabulary.
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

    @classmethod
    def coerce(cls, value: LearningLevel | str | None) -> LearningLevel:
        """Normalize whatever a caller passed into a reading level.

        Public methods accept plain strings so users never have to import an
        enum to ask for a level. Every entry point routes through here, which is
        also where an unrecognised level is rejected: quietly falling back to a
        default would leave someone believing they had asked for more depth than
        they received.

        Parameters
        ----------
        value:
            An enum member, one of ``'beginner'`` / ``'intermediate'`` /
            ``'advanced'`` in any case, or ``None`` for the default.

        Returns
        -------
        LearningLevel
            The matching member; ``BEGINNER`` when ``value`` is ``None``.

        Raises
        ------
        ValueError
            The string does not name a reading level. The message lists the
            valid ones.
        """
        if value is None:
            return cls.BEGINNER
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        valid = ", ".join(member.value for member in cls)
        raise ValueError(f"Unknown learning level {value!r}; expected one of: {valid}")

    @property
    def rank(self) -> int:
        """Ordinal depth so renderers can compare tiers."""
        return _LEARNING_LEVEL_ORDER[self]

    def includes(self, other: LearningLevel) -> bool:
        """Decide whether material written for another level belongs at this one.

        Levels are cumulative in one direction: an advanced reader can be shown
        anything, while a beginner should not be handed material that assumes
        vocabulary they have not met.

        Parameters
        ----------
        other:
            The level the material was authored for.

        Returns
        -------
        bool
            ``True`` when this level is at least as deep as ``other``.
        """
        return other.rank <= self.rank


_LEARNING_LEVEL_ORDER: dict[LearningLevel, int] = {
    LearningLevel.BEGINNER: 0,
    LearningLevel.INTERMEDIATE: 1,
    LearningLevel.ADVANCED: 2,
}


class ConceptDifficulty(str, Enum):
    """Where a concept sits on the BuildML learning ladder.

    ``FOUNDATION`` concepts are safe first reads for a complete beginner.
    ``CORE`` concepts assume the foundations. ``ADVANCED`` concepts assume both
    and usually describe a specialized domain surface.
    """

    FOUNDATION = "foundation"
    CORE = "core"
    ADVANCED = "advanced"

    @property
    def rank(self) -> int:
        """Ordinal depth, so rungs can be compared and sorted."""
        return _CONCEPT_DIFFICULTY_ORDER[self]


_CONCEPT_DIFFICULTY_ORDER: dict[ConceptDifficulty, int] = {
    ConceptDifficulty.FOUNDATION: 0,
    ConceptDifficulty.CORE: 1,
    ConceptDifficulty.ADVANCED: 2,
}


@dataclass(frozen=True, slots=True)
class SerializableSchema:
    """Mixin providing a strict JSON-compatible representation."""

    def to_dict(self) -> dict[str, JsonValue]:
        """Render the schema as plain JSON-compatible Python.

        Explanations travel: into transcripts, HTML reports, AI tool results,
        and generated indexes. Conversion is strict rather than best-effort, so
        a value that cannot round-trip raises here instead of producing an
        artifact that silently lost a field.

        Returns
        -------
        dict
            Field name mapped to a value built only from ``None``, ``bool``,
            ``int``, ``float``, ``str``, ``list``, and ``dict``. Enums become
            their values, paths become strings, and nested schemas recurse.

        Raises
        ------
        TypeError
            A field holds something with no JSON representation.
        """
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
class GlossaryTerm(SerializableSchema):
    """One piece of jargon translated into everyday language.

    Glossary entries are what make an explanation readable without prior
    training: any term a beginner is unlikely to know should appear here with a
    plain meaning, so the reader never has to leave the explanation to decode it.
    """

    term: str
    plain_meaning: str
    also_called: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Misconception(SerializableSchema):
    """A belief many learners arrive with, paired with what is actually true."""

    myth: str
    reality: str


@dataclass(frozen=True, slots=True)
class ParameterMeaning(SerializableSchema):
    """Plain-language reading of one knob: what it controls and how to move it."""

    name: str
    plain_meaning: str
    effect_of_increase: str = ""
    effect_of_decrease: str = ""
    typical_choice: str = ""


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
    """Reusable teaching note linked from operation specifications.

    A note is written in three layers so one artifact serves every reader:

    * **Beginner layer**: ``plain_summary``, ``analogy``, ``beginner_steps``,
      ``when_to_use`` / ``when_not_to_use``, ``misconceptions``, ``glossary``,
      ``mini_example``, ``check_yourself``. No prior ML vocabulary assumed.
    * **Intermediate layer**: ``definition``, ``why_it_matters``,
      ``how_buildml_uses``, ``interpretation_rules``, ``buildml_tools``.
    * **Advanced layer**: ``formal_idea``, ``assumptions``, ``failure_modes``,
      ``anti_patterns``, ``worked_example_pattern``.

    ``details`` remains the searchable flat paragraph list and is rebuilt to
    include beginner prose so search never returns only the expert phrasing.
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
    # --- beginner layer -------------------------------------------------
    plain_summary: str = ""
    analogy: str = ""
    beginner_steps: tuple[str, ...] = ()
    when_to_use: tuple[str, ...] = ()
    when_not_to_use: tuple[str, ...] = ()
    misconceptions: tuple[Misconception, ...] = ()
    glossary: tuple[GlossaryTerm, ...] = ()
    mini_example: tuple[str, ...] = ()
    check_yourself: tuple[str, ...] = ()
    # --- navigation -----------------------------------------------------
    buildml_tools: tuple[str, ...] = ()
    prerequisite_concepts: tuple[str, ...] = ()
    next_concepts: tuple[str, ...] = ()
    difficulty: ConceptDifficulty = ConceptDifficulty.CORE

    def __post_init__(self) -> None:
        _normalize_prose_tuples(self, _CONCEPT_NOTE_SEQUENCES)

    @property
    def has_beginner_layer(self) -> bool:
        """True when the note satisfies the beginner content standard."""
        return bool(
            self.plain_summary
            and self.analogy
            and self.beginner_steps
            and self.when_to_use
            and self.when_not_to_use
            and self.misconceptions
            and self.mini_example
        )


#: Prose-sequence fields of :class:`ConceptNote`, normalized on construction.
_CONCEPT_NOTE_SEQUENCES = frozenset(
    {
        "details",
        "related_concepts",
        "references",
        "why_it_matters",
        "how_buildml_uses",
        "interpretation_rules",
        "assumptions",
        "failure_modes",
        "anti_patterns",
        "worked_example_pattern",
        "beginner_steps",
        "when_to_use",
        "when_not_to_use",
        "mini_example",
        "check_yourself",
        "buildml_tools",
        "prerequisite_concepts",
        "next_concepts",
    }
)


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
    beginner: OperationPrimer | None = None


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
    beginner: OperationPrimer | None = None


@dataclass(frozen=True, slots=True)
class OperationSpec(SerializableSchema):
    """Editorial contract for one public :class:`buildml.Session` operation.

    Beginner-facing fields (``plain_summary`` through ``mini_example``) are
    optional at authoring time: :mod:`buildml.explain.pedagogy` derives them
    from the operation kind, prerequisites, parameters, and linked concept notes
    so no operation can ship without a plain-language layer.
    """

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
    plain_summary: str = ""
    analogy: str = ""
    beginner_steps: tuple[str, ...] = ()
    when_to_use: tuple[str, ...] = ()
    when_not_to_use: tuple[str, ...] = ()
    mini_example: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _normalize_prose_tuples(self, _OPERATION_SPEC_SEQUENCES)


#: Prose-sequence fields of :class:`OperationSpec`, normalized on construction.
_OPERATION_SPEC_SEQUENCES = frozenset(
    {
        "mechanism",
        "inputs",
        "outputs",
        "usual_ordering",
        "alternatives",
        "selection_rationale",
        "assumptions",
        "failure_modes",
        "leakage_risks",
        "anti_patterns",
        "state_changes",
        "result_reading",
        "next_considerations",
        "concept_links",
        "beginner_steps",
        "when_to_use",
        "when_not_to_use",
        "mini_example",
    }
)


@dataclass(frozen=True, slots=True)
class OperationPrimer(SerializableSchema):
    """Beginner-first briefing attached to every operation explanation.

    The primer answers the questions a newcomer actually asks: *what is this,
    why would I run it, what has to be true first, what will change, what words
    am I looking at*: before any expert prose appears.
    """

    operation: str
    level: LearningLevel
    plain_summary: str
    analogy: str
    why_it_exists: str
    steps: tuple[str, ...]
    prerequisites_in_plain_words: tuple[str, ...]
    when_to_use: tuple[str, ...]
    when_not_to_use: tuple[str, ...]
    key_parameters: tuple[ParameterMeaning, ...]
    what_changes: tuple[str, ...]
    how_to_read_the_result: tuple[str, ...]
    common_pitfalls: tuple[str, ...]
    glossary: tuple[GlossaryTerm, ...]
    mini_example: tuple[str, ...]
    related_tools: tuple[str, ...]
    learn_next: tuple[str, ...]

