# ruff: noqa: E501
"""Topic lookup across concepts and operations, with a reading order.

:mod:`buildml.explain.concepts` answers "what does this concept say?" and
:mod:`buildml.explain.pedagogy` answers "what does this operation do?". This
module is the front door over both: you name a topic, it works out whether you
mean a concept or an operation, and it returns the material plus the order in
which to read it.

It exists because a beginner does not know which of the two they are asking
about. ``learn("split")`` and ``learn("data-splitting")`` are the same question
asked from either side of the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache

from buildml.explain.catalog import OPERATION_CATALOG
from buildml.explain.concepts import CONCEPT_NOTES, learning_path
from buildml.explain.glossary import GLOSSARY, concept_for_term, lookup
from buildml.explain.pedagogy import primer_for
from buildml.explain.schemas import (
    ConceptDifficulty,
    ConceptNote,
    GlossaryTerm,
    LearningLevel,
    OperationPrimer,
    SerializableSchema,
)


@dataclass(frozen=True, slots=True)
class LearningBrief(SerializableSchema):
    """Everything BuildML can teach about one topic, at one reading level.

    Exactly one of ``concept`` or ``operation`` is populated for a resolved
    topic; both are empty for the starting-point index, where ``suggested`` is
    the useful field.
    """

    topic: str
    level: LearningLevel
    kind: str
    concept: ConceptNote | None
    operation: OperationPrimer | None
    glossary_term: GlossaryTerm | None
    read_first: tuple[ConceptNote, ...]
    read_next: tuple[ConceptNote, ...]
    related_operations: tuple[str, ...]
    suggested: tuple[str, ...]


_RELATED_OPERATION_LIMIT = 12

_DIFFICULTY_ORDER: dict[ConceptDifficulty, int] = {
    ConceptDifficulty.FOUNDATION: 0,
    ConceptDifficulty.CORE: 1,
    ConceptDifficulty.ADVANCED: 2,
}


def _operations_for_concept(key: str) -> tuple[str, ...]:
    """Operations that teach this concept, most representative first.

    A concept like ``leakage-boundary`` is linked from well over a hundred
    operations, so an alphabetical slice would hand a beginner whatever happens
    to start with 'a'. Operations the concept itself names as its tools come
    first, then operations that list this concept as their primary link.
    """
    note = CONCEPT_NOTES.get(key)
    preferred = set(note.buildml_tools) if note is not None else set()

    def rank(item: tuple[str, object]) -> tuple[int, int, str]:
        name, spec = item
        links = spec.concept_links  # type: ignore[attr-defined]
        return (0 if name in preferred else 1, links.index(key), name)

    matches = [
        (name, spec)
        for name, spec in OPERATION_CATALOG.items()
        if key in spec.concept_links
    ]
    matches.sort(key=rank)
    return tuple(name for name, _ in matches[:_RELATED_OPERATION_LIMIT])


def _starting_points() -> tuple[str, ...]:
    """Foundation concepts first, then the core ones, in stable key order."""
    ladder = (ConceptDifficulty.FOUNDATION, ConceptDifficulty.CORE)
    return tuple(
        key
        for rung in ladder
        for key in sorted(CONCEPT_NOTES)
        if CONCEPT_NOTES[key].difficulty is rung
    )


def _index_brief(level: LearningLevel) -> LearningBrief:
    foundation = tuple(
        CONCEPT_NOTES[key]
        for key in sorted(CONCEPT_NOTES)
        if CONCEPT_NOTES[key].difficulty is ConceptDifficulty.FOUNDATION
    )
    return LearningBrief(
        topic="",
        level=level,
        kind="index",
        concept=None,
        operation=None,
        glossary_term=None,
        read_first=foundation,
        read_next=(),
        related_operations=(),
        suggested=_starting_points()[:24],
    )


def _concept_brief(key: str, level: LearningLevel) -> LearningBrief:
    note = CONCEPT_NOTES[key]
    path = tuple(
        CONCEPT_NOTES[item] for item in learning_path(key, level=level) if item != key
    )
    following = tuple(
        CONCEPT_NOTES[item] for item in note.next_concepts if item in CONCEPT_NOTES
    )
    return LearningBrief(
        topic=key,
        level=level,
        kind="concept",
        concept=note,
        operation=None,
        glossary_term=lookup(note.title),
        read_first=path,
        read_next=following,
        related_operations=_operations_for_concept(key),
        suggested=note.buildml_tools,
    )


def _operation_brief(name: str, level: LearningLevel) -> LearningBrief:
    spec = OPERATION_CATALOG[name]
    linked = tuple(
        CONCEPT_NOTES[key] for key in spec.concept_links if key in CONCEPT_NOTES
    )
    prerequisites: list[ConceptNote] = []
    for note in linked:
        for key in note.prerequisite_concepts:
            other = CONCEPT_NOTES.get(key)
            if other is not None and other not in prerequisites and other not in linked:
                prerequisites.append(other)
    return LearningBrief(
        topic=name,
        level=level,
        kind="operation",
        concept=None,
        operation=primer_for(name, level=level),
        glossary_term=None,
        read_first=tuple(prerequisites),
        read_next=linked,
        related_operations=spec.alternatives,
        suggested=spec.next_considerations,
    )


@lru_cache(maxsize=1)
def _concept_by_declared_term() -> dict[str, str]:
    """Term (lowercased) → the concept note that declares it in its glossary.

    ``CONCEPT_FOR_TERM`` is a hand-curated shortlist for the terms with an
    obvious home. Every beginner layer also declares the vocabulary it teaches,
    which covers far more ground, so a term with no curated mapping can still
    land on a real concept instead of a bare definition. Foundation concepts win
    ties, because a beginner asking about a word wants the gentlest owner of it.
    """
    ranked = sorted(
        CONCEPT_NOTES.values(),
        key=lambda note: (_DIFFICULTY_ORDER[note.difficulty], note.key),
    )
    index: dict[str, str] = {}
    for note in ranked:
        for entry in note.glossary:
            for name in (entry.term, *entry.also_called):
                index.setdefault(name.casefold(), note.key)
    return index


def _concept_for(term: str) -> str | None:
    curated = concept_for_term(term)
    if curated is not None:
        return curated
    entry = lookup(term)
    if entry is None:
        return None
    index = _concept_by_declared_term()
    for name in (entry.term, *entry.also_called):
        key = index.get(name.casefold())
        if key is not None:
            return key
    return None


def _glossary_brief(term: str, level: LearningLevel) -> LearningBrief | None:
    entry = lookup(term)
    if entry is None:
        return None
    key = _concept_for(term)
    if key is not None and key in CONCEPT_NOTES:
        brief = _concept_brief(key, level)
        return LearningBrief(
            topic=entry.term,
            level=level,
            kind="term",
            concept=brief.concept,
            operation=None,
            glossary_term=entry,
            read_first=brief.read_first,
            read_next=brief.read_next,
            related_operations=brief.related_operations,
            suggested=brief.suggested,
        )
    return LearningBrief(
        topic=entry.term,
        level=level,
        kind="term",
        concept=None,
        operation=None,
        glossary_term=entry,
        read_first=(),
        read_next=(),
        related_operations=(),
        suggested=_starting_points()[:8],
    )


def _suggest(topic: str) -> tuple[str, ...]:
    pool = [*CONCEPT_NOTES, *OPERATION_CATALOG, *GLOSSARY]
    return tuple(get_close_matches(topic, pool, n=5, cutoff=0.5))


def _spellings(topic: str) -> tuple[str, ...]:
    """Forms of a topic worth trying, since nobody remembers the punctuation.

    'ROC AUC', 'roc-auc', and 'roc_auc' are the same question. Concept keys are
    hyphenated, operation names use underscores, and glossary terms are spaced,
    so a beginner typing any one of them should land on the right entry.
    """
    base = topic.strip()
    folded = base.casefold()
    forms = (
        base,
        folded,
        folded.replace(" ", "-").replace("_", "-"),
        folded.replace(" ", "_").replace("-", "_"),
        folded.replace("-", " ").replace("_", " "),
    )
    seen: set[str] = set()
    return tuple(form for form in forms if form and not (form in seen or seen.add(form)))


def learn(
    topic: str | None = None,
    *,
    level: LearningLevel | str | None = LearningLevel.BEGINNER,
) -> LearningBrief:
    """Teach one topic, and say what should be read before and after it.

    A beginner does not know whether the thing confusing them is a concept, an
    operation, or a word, so this resolves all three from one argument. Concept
    keys are tried first, then operation names, then the glossary, and each is
    tried under several spellings: ``'ROC AUC'``, ``'roc-auc'``, and
    ``'roc_auc'`` are the same question.

    Parameters
    ----------
    topic:
        A concept key (``'leakage-boundary'``), an operation name
        (``'split'``), or a piece of jargon (``'overfitting'``). ``None``
        returns the foundation reading list, which is where a newcomer should
        start.
    level:
        ``'beginner'`` (the default), ``'intermediate'``, or ``'advanced'``.
        Affects how much of an operation primer is rendered; concept notes are
        returned whole at every level.

    Returns
    -------
    LearningBrief
        The resolved subject plus a reading order. ``kind`` says which of
        ``'concept'``, ``'operation'``, ``'term'``, or ``'index'`` was matched.

    Raises
    ------
    KeyError
        Nothing matches the topic. Close matches are named in the message.
    ValueError
        ``level`` is not one of the three reading levels.

    Examples
    --------
    >>> from buildml.explain import learn
    >>> learn("data-splitting").kind
    'concept'
    >>> learn("split").kind
    'operation'
    >>> learn("cardinality").kind
    'term'

    Jargon that a concept note is built around resolves to that note rather
    than to its one-line definition, so the reader lands on the teaching:

    >>> learn("overfitting").kind
    'concept'

    See Also
    --------
    buildml.explain.explain : What an operation does in a specific session.
    starting_points : The reading order for someone new to all of it.
    """
    tier = LearningLevel.coerce(level)
    if topic is None:
        return _index_brief(tier)
    name = str(topic).strip()
    for form in _spellings(name):
        if form in CONCEPT_NOTES:
            return _concept_brief(form, tier)
        if form in OPERATION_CATALOG:
            return _operation_brief(form, tier)
        term_brief = _glossary_brief(form, tier)
        if term_brief is not None:
            return term_brief
    hints = _suggest(name)
    suffix = f" Did you mean: {', '.join(hints)}?" if hints else ""
    raise KeyError(
        f"BuildML has no concept, operation, or glossary term named {name!r}.{suffix}"
    )


def starting_points() -> tuple[str, ...]:
    """List the concepts to learn first, in the order to learn them.

    An alphabetical index is useless to someone who does not yet know which
    entries matter. This returns the foundation concepts before the core ones,
    so the list can be read top to bottom.

    Returns
    -------
    tuple of str
        Concept keys, foundation rung first, alphabetical within each rung.

    See Also
    --------
    learn : Called with no topic, returns these as a brief.
    """
    return _starting_points()


__all__ = ["LearningBrief", "learn", "starting_points"]
