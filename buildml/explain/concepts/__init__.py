# ruff: noqa: E501
"""Shared concept notes referenced by the operation catalog and Concept Academy.

Technical prose lives in one module per domain and is merged here. The matching
beginner layer from :mod:`buildml.explain.beginner` is then folded onto each note
so a single :class:`~buildml.explain.schemas.ConceptNote` carries all three
reading levels: plain language first, then the working definition, then the
formal statement.

The merge also derives the navigational fields a learner needs — which glossary
terms the note uses, which concepts to read first, and which to read next — from
material that already exists, so nothing is authored twice.
"""

from __future__ import annotations

from dataclasses import replace

from buildml.explain.beginner import BEGINNER_LAYERS, BeginnerLayer
from buildml.explain.concepts.activelearning import ACTIVELEARNING_NOTES
from buildml.explain.concepts.ai import AI_NOTES
from buildml.explain.concepts.anomaly import ANOMALY_NOTES
from buildml.explain.concepts.automl import AUTOML_NOTES
from buildml.explain.concepts.causal import CAUSAL_NOTES
from buildml.explain.concepts.cbr import CBR_NOTES
from buildml.explain.concepts.classical import CLASSICAL_NOTES
from buildml.explain.concepts.dl import DL_NOTES
from buildml.explain.concepts.ensemble import ENSEMBLE_NOTES
from buildml.explain.concepts.federated import FEDERATED_NOTES
from buildml.explain.concepts.forecasting import FORECASTING_NOTES
from buildml.explain.concepts.graph import GRAPH_NOTES
from buildml.explain.concepts.kg import KG_NOTES
from buildml.explain.concepts.metalearning import METALEARNING_NOTES
from buildml.explain.concepts.multitask import MULTITASK_NOTES
from buildml.explain.concepts.nlp import NLP_NOTES
from buildml.explain.concepts.online import ONLINE_NOTES
from buildml.explain.concepts.optimize import OPTIMIZE_NOTES
from buildml.explain.concepts.probabilistic import PROBABILISTIC_NOTES
from buildml.explain.concepts.rag import RAG_NOTES
from buildml.explain.concepts.ranking import RANKING_NOTES
from buildml.explain.concepts.recommenders import RECOMMENDER_NOTES
from buildml.explain.concepts.rl import RL_NOTES
from buildml.explain.concepts.selfsupervised import SELFSUPERVISED_NOTES
from buildml.explain.concepts.semisupervised import SEMISUPERVISED_NOTES
from buildml.explain.concepts.symbolic import SYMBOLIC_NOTES
from buildml.explain.concepts.synthetic import SYNTHETIC_NOTES
from buildml.explain.concepts.tda import TDA_NOTES
from buildml.explain.concepts.teaching import TEACHING_NOTES
from buildml.explain.concepts.timeseries import TIMESERIES_NOTES
from buildml.explain.concepts.unsupervised import UNSUPERVISED_NOTES
from buildml.explain.glossary import detect_terms, require
from buildml.explain.schemas import ConceptDifficulty, ConceptNote, GlossaryTerm, LearningLevel

_TECHNICAL_NOTES: dict[str, ConceptNote] = {
    **CLASSICAL_NOTES,
    **DL_NOTES,
    **RAG_NOTES,
    **AI_NOTES,
    **UNSUPERVISED_NOTES,
    **ENSEMBLE_NOTES,
    **AUTOML_NOTES,
    **FORECASTING_NOTES,
    **ANOMALY_NOTES,
    **SEMISUPERVISED_NOTES,
    **SELFSUPERVISED_NOTES,
    **ACTIVELEARNING_NOTES,
    **ONLINE_NOTES,
    **MULTITASK_NOTES,
    **METALEARNING_NOTES,
    **FEDERATED_NOTES,
    **PROBABILISTIC_NOTES,
    **CAUSAL_NOTES,
    **GRAPH_NOTES,
    **SYMBOLIC_NOTES,
    **CBR_NOTES,
    **RL_NOTES,
    **TDA_NOTES,
    **RECOMMENDER_NOTES,
    **RANKING_NOTES,
    **KG_NOTES,
    **OPTIMIZE_NOTES,
    **SYNTHETIC_NOTES,
    **NLP_NOTES,
    **TEACHING_NOTES,
    **TIMESERIES_NOTES,
}

_MAX_DETECTED_TERMS = 8


def _glossary_for(layer: BeginnerLayer, note: ConceptNote) -> tuple[GlossaryTerm, ...]:
    """Resolve the note's declared jargon, then top up from the prose it uses.

    Declared terms come first and in the author's order, because those are the
    words the beginner layer deliberately leans on. Detected terms fill the rest
    so a reader never meets an undefined word that the glossary already knows.
    """
    entries: list[GlossaryTerm] = []
    seen: set[str] = set()
    for term in layer.glossary_terms:
        entry = require(term)
        if entry.term.lower() not in seen:
            entries.append(entry)
            seen.add(entry.term.lower())
    remaining = _MAX_DETECTED_TERMS - len(entries)
    if remaining > 0:
        detected = detect_terms(
            (layer.plain_summary, note.summary, note.definition),
            limit=remaining,
            exclude=seen,
        )
        entries.extend(detected)
    return tuple(entries)


def _learning_path(
    note: ConceptNote,
    difficulty: ConceptDifficulty,
    difficulties: dict[str, ConceptDifficulty],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split related concepts into read-first and read-next by relative depth."""
    before: list[str] = []
    after: list[str] = []
    for key in note.related_concepts:
        other = difficulties.get(key)
        if other is None:
            continue
        if other.rank < difficulty.rank:
            before.append(key)
        else:
            after.append(key)
    return tuple(before), tuple(after)


def _merge(note: ConceptNote, layer: BeginnerLayer, difficulties: dict[str, ConceptDifficulty]) -> ConceptNote:
    """Fold one beginner layer onto its technical note."""
    prerequisites, next_concepts = _learning_path(note, layer.difficulty, difficulties)
    # Beginner prose leads ``details`` so keyword search surfaces the readable
    # phrasing rather than only the formal statement.
    details = (layer.plain_summary, layer.analogy, *layer.steps, *note.details)
    return replace(
        note,
        details=details,
        plain_summary=layer.plain_summary,
        analogy=layer.analogy,
        beginner_steps=layer.steps,
        when_to_use=layer.when_to_use,
        when_not_to_use=layer.when_not_to_use,
        misconceptions=layer.misconceptions,
        glossary=_glossary_for(layer, note),
        mini_example=layer.mini_example,
        check_yourself=layer.check_yourself,
        buildml_tools=layer.tools,
        prerequisite_concepts=prerequisites,
        next_concepts=next_concepts,
        difficulty=layer.difficulty,
    )


def _build() -> dict[str, ConceptNote]:
    missing = sorted(set(_TECHNICAL_NOTES) - set(BEGINNER_LAYERS))
    if missing:
        raise ValueError(
            "Every concept note must ship a beginner layer; missing: " + ", ".join(missing)
        )
    orphans = sorted(set(BEGINNER_LAYERS) - set(_TECHNICAL_NOTES))
    if orphans:
        raise ValueError(
            "Beginner layers must describe a real concept; unknown: " + ", ".join(orphans)
        )
    difficulties = {key: layer.difficulty for key, layer in BEGINNER_LAYERS.items()}
    return {
        key: _merge(note, BEGINNER_LAYERS[key], difficulties)
        for key, note in _TECHNICAL_NOTES.items()
    }


CONCEPT_NOTES: dict[str, ConceptNote] = _build()


def get_concept(key: str) -> ConceptNote:
    """Fetch one teaching note, with its beginner layer already merged in.

    Notes are authored in two halves — the technical note and the beginner
    layer — and joined at import. Callers always receive the joined note, so no
    surface can accidentally render only the expert phrasing.

    Parameters
    ----------
    key:
        A concept key, such as ``'leakage-boundary'``.

    Returns
    -------
    ~buildml.explain.schemas.ConceptNote
        The merged note.

    Raises
    ------
    KeyError
        No concept has that key.

    See Also
    --------
    buildml.explain.learn : Resolves concepts, operations, and jargon together.
    """
    try:
        return CONCEPT_NOTES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown BuildML concept: {key}") from exc


def list_concepts() -> tuple[ConceptNote, ...]:
    """List every teaching note in a stable order.

    Sorting by key rather than by authoring order keeps generated indexes and
    coverage reports from churning when a domain module is edited.

    Returns
    -------
    tuple of ~buildml.explain.schemas.ConceptNote
        All notes, sorted by key.
    """
    return tuple(CONCEPT_NOTES[key] for key in sorted(CONCEPT_NOTES))


def concepts_at(difficulty: ConceptDifficulty | str) -> tuple[ConceptNote, ...]:
    """Select the concepts on one rung of the learning ladder.

    Every note is tagged ``foundation``, ``core``, or ``advanced``. Filtering by
    rung is what lets a caller build an entry-level reading list without
    hand-maintaining one alongside the notes.

    Parameters
    ----------
    difficulty:
        A :class:`~buildml.explain.schemas.ConceptDifficulty` member or its
        string value.

    Returns
    -------
    tuple of ~buildml.explain.schemas.ConceptNote
        Notes at that rung, in stable key order.

    Raises
    ------
    ValueError
        The string does not name a difficulty rung.
    """
    wanted = difficulty if isinstance(difficulty, ConceptDifficulty) else ConceptDifficulty(str(difficulty).strip().lower())
    return tuple(note for note in list_concepts() if note.difficulty is wanted)


def learning_path(key: str, *, level: LearningLevel | str | None = None) -> tuple[str, ...]:
    """Order the concepts a reader needs before one they asked about.

    Walks ``prerequisite_concepts`` depth-first, so a newcomer is never handed a
    note whose vocabulary depends on one they have not read. Cycles and unknown
    keys are skipped rather than raising, since a broken link should degrade the
    reading order rather than the explanation.

    Parameters
    ----------
    key:
        The concept to arrive at. Unknown keys return just that key.
    level:
        At ``'beginner'`` the walk stops at foundation-level material instead of
        dragging in advanced neighbours. ``None`` behaves as ``'beginner'``.

    Returns
    -------
    tuple of str
        Concept keys in reading order, always ending at ``key``.

    Examples
    --------
    >>> from buildml.explain import learning_path
    >>> learning_path("cross-validation")[-1]
    'cross-validation'
    """
    tier = LearningLevel.coerce(level)
    order: list[str] = []
    seen: set[str] = set()

    def visit(current: str) -> None:
        if current in seen:
            return
        seen.add(current)
        note = CONCEPT_NOTES.get(current)
        if note is None:
            return
        for parent in note.prerequisite_concepts:
            parent_note = CONCEPT_NOTES.get(parent)
            if parent_note is None:
                continue
            if tier is LearningLevel.BEGINNER and parent_note.difficulty is ConceptDifficulty.ADVANCED:
                continue
            visit(parent)
        order.append(current)

    visit(key)
    return tuple(order)


__all__ = [
    "CONCEPT_NOTES",
    "concepts_at",
    "get_concept",
    "learning_path",
    "list_concepts",
]
