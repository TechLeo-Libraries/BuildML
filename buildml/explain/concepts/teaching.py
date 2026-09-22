# ruff: noqa: E501
"""Concept notes about BuildML's own teaching surface."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

TEACHING_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="explain-learning-levels",
            title="Layered explanations: beginner, intermediate, advanced",
            summary=(
                "Every BuildML explanation carries all three reading levels in one "
                "artifact; the level you request decides how much is rendered, not "
                "which facts are true."
            ),
            definition=(
                "A BuildML explanation is layered rather than duplicated. One "
                "ConceptNote holds a beginner layer (plain summary, analogy, "
                "step-by-step, when to use and when not to, misconceptions, "
                "glossary, mini example, self-check), an intermediate layer "
                "(definition, why it matters, how BuildML uses it, interpretation "
                "rules), and an advanced layer (formal statement, assumptions, "
                "failure modes, anti-patterns). Every operation additionally "
                "carries a derived OperationPrimer written for a newcomer."
            ),
            intuition=(
                "One document, three depths of reading. A beginner is not handed a "
                "different, softer truth; they are handed the same truth with the "
                "vocabulary supplied."
            ),
            formal_idea=(
                "Levels increase in depth: beginner, intermediate, advanced. "
                "Operation primers introduce the parameters, prerequisites, and "
                "related concepts; advanced notes add assumptions and failure modes."
            ),
            why_it_matters=(
                "An explanation that assumes the vocabulary it is explaining is not an explanation.",
                "Operation primers help you understand the inputs before running an unfamiliar operation.",
                "Difficulty tags and prerequisite links help you choose a reading order.",
            ),
            how_buildml_uses=(
                "Session.explain(..., level=...) and Session.learn(..., level=...) select the tier.",
                "ConceptNote.difficulty places each concept on the foundation / core / advanced ladder.",
                "buildml.explain.glossary supplies plain meanings so jargon is defined in place.",
            ),
            interpretation_rules=(
                "level changes depth of rendering, never correctness.",
                "difficulty describes the concept; level describes the reader.",
                "A note's prerequisite_concepts are what to read first, not what to install.",
            ),
            assumptions=(
                "Examples assume that their input data and any named estimators are available.",
                "An explanation describes the API; it does not validate your dataset or modeling assumptions.",
            ),
            failure_modes=(
                "Requesting an unknown level string raises rather than silently defaulting.",
                "Reading only a short summary can miss assumptions needed for interpretation.",
            ),
            anti_patterns=(
                "Running an example without checking its data roles and prerequisite operations.",
                "Skipping assumptions because an operation has a simple interface.",
            ),
            worked_example_pattern=(
                "session.explain('split') -> read .beginner; session.learn('data-splitting').",
            ),
            related_concepts=(
                "operation-history",
                "reproducibility",
                "leakage-boundary",
            ),
        ),
    )
}
