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
                "LearningLevel is a total order (beginner < intermediate < advanced) "
                "and renderers select content by level rank. Operation primers are "
                "derived from the catalog rather than authored, so the beginner tier "
                "cannot drift from the parameters, prerequisites, and concepts that "
                "are already maintained."
            ),
            why_it_matters=(
                "An explanation that assumes the vocabulary it is explaining is not an explanation.",
                "Derived primers guarantee coverage: no operation can ship without a plain-language layer.",
                "Difficulty tags and prerequisite links turn 187 notes into a reading order instead of an index.",
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
                "Every concept note has an authored beginner layer; the merge fails loudly otherwise.",
                "Operation primers are derived, so overlay prose stays the single source of truth.",
            ),
            failure_modes=(
                "Requesting an unknown level string raises rather than silently defaulting.",
                "A beginner layer whose concept key no longer exists fails at import.",
            ),
            anti_patterns=(
                "Writing a separate beginner document that will drift from the technical note.",
                "Treating 'advanced' as the real explanation and 'beginner' as marketing.",
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
