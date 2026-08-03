# ruff: noqa: E501
"""Beginner layer for BuildML's own teaching surface."""

from __future__ import annotations

from buildml.explain.beginner._builder import FOUNDATION, BeginnerLayer, _index, _layer

TEACHING_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "explain-learning-levels",
        plain=(
            "BuildML explains itself in three depths at once. Ask for the beginner level and you get "
            "plain language, an analogy, the steps in order, and the words defined as they appear. Ask "
            "for advanced and you get the formal statement and the edge cases. It is the same "
            "explanation; you choose how much of it is shown."
        ),
        analogy=(
            "A good museum label. The big print tells you what you are looking at, the small print tells "
            "you the provenance, and the catalogue in the shop has the full scholarship. Nobody is lied "
            "to; people just start at different places."
        ),
        steps=(
            "Ask about anything: `session.explain('split')` for an operation, `session.learn('data-splitting')` for a concept.",
            "Pass `level='beginner'` (the default), `'intermediate'`, or `'advanced'`.",
            "At beginner level you also get an analogy, a glossary of the terms used, and a worked mini example.",
            "Every concept is tagged foundation, core, or advanced, and links to what to read first.",
            "`session.learn()` with no argument returns the foundation concepts — the sensible place to start.",
        ),
        use=(
            "Whenever you are about to run something you have not run before.",
            "Whenever a result arrives and you are not sure what it is telling you.",
            "When you want a reading order rather than an alphabetical list of topics.",
        ),
        avoid=(
            "Do not treat the beginner level as a simplified or approximate answer; it is the same material with the vocabulary supplied.",
            "Do not use explanations as a substitute for checking your own data; they describe BuildML, not your dataset.",
        ),
        myths=(
            (
                "The beginner level leaves out the hard truths.",
                "It leads with the plain reading and still names the leakage risks, the failure modes, and the misconceptions. Nothing correct is withheld.",
            ),
            (
                "Explanations are written by hand for every operation, so some must be out of date.",
                "Operation primers are derived from the same catalog that defines the parameters and prerequisites. They cannot describe a signature that no longer exists.",
            ),
        ),
        example=(
            "brief = session.explain('split')",
            "print(brief.beginner.plain_summary)",
            "print(brief.beginner.analogy)",
            "for term in brief.beginner.glossary: print(term.term, '-', term.plain_meaning)",
            "",
            "session.learn()                       # foundation concepts, in reading order",
            "session.learn('leakage-boundary')     # one concept, all three layers",
            "session.learn('fit', level='advanced')  # an operation, expert tier",
        ),
        check=(
            "Can you say, in your own words, what the operation you are about to run will change?",
            "Which words in the explanation would you struggle to define? Those are the ones in the glossary.",
        ),
        tools=("explain", "learn", "workflow", "walkthrough"),
        terms=("operation", "Session", "prerequisite", "leakage"),
        difficulty=FOUNDATION,
    ),
)

__all__ = ["TEACHING_BEGINNER"]
