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
            "you the provenance, and the catalogue in the shop has the full scholarship. Readers can choose the detail they need."
        ),
        steps=(
            "Ask about anything: `session.explain('split')` for an operation, `session.learn('data-splitting')` for a concept.",
            "Pass `level='beginner'` (the default), `'intermediate'`, or `'advanced'`.",
            "At beginner level you also get an analogy, a glossary of the terms used, and a worked mini example.",
            "Every concept is tagged foundation, core, or advanced, and links to what to read first.",
            "`session.learn()` with no argument returns the foundation concepts: the sensible place to start.",
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
                "The beginner level omits assumptions and limitations.",
                "It leads with the plain reading and still names the leakage risks, the failure modes, and the misconceptions. Advanced notes provide additional technical detail.",
            ),
            (
                "Explanations are written by hand for every operation, so some must be out of date.",
                "Operation primers are derived from the same catalog that defines the parameters and prerequisites. Examples must still be checked against the installed version.",
            ),
        ),
        example=(
            'import pandas as pd',
            'from sklearn.datasets import make_classification',
            'from sklearn.tree import DecisionTreeClassifier',
            'from buildml import Session',
            'X, y = make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=42)',
            'frame = pd.DataFrame(X, columns=["age", "income", "spend", "visits"])',
            'frame["target"] = y',
            'session = Session.ingest(frame).set_roles({"target": "target"})',
            'session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)',
            'brief = session.explain("split")',
            'print(brief.beginner.plain_summary)',
            'foundation = session.learn()',
            'leakage = session.learn("leakage-boundary")',
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
