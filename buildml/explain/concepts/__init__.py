# ruff: noqa: E501
"""Shared concept notes referenced by the operation catalog and Concept Academy.

Notes are split by domain modules (classical / dl / rag / ai) and merged here.
"""

from __future__ import annotations

from buildml.explain.concepts.ai import AI_NOTES
from buildml.explain.concepts.classical import CLASSICAL_NOTES
from buildml.explain.concepts.dl import DL_NOTES
from buildml.explain.concepts.rag import RAG_NOTES
from buildml.explain.schemas import ConceptNote

CONCEPT_NOTES: dict[str, ConceptNote] = {
    **CLASSICAL_NOTES,
    **DL_NOTES,
    **RAG_NOTES,
    **AI_NOTES,
}


def get_concept(key: str) -> ConceptNote:
    """Return a concept note or raise a precise catalog error."""
    try:
        return CONCEPT_NOTES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown BuildML concept: {key}") from exc


def list_concepts() -> tuple[ConceptNote, ...]:
    """Return concept notes in stable key order."""
    return tuple(CONCEPT_NOTES[key] for key in sorted(CONCEPT_NOTES))
