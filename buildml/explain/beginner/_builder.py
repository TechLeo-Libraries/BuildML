# ruff: noqa: E501
"""Construction helpers for beginner teaching layers.

Every concept note in :mod:`buildml.explain.concepts` carries expert-grade
prose. That prose is correct but assumes vocabulary a newcomer does not have,
which defeats the point of an explain system. A *beginner layer* supplies the
missing tier for the same concept key: plain language, an analogy, a
step-by-step walkthrough, when to reach for it and when not to, the myths
people arrive with, and a runnable-shaped example.

Layers live beside the technical notes (one module per domain, mirroring
``concepts/``) and are merged onto the note in ``concepts/__init__``. Nothing is
duplicated: a layer holds only the fields the technical note lacks.
"""

from __future__ import annotations

from dataclasses import dataclass

from buildml.explain.schemas import ConceptDifficulty, Misconception

FOUNDATION = ConceptDifficulty.FOUNDATION
CORE = ConceptDifficulty.CORE
ADVANCED = ConceptDifficulty.ADVANCED


@dataclass(frozen=True, slots=True)
class BeginnerLayer:
    """Beginner-tier content for one concept key."""

    key: str
    plain_summary: str
    analogy: str
    steps: tuple[str, ...]
    when_to_use: tuple[str, ...]
    when_not_to_use: tuple[str, ...]
    misconceptions: tuple[Misconception, ...]
    mini_example: tuple[str, ...]
    check_yourself: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    glossary_terms: tuple[str, ...] = ()
    difficulty: ConceptDifficulty = ConceptDifficulty.CORE


def _layer(
    key: str,
    *,
    plain: str,
    analogy: str,
    steps: tuple[str, ...],
    use: tuple[str, ...],
    avoid: tuple[str, ...],
    myths: tuple[tuple[str, str], ...],
    example: tuple[str, ...],
    check: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
    terms: tuple[str, ...] = (),
    difficulty: ConceptDifficulty = ConceptDifficulty.CORE,
) -> BeginnerLayer:
    """Build one beginner layer, validating the minimum teaching contract."""
    if len(plain.split()) < 12:
        raise ValueError(f"{key}: plain_summary must be a real explanation, not a label")
    if not analogy.strip():
        raise ValueError(f"{key}: analogy is required")
    for name, group, minimum in (
        ("steps", steps, 3),
        ("use", use, 2),
        ("avoid", avoid, 2),
        ("myths", myths, 1),
        ("example", example, 2),
    ):
        if len(group) < minimum:
            raise ValueError(f"{key}: {name} needs at least {minimum} entries")
    return BeginnerLayer(
        key=key,
        plain_summary=plain.strip(),
        analogy=analogy.strip(),
        steps=steps,
        when_to_use=use,
        when_not_to_use=avoid,
        misconceptions=tuple(Misconception(myth=myth, reality=reality) for myth, reality in myths),
        mini_example=example,
        check_yourself=check,
        tools=tools,
        glossary_terms=terms,
        difficulty=difficulty,
    )


def _index(*layers: BeginnerLayer) -> dict[str, BeginnerLayer]:
    """Key layers by concept key, rejecting accidental duplicates."""
    index: dict[str, BeginnerLayer] = {}
    for layer in layers:
        if layer.key in index:
            raise ValueError(f"Duplicate beginner layer for concept {layer.key!r}")
        index[layer.key] = layer
    return index


__all__ = ["ADVANCED", "CORE", "FOUNDATION", "BeginnerLayer", "_index", "_layer"]
