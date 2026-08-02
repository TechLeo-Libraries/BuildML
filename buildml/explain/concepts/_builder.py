# ruff: noqa: E501
"""ConceptNote construction helpers."""

from __future__ import annotations

from buildml.explain.schemas import ConceptNote


def _flatten_details(
    *,
    definition: str,
    intuition: str,
    formal_idea: str,
    why_it_matters: tuple[str, ...],
    how_buildml_uses: tuple[str, ...],
    interpretation_rules: tuple[str, ...],
    assumptions: tuple[str, ...],
    failure_modes: tuple[str, ...],
    anti_patterns: tuple[str, ...],
    worked_example_pattern: tuple[str, ...],
) -> tuple[str, ...]:
    """Build a searchable flat paragraph list from structured teaching sections."""
    parts: list[str] = []
    for paragraph in (definition, intuition, formal_idea):
        text = paragraph.strip()
        if text:
            parts.append(text)
    for group in (
        why_it_matters,
        how_buildml_uses,
        interpretation_rules,
        assumptions,
        failure_modes,
        anti_patterns,
        worked_example_pattern,
    ):
        for item in group:
            text = item.strip()
            if text:
                parts.append(text)
    return tuple(parts)


def _note(
    *,
    key: str,
    title: str,
    summary: str,
    definition: str,
    intuition: str,
    formal_idea: str,
    why_it_matters: tuple[str, ...],
    how_buildml_uses: tuple[str, ...],
    interpretation_rules: tuple[str, ...],
    assumptions: tuple[str, ...],
    failure_modes: tuple[str, ...],
    anti_patterns: tuple[str, ...],
    worked_example_pattern: tuple[str, ...],
    related_concepts: tuple[str, ...] = (),
    references: tuple[str, ...] = (),
    details: tuple[str, ...] | None = None,
) -> ConceptNote:
    """Construct a ConceptNote and auto-build searchable ``details`` when omitted."""
    flat = details if details is not None else _flatten_details(
        definition=definition,
        intuition=intuition,
        formal_idea=formal_idea,
        why_it_matters=why_it_matters,
        how_buildml_uses=how_buildml_uses,
        interpretation_rules=interpretation_rules,
        assumptions=assumptions,
        failure_modes=failure_modes,
        anti_patterns=anti_patterns,
        worked_example_pattern=worked_example_pattern,
    )
    return ConceptNote(
        key=key,
        title=title,
        summary=summary,
        details=flat,
        related_concepts=related_concepts,
        references=references,
        definition=definition,
        intuition=intuition,
        formal_idea=formal_idea,
        why_it_matters=why_it_matters,
        how_buildml_uses=how_buildml_uses,
        interpretation_rules=interpretation_rules,
        assumptions=assumptions,
        failure_modes=failure_modes,
        anti_patterns=anti_patterns,
        worked_example_pattern=worked_example_pattern,
    )

