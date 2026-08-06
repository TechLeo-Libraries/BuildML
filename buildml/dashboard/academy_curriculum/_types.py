"""Curriculum lesson types for the Concept Academy learning hub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

Ctx = dict[str, Any]
TextFn = Callable[[Ctx], str]
ListFn = Callable[[Ctx], list[str]]


@dataclass(frozen=True, slots=True)
class LessonSpec:
    """One Academy curriculum entry (EDA -> modeling readiness path)."""

    slug: str
    stage: int
    title: str
    plain: tuple[str, ...]
    technical: tuple[str, ...]
    why: tuple[str, ...]
    formula: str | None
    calculation: TextFn
    session_evidence: TextFn
    example_code: TextFn
    what_to_change: ListFn
    pitfalls: ListFn
    decide: TextFn
    read_steps: ListFn
    concept_key: str | None = None
    search_terms: tuple[str, ...] = ()
    order: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)


def lesson(
    *,
    slug: str,
    stage: int,
    title: str,
    plain: tuple[str, ...] | list[str],
    technical: tuple[str, ...] | list[str],
    why: tuple[str, ...] | list[str],
    formula: str | None,
    calculation: TextFn,
    session_evidence: TextFn,
    example_code: TextFn,
    what_to_change: ListFn | tuple[str, ...] | list[str],
    pitfalls: ListFn | tuple[str, ...] | list[str],
    decide: TextFn | str,
    read_steps: ListFn | tuple[str, ...] | list[str],
    concept_key: str | None = None,
    search_terms: tuple[str, ...] = (),
    order: int = 0,
    tags: tuple[str, ...] = (),
) -> LessonSpec:
    """Build a lesson with static or adaptive callables."""

    def _as_text(value: TextFn | str) -> TextFn:
        if callable(value):
            return value
        return lambda _ctx, text=str(value): text

    def _as_list(value: ListFn | tuple[str, ...] | list[str]) -> ListFn:
        if callable(value):
            return value
        items = tuple(str(x) for x in value)
        return lambda _ctx, fixed=items: list(fixed)

    return LessonSpec(
        slug=slug,
        stage=stage,
        title=title,
        plain=tuple(plain),
        technical=tuple(technical),
        why=tuple(why),
        formula=formula,
        calculation=calculation,
        session_evidence=session_evidence,
        example_code=example_code,
        what_to_change=_as_list(what_to_change),
        pitfalls=_as_list(pitfalls),
        decide=_as_text(decide),
        read_steps=_as_list(read_steps),
        concept_key=concept_key,
        search_terms=search_terms,
        order=order,
        tags=tags,
    )
