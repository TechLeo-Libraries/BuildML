"""Compact lesson factory for dense curriculum stages."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from buildml.dashboard.academy_curriculum._helpers import (
    code_block,
    first_feature,
    fmt_n,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec, lesson

Ctx = dict[str, Any]
Fn = Callable[[Ctx], Any]


def L(
    *,
    slug: str,
    stage: int,
    order: int,
    concept_key: str | None,
    plain: tuple[str, ...] | list[str],
    technical: tuple[str, ...] | list[str],
    why: tuple[str, ...] | list[str],
    formula: str | None,
    calculation: Fn | str,
    session_evidence: Fn | str,
    example_code: Fn | str,
    what_to_change: tuple[str, ...] | list[str] | Fn,
    pitfalls: tuple[str, ...] | list[str] | Fn,
    decide: Fn | str,
    read_steps: tuple[str, ...] | list[str] | Fn,
    title: str | None = None,
    tags: tuple[str, ...] = (),
    search_terms: tuple[str, ...] = (),
) -> LessonSpec:
    def _text(value: Fn | str) -> Fn:
        if callable(value):
            return value
        return lambda _c, t=str(value): t

    def _code(value: Fn | str) -> Fn:
        if callable(value):
            return value
        return lambda _c, t=str(value): t

    return lesson(
        slug=slug,
        stage=stage,
        title=title or slug,
        order=order,
        concept_key=concept_key,
        tags=tags,
        search_terms=search_terms or (slug,),
        plain=plain,
        technical=technical,
        why=why,
        formula=formula,
        calculation=_text(calculation),
        session_evidence=_text(session_evidence),
        example_code=_code(example_code),
        what_to_change=what_to_change,
        pitfalls=pitfalls,
        decide=decide,
        read_steps=read_steps,
    )


def starter_session(ctx: Ctx, *, stratify: bool | None = None) -> list[str]:
    if stratify is None:
        target = ctx.get("target") or {}
        use_stratify = isinstance(target, dict) and target.get("task") == "classification"
    else:
        use_stratify = bool(stratify)
    split_args = "test_size=0.2, random_state=0"
    if use_stratify:
        split_args += ", stratify=True"
    return [
        "from buildml import Session",
        "import pandas as pd",
        "",
        "session = (",
        "    Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change path/columns",
        "    .set_roles({",
        f'        "{target_name(ctx)}": "target",  # <-- change',
        f'        "{first_feature(ctx)}": "feature",  # <-- add all predictors',
        "    })",
        f"    .split({split_args})",
        ")",
    ]


def with_starter(ctx: Ctx, *extra: str, stratify: bool | None = None) -> str:
    return code_block(*starter_session(ctx, stratify=stratify), "", *extra)


def collect(groups: Iterable[Iterable[LessonSpec]]) -> list[LessonSpec]:
    out: list[LessonSpec] = []
    for group in groups:
        out.extend(group)
    return out


def rows_blurb(ctx: Ctx) -> str:
    return f"{fmt_n(ctx.get('rows'))} rows x {fmt_n(ctx.get('colCount'))} columns"
