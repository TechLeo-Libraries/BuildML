"""EDA → modeling Concept Academy curriculum (Industry redesign + full catalog).

The learning hub covers every ``CONCEPT_NOTES`` entry (~204) as a first-class
lesson, plus redesign readiness-path slugs that are not themselves catalog keys.
Handcrafted ``stage_*.py`` / ``stage_gaps.py`` lessons win on slug collision;
``note_binder`` fills the rest at the same pedagogical bar.
"""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._helpers import build_academy_context
from buildml.dashboard.academy_curriculum._stage_map import DOMAIN_STAGE
from buildml.dashboard.academy_curriculum._types import LessonSpec
from buildml.dashboard.academy_curriculum.note_binder import (
    catalog_concept_count,
    note_lessons,
)
from buildml.dashboard.academy_curriculum.stage_00 import lessons as stage_00
from buildml.dashboard.academy_curriculum.stage_01 import lessons as stage_01
from buildml.dashboard.academy_curriculum.stage_02 import lessons as stage_02
from buildml.dashboard.academy_curriculum.stage_03 import lessons as stage_03
from buildml.dashboard.academy_curriculum.stage_04 import lessons as stage_04
from buildml.dashboard.academy_curriculum.stage_05 import lessons as stage_05
from buildml.dashboard.academy_curriculum.stage_gaps import lessons as stage_gaps
from buildml.dashboard.gates import GATE_STAGES

CURRICULUM_STAGES: tuple[dict, ...] = tuple(dict(stage) for stage in GATE_STAGES)

# Backward-compatible name; stage 06 is domain depth with full lessons.
EXTENDED_STAGE = dict(DOMAIN_STAGE)


def _handcrafted_lessons() -> list[LessonSpec]:
    lessons = [
        *stage_00(),
        *stage_01(),
        *stage_02(),
        *stage_03(),
        *stage_04(),
        *stage_05(),
        *stage_gaps(),
    ]
    lessons.sort(key=lambda item: (item.stage, item.order, item.slug))
    return lessons


def all_lessons() -> list[LessonSpec]:
    """Union of handcrafted readiness lessons and every remaining CONCEPT_NOTES lesson."""
    handcrafted = _handcrafted_lessons()
    handcrafted_slugs = {lesson.slug for lesson in handcrafted}
    catalog = note_lessons(exclude_slugs=handcrafted_slugs)
    lessons = [*handcrafted, *catalog]
    by_slug: dict[str, LessonSpec] = {}
    for item in lessons:
        if item.slug not in by_slug:
            by_slug[item.slug] = item
    ordered = list(by_slug.values())
    ordered.sort(key=lambda item: (item.stage, item.order, item.slug))
    return ordered


def curriculum_by_slug() -> dict[str, LessonSpec]:
    return {lesson.slug: lesson for lesson in all_lessons()}


def curriculum_slugs() -> frozenset[str]:
    return frozenset(curriculum_by_slug())


def readiness_slugs() -> frozenset[str]:
    """Handcrafted redesign + gap slugs (stages 00–05 spine)."""
    return frozenset(lesson.slug for lesson in _handcrafted_lessons())


__all__ = [
    "CURRICULUM_STAGES",
    "DOMAIN_STAGE",
    "EXTENDED_STAGE",
    "LessonSpec",
    "all_lessons",
    "build_academy_context",
    "catalog_concept_count",
    "curriculum_by_slug",
    "curriculum_slugs",
    "readiness_slugs",
]
