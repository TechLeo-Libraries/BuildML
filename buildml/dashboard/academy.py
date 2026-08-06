"""Concept Academy learning-hub payload for the Industry EDA App.

Builds a staged, dataset-adaptive curriculum covering every BuildML
``CONCEPT_NOTES`` entry (~204) plus readiness-path curriculum slugs that are
not themselves catalog keys. Each lesson has plain-language teaching,
calculations, real BuildML Session examples, and live-report evidence.
There is no thin "extended" dump.
"""

from __future__ import annotations

from typing import Any

from buildml.dashboard.academy_curriculum import (
    CURRICULUM_STAGES,
    DOMAIN_STAGE,
    all_lessons,
    build_academy_context,
    catalog_concept_count,
    curriculum_by_slug,
    readiness_slugs,
)
from buildml.dashboard.gates import (
    CONCEPT_ALIASES,
    FINDING_CONCEPT_SLUG,
    resolve_concept_key,
)
from buildml.explain.concepts import CONCEPT_NOTES


def academy_stages() -> list[dict[str, Any]]:
    return [dict(stage) for stage in CURRICULUM_STAGES] + [dict(DOMAIN_STAGE)]


def concept_stage(key: str) -> int:
    """Stage for a curriculum slug or CONCEPT_NOTES key."""
    lesson = curriculum_by_slug().get(key)
    if lesson is not None:
        return lesson.stage
    for slug, alias in CONCEPT_ALIASES.items():
        if alias == key:
            linked = curriculum_by_slug().get(slug)
            if linked is not None:
                return linked.stage
    from buildml.dashboard.academy_curriculum._stage_map import stage_for_concept

    if key in CONCEPT_NOTES:
        return stage_for_concept(key)
    return 6


def cited_concept_keys(report: dict[str, Any]) -> dict[str, int]:
    """Count findings that cite each BuildML concept key."""
    counts: dict[str, int] = {}
    for item in report.get("findings") or []:
        slug = FINDING_CONCEPT_SLUG.get(str(item.get("key", "")))
        if not slug:
            continue
        concept_key = resolve_concept_key(slug) or CONCEPT_ALIASES.get(slug)
        if not concept_key:
            continue
        counts[concept_key] = counts.get(concept_key, 0) + 1
    return counts


def cited_curriculum_slugs(report: dict[str, Any]) -> dict[str, int]:
    """Count findings that cite each curriculum slug (readiness-path keys)."""
    counts: dict[str, int] = {}
    for item in report.get("findings") or []:
        slug = FINDING_CONCEPT_SLUG.get(str(item.get("key", "")))
        if not slug:
            continue
        counts[slug] = counts.get(slug, 0) + 1
        # Also credit the aliased catalog key so note lessons light up.
        alias = CONCEPT_ALIASES.get(slug) or resolve_concept_key(slug)
        if alias and alias != slug:
            counts[alias] = counts.get(alias, 0) + 1
    return counts


def _safe_call(fn: Any, ctx: dict[str, Any], default: str = "") -> str:
    try:
        value = fn(ctx)
    except Exception:  # noqa: BLE001 — adaptive binders must not break the board
        return default
    return str(value) if value is not None else default


def _safe_list(fn: Any, ctx: dict[str, Any]) -> list[str]:
    try:
        value = fn(ctx)
    except Exception:  # noqa: BLE001
        return []
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def _note_fields(concept_key: str | None) -> dict[str, Any]:
    if not concept_key or concept_key not in CONCEPT_NOTES:
        return {
            "definition": "",
            "intuition": "",
            "formal_idea": "",
            "why_it_matters": [],
            "how_buildml_uses": [],
            "interpretation_rules": [],
            "assumptions": [],
            "failure_modes": [],
            "anti_patterns": [],
            "worked_example_pattern": [],
            "related_concepts": [],
            "details": [],
            "summary": "",
        }
    note = CONCEPT_NOTES[concept_key]
    return {
        "definition": note.definition,
        "intuition": note.intuition,
        "formal_idea": note.formal_idea,
        "why_it_matters": list(note.why_it_matters),
        "how_buildml_uses": list(note.how_buildml_uses),
        "interpretation_rules": list(note.interpretation_rules),
        "assumptions": list(note.assumptions),
        "failure_modes": list(note.failure_modes),
        "anti_patterns": list(note.anti_patterns),
        "worked_example_pattern": list(note.worked_example_pattern),
        "related_concepts": list(note.related_concepts),
        "details": list(note.details),
        "summary": note.summary,
    }


def _finding_chips(ctx: dict[str, Any], slug: str) -> list[dict[str, Any]]:
    rows = (ctx.get("findings_by_slug") or {}).get(slug) or []
    chips = []
    for item in rows[:6]:
        chips.append(
            {
                "key": item.get("key"),
                "label": item.get("title") or item.get("key"),
                "severity": item.get("severity"),
            }
        )
    return chips


def _build_curriculum_entry(
    lesson: Any,
    ctx: dict[str, Any],
    cite_counts: dict[str, int],
    readiness: frozenset[str],
) -> dict[str, Any]:
    cite_count = int(cite_counts.get(lesson.slug, 0))
    if cite_count == 0 and lesson.concept_key:
        cite_count = int(cite_counts.get(lesson.concept_key, 0))
    concept_key = lesson.concept_key
    if concept_key is None:
        concept_key = resolve_concept_key(lesson.slug)
    note = _note_fields(concept_key)

    plain = list(lesson.plain)
    technical = list(lesson.technical)
    prose = [*plain, *technical]

    session_line = _safe_call(lesson.session_evidence, ctx)
    calculation = _safe_call(lesson.calculation, ctx)
    example = _safe_call(lesson.example_code, ctx)
    decide = _safe_call(lesson.decide, ctx)
    read = _safe_list(lesson.read_steps, ctx)
    pitfalls = _safe_list(lesson.pitfalls, ctx)
    what_to_change = _safe_list(lesson.what_to_change, ctx)
    why = list(lesson.why) or list(note["why_it_matters"][:4])

    if not session_line:
        session_line = (
            f"Cited by {cite_count} finding(s) on this session's report."
            if cite_count
            else "Reference teaching for this stage — not triggered by a finding here."
        )

    is_readiness = lesson.slug in readiness
    is_catalog = bool(concept_key and concept_key in CONCEPT_NOTES)
    tags = list(lesson.tags or ())
    if is_readiness and "readiness" not in tags:
        tags.append("readiness")
    if is_catalog and "catalog" not in tags:
        tags.append("catalog")

    search_bits = [
        lesson.slug,
        lesson.title,
        *plain,
        *technical,
        *why,
        session_line,
        calculation,
        example,
        decide,
        *read,
        *pitfalls,
        *what_to_change,
        *(lesson.search_terms or ()),
        *tags,
        note["summary"],
        note["definition"],
    ]

    return {
        "key": lesson.slug,
        "slug": lesson.slug,
        "title": lesson.title,
        "curriculum": True,
        "readiness_path": is_readiness,
        "catalog": is_catalog,
        "stage": lesson.stage,
        "cited": cite_count > 0,
        "uncited": cite_count == 0,
        "cite_count": cite_count,
        "concept_key": concept_key,
        "sections": {
            "what_it_means": plain,
            "technical_depth": technical,
            "why_it_matters": why,
            "calculation": {
                "formula": lesson.formula,
                "walkthrough": calculation,
            },
            "worked_example": {
                "language": "python",
                "code": example,
                "what_to_change": what_to_change,
            },
            "pitfalls": pitfalls,
            "evidence": {
                "session": session_line,
                "findings": _finding_chips(ctx, lesson.slug),
            },
            "how_to_read": read,
            "decide": decide,
        },
        "summary": plain[0] if plain else note["summary"],
        "prose": prose,
        "session": session_line,
        "decide": decide,
        "has_decide": bool(decide),
        "read": read,
        "example": example,
        "pitfalls": pitfalls,
        "formula": lesson.formula,
        "calculation": calculation,
        "what_to_change": what_to_change,
        "why": why,
        "tags": tags,
        "search": " ".join(str(bit) for bit in search_bits if bit),
        **note,
        "details": list(note["details"]),
        "related_concepts": list(note["related_concepts"]),
    }


def build_academy_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Academy index with stages, adaptive sections, and cited/reference flags."""
    ctx = build_academy_context(report)
    cite_slugs = cited_curriculum_slugs(report)
    lessons = all_lessons()
    readiness = readiness_slugs()

    concepts: list[dict[str, Any]] = [
        _build_curriculum_entry(lesson, ctx, cite_slugs, readiness) for lesson in lessons
    ]
    concepts.sort(
        key=lambda item: (
            item["stage"],
            0 if item.get("readiness_path") else 1,
            item["title"].lower(),
        )
    )

    stages = academy_stages()
    stage_rows = []
    for stage in stages:
        entries = [c for c in concepts if c["stage"] == stage["key"]]
        cited_n = sum(1 for c in entries if c["cited"])
        stage_rows.append(
            {
                **stage,
                "entries": entries,
                "count_label": f"{cited_n} cited · {len(entries) - cited_n} reference · {len(entries)} lessons",
                "chips": [
                    {
                        "key": c["key"],
                        "slug": c["slug"],
                        "cited": c["cited"],
                        "count": c["cite_count"],
                        "curriculum": True,
                        "readiness_path": bool(c.get("readiness_path")),
                        "catalog": bool(c.get("catalog")),
                    }
                    for c in entries
                ],
            }
        )

    by_slug = {c["slug"]: c for c in concepts}
    catalog_covered = sum(1 for key in CONCEPT_NOTES if key in by_slug)
    target = ctx.get("target") if isinstance(ctx.get("target"), dict) else None
    catalog_n = catalog_concept_count()
    readiness_n = sum(1 for c in concepts if c.get("readiness_path"))
    return {
        "concepts": concepts,
        "stages": stage_rows,
        "cited_count": sum(1 for c in concepts if c["cited"]),
        "concept_count": len(concepts),
        "curriculum_count": len(concepts),
        "catalog_count": catalog_n,
        "catalog_covered": catalog_covered,
        "readiness_count": readiness_n,
        "extended_count": 0,
        "curriculum_note": (
            f"All {len(concepts)} lessons are first-class teaching "
            f"({catalog_n} BuildML CONCEPT_NOTES + readiness-path slugs). "
            "Filled chips are cited by a finding on this session's report; "
            "outlined chips are reference teaching. Stages 00-05 are the ML "
            "readiness spine; 06 is domain depth with the same pedagogical bar."
        ),
        "adaptivity": {
            "task": ctx.get("task"),
            "target": target.get("name") if target else None,
            "rows": ctx.get("rows"),
            "columns": ctx.get("colCount"),
            "eligible_features": ctx.get("eligible"),
            "has_mi": bool(ctx.get("analyzers", {}).get("mi")),
            "has_vif": bool(ctx.get("analyzers", {}).get("vif")),
            "sampled": bool(ctx.get("sampled")),
        },
        "context": {
            "rows": ctx.get("rows"),
            "colCount": ctx.get("colCount"),
            "task": ctx.get("task"),
            "target": target,
            "completeness": ctx.get("completeness"),
            "missingCells": ctx.get("missingCells"),
            "eligible": ctx.get("eligible"),
        },
    }


# Alias kept for older imports / docs.
EXTENDED_STAGE = DOMAIN_STAGE
