"""Bind every CONCEPT_NOTES entry into a first-class Academy LessonSpec.

Handcrafted stage_*.py lessons win when their slug matches a note key.
Everything else is generated here from the merged ConceptNote (beginner →
advanced) plus session-adaptive binders — never a one-line extended stub.
"""

from __future__ import annotations

from typing import Any

from buildml.dashboard.academy_curriculum._factory import starter_session, with_starter
from buildml.dashboard.academy_curriculum._helpers import (
    code_block,
    first_feature,
    first_missing,
    first_numeric,
    fmt_compact,
    fmt_n,
    fmt_pct,
    is_classification,
    is_regression,
    list_names,
    target_name,
)
from buildml.dashboard.academy_curriculum._stage_map import stage_for_concept
from buildml.dashboard.academy_curriculum._types import LessonSpec, lesson
from buildml.explain.concepts import CONCEPT_NOTES

Ctx = dict[str, Any]


def _plain(note: Any) -> tuple[str, ...]:
    parts = [
        note.plain_summary or note.summary,
        note.analogy,
        note.intuition,
    ]
    return tuple(p for p in parts if p)


def _technical(note: Any) -> tuple[str, ...]:
    parts = [
        note.definition,
        note.formal_idea,
        *(note.how_buildml_uses[:3] if note.how_buildml_uses else ()),
    ]
    return tuple(p for p in parts if p)


def _why(note: Any) -> tuple[str, ...]:
    why = tuple(note.why_it_matters[:5]) if note.why_it_matters else ()
    if why:
        return why
    if note.summary:
        return (note.summary,)
    return ("This concept shapes how you read BuildML diagnostics and Session ops.",)


def _formula(note: Any) -> str | None:
    formal = (note.formal_idea or "").strip()
    if not formal:
        return None
    # Keep formulas readable in the calc block; full prose stays in walkthrough.
    if len(formal) <= 220:
        return formal
    return formal[:217].rstrip() + "…"


def _misconception_lines(note: Any) -> list[str]:
    lines: list[str] = []
    for item in note.misconceptions or ():
        wrong = getattr(item, "wrong", None) or (item.get("wrong") if isinstance(item, dict) else None)
        right = getattr(item, "right", None) or (item.get("right") if isinstance(item, dict) else None)
        if wrong and right:
            lines.append(f"Myth: {wrong} → Actually: {right}")
        elif wrong:
            lines.append(f"Myth: {wrong}")
    return lines


def _pitfalls(note: Any) -> list[str]:
    out: list[str] = []
    out.extend(str(x) for x in (note.failure_modes or ())[:4])
    out.extend(str(x) for x in (note.anti_patterns or ())[:3])
    out.extend(_misconception_lines(note)[:2])
    # Deduplicate preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for line in out:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique or [
        "Skipping the beginner brief and jumping to metrics usually hides the real decision.",
        "Treating a reference concept as live evidence when this session never triggered it.",
    ]


def _read_steps(note: Any) -> list[str]:
    steps = [str(s) for s in (note.beginner_steps or ())[:6]]
    if not steps:
        steps = [str(s) for s in (note.interpretation_rules or ())[:6]]
    if not steps:
        steps = [
            "Read the plain-language summary.",
            "Open Calculation and bind the numbers to this session.",
            "Run the worked Session example on your frame.",
            "Check pitfalls before you trust a metric.",
        ]
    return steps


def _decide(note: Any) -> str:
    if note.when_to_use:
        return str(note.when_to_use[0])
    if note.why_it_matters:
        return str(note.why_it_matters[0])
    return f"Decide whether '{note.title}' changes a role, split, transform, or metric on this session."


def _applies_hint(key: str, ctx: Ctx) -> str:
    """Honest applicability line — N/A when the live frame cannot speak to it."""
    task = str(ctx.get("task") or "")
    has_target = bool(ctx.get("has_target"))
    analyzers = ctx.get("analyzers") or {}

    if key in {"class-imbalance", "probability-calibration", "thresholds"} or key.startswith(
        "decision-"
    ):
        if is_regression(ctx):
            return "N/A for this regression target — classification / decision thresholds do not apply."
        if not has_target:
            return "N/A until a classification target role is declared."
        if is_classification(ctx):
            return f"Applies: classification target '{target_name(ctx)}'."

    if key in {"mutual-information", "variance-inflation", "feature-selection"}:
        if not has_target and key == "mutual-information":
            return "N/A for MI-vs-target until a target role is set."
        if key == "variance-inflation" and not analyzers.get("vif"):
            return "VIF analyzer unavailable or skipped on this report — treat as reference teaching."
        if key == "mutual-information" and not analyzers.get("mi"):
            return "MI analyzer unavailable or skipped — treat as reference teaching."

    if key in {"dataset-drift"} and not analyzers.get("drift"):
        return "Drift screen not present on this report — reference teaching until partitions exist."

    if key.startswith("nlp-") or key == "text-features":
        cats = ctx.get("categorical") or []
        # Heuristic: no strong text signal in overview — still teach, mark honesty.
        return (
            f"Session has {fmt_n(len(cats))} categorical/text-like columns to audit for text workflows; "
            "if your corpus lives outside this table, treat this as domain reference."
        )

    if key.startswith(("rl-", "imitation-", "graph-", "kg-", "tda-", "federated-", "rag-")):
        return (
            "Domain concept — this tabular EDA session may not exercise it directly; "
            "the lesson still teaches the BuildML contract so you can apply it when that surface is in play."
        )

    if key.startswith("forecast-") or key.startswith("ts-"):
        return (
            "Time-series / forecast concept — apply when a time role or ordered index exists; "
            "otherwise keep as reference for temporal projects."
        )

    if not has_target and any(
        token in key for token in ("target", "supervised", "calibr", "baseline", "metric")
    ):
        return "No target declared — supervised reading is reference until roles are set."

    return ""


def _session_evidence(key: str, note: Any, ctx: Ctx) -> str:
    findings = (ctx.get("findings_by_slug") or {}).get(key) or []
    apply = _applies_hint(key, ctx)
    scope = (
        f"{fmt_n(ctx.get('rows'))} rows × {fmt_n(ctx.get('colCount'))} columns · "
        f"task={ctx.get('task') or 'undeclared'} · "
        f"target={target_name(ctx) if ctx.get('has_target') else 'none'} · "
        f"{fmt_n(ctx.get('eligible'))} eligible features"
    )
    if findings:
        top = findings[0]
        cite = f"Cited here: {top.get('severity') or ''} · {top.get('title') or key}".strip()
        extra = f" (+{len(findings) - 1} more)" if len(findings) > 1 else ""
        base = f"{cite}{extra}. {scope}."
    else:
        base = (
            f"Reference on this report (no finding keyed to '{key}'). {scope}."
        )
    if apply:
        return f"{base} {apply}"
    # Note-specific live numbers when analyzers speak.
    if key == "missing-data":
        return (
            f"{base} Completeness≈{fmt_pct(ctx.get('completeness') or 0)} · "
            f"missing cells={fmt_n(ctx.get('missingCells'))} · "
            f"example column={first_missing(ctx)}."
        )
    if key == "class-imbalance" and is_classification(ctx):
        target = ctx.get("target") or {}
        return f"{base} Classification on '{target.get('name')}'."
    if key == "mutual-information" and ctx.get("mi"):
        top_mi = ctx["mi"][0]
        return (
            f"{base} Top MI: {top_mi.get('name')}={fmt_compact(top_mi.get('mi'))}."
        )
    if key == "variance-inflation" and ctx.get("vif"):
        return f"{base} VIF rows available: {fmt_n(len(ctx.get('vif') or []))}."
    if note.when_not_to_use:
        return f"{base} When not to lean on it: {note.when_not_to_use[0]}"
    return base


def _calculation(key: str, note: Any, ctx: Ctx) -> str:
    apply = _applies_hint(key, ctx)
    rows = int(ctx.get("rows") or 0)
    cols = int(ctx.get("colCount") or 0)
    lines: list[str] = []

    if key == "missing-data":
        cells = max(rows * max(cols, 1), 1)
        missing = int(ctx.get("missingCells") or 0)
        comp = ctx.get("completeness")
        lines.append(
            f"completeness = 1 − missing_cells/(rows×cols) = 1 − {fmt_n(missing)}/{fmt_n(cells)} "
            f"≈ {fmt_pct(comp if comp is not None else 1 - missing / cells)}"
        )
        lines.append(f"Inspect column '{first_missing(ctx)}' before choosing an impute strategy.")
    elif key == "class-imbalance":
        if is_classification(ctx):
            lines.append(
                f"For target '{target_name(ctx)}', compare per-class support and "
                "minority/majority ratio before picking a metric or resample."
            )
        else:
            lines.append(apply or "Class imbalance math does not apply to this task.")
    elif key == "data-splitting" or key == "evaluation-partitions":
        n = rows or 0
        lines.append(
            f"With n={fmt_n(n)}, a 60/20/20 split ≈ "
            f"{fmt_n(int(0.6 * n))}/{fmt_n(int(0.2 * n))}/{fmt_n(int(0.2 * n))} rows "
            "(adjust when groups or time forbid random cuts)."
        )
    elif key == "mutual-information":
        if ctx.get("mi"):
            top = ", ".join(
                f"{r['name']}={fmt_compact(r['mi'])}" for r in ctx["mi"][:3]
            )
            lines.append(f"MI vs target (top): {top}.")
        else:
            lines.append(apply or "MI vs target not computed on this report.")
    elif key == "variance-inflation":
        vif = ctx.get("vif") or []
        if vif:
            names = list_names(vif, limit=4)
            lines.append(
                f"VIF table present for: {names}. Flag columns with VIF much greater than 5-10."
            )
        else:
            lines.append(apply or "VIF not available — need complete numeric cases.")
    elif key == "cross-validation":
        lines.append(
            f"With n={fmt_n(rows)}, k-fold uses ≈{fmt_n(max(rows // 5, 1))} rows per fold at k=5; "
            "group/time CV replaces random folds when rows are dependent."
        )
    elif key.startswith("probabilistic-") or "calibr" in key:
        holdout = fmt_n(int(0.2 * rows)) if rows else "—"
        lines.append(
            "Calibration compares predicted probabilities to observed frequencies "
            f"on held-out rows (n≈{holdout} if test_size=0.2)."
        )
    else:
        # Lift formal idea + session scope — still a real walkthrough, not a stub.
        if note.formal_idea:
            lines.append(note.formal_idea)
        lines.append(
            f"On this session: n={fmt_n(rows)}, p≈{fmt_n(ctx.get('eligible') or cols)}, "
            f"task={ctx.get('task') or 'undeclared'}."
        )
        if apply:
            lines.append(apply)
        elif note.interpretation_rules:
            lines.append(str(note.interpretation_rules[0]))

    return "\n".join(lines)


def _tool_calls(note: Any, key: str) -> list[str]:
    tools = [str(t) for t in (note.buildml_tools or ()) if t][:4]
    lines: list[str] = []
    for tool in tools:
        name = tool.strip()
        if not name:
            continue
        if name.startswith("session."):
            lines.append(f"{name}  # from concept tools")
        elif "(" in name:
            lines.append(name)
        else:
            # Catalog-style short names → session method guess.
            safe = name.replace("session.", "")
            lines.append(f"session.{safe}()  # <-- confirm signature for your build")
    if not lines:
        lines = [
            f'brief = session.learn("{key}", level="beginner")',
            "print(brief.concept.plain_summary if brief.concept else brief.suggested)",
        ]
    else:
        lines.insert(0, f'brief = session.learn("{key}", level="beginner")')
        lines.insert(1, "print(brief.concept.summary if brief.concept else brief.suggested)")
    return lines


def _example_code(key: str, note: Any, ctx: Ctx) -> str:
    header = [
        "# Worked BuildML Session example — change paths/columns to your data.",
        *[f"# Pattern: {step}" for step in (note.worked_example_pattern or ())[:3]],
    ]
    mini = [str(line) for line in (note.mini_example or ())[:6]]
    tools = _tool_calls(note, key)
    # Prefer mini_example when it already looks like Session code.
    if any("session" in line.lower() or "Session" in line for line in mini):
        body = [
            "from buildml import Session",
            "import pandas as pd",
            "",
            'frame = pd.read_csv("your_data.csv")  # <-- change',
            "session = Session.ingest(frame)",
            "session = session.set_roles({",
            f'    "{target_name(ctx)}": "target",  # <-- change',
            f'    "{first_feature(ctx)}": "feature",  # <-- add predictors',
            "})",
            "session = session.split(test_size=0.2, random_state=0"
            + (", stratify=True" if is_classification(ctx) else "")
            + ")",
            "",
            *mini,
            "",
            *tools[:3],
        ]
        return code_block(*header, "", *body)

    extras = [
        *tools,
        "",
        f'session.explain("{key}", moment="before")  # when an operation maps here',
    ]
    if key == "missing-data":
        extras = [
            f'brief = session.learn("{key}", level="beginner")',
            f'session = session.impute(columns=["{first_missing(ctx)}"], strategy="median")  # <-- change',
        ]
    elif key in {"data-splitting", "leakage-boundary", "evaluation-partitions"}:
        extras = [
            f'brief = session.learn("{key}", level="beginner")',
            "# Re-split deliberately if partitions were wrong:",
            "session = session.split(test_size=0.2, random_state=0"
            + (", stratify=True" if is_classification(ctx) else "")
            + ")",
        ]
    elif key == "feature-scaling":
        extras = [
            f'brief = session.learn("{key}", level="beginner")',
            f'session = session.scale(columns=["{first_numeric(ctx)}"], method="standard")  # <-- change',
        ]
    return with_starter(ctx, *extras) if not header else code_block(
        *header,
        "",
        *starter_session(ctx),
        "",
        *extras,
    )


def _what_to_change(note: Any, ctx: Ctx) -> list[str]:
    items = [
        'Point Session.ingest(...) at your DataFrame / CSV (replace "your_data.csv").',
        f'Update set_roles target/feature names (live hint: target="{target_name(ctx)}", feature="{first_feature(ctx)}").',
        "Raise learn(level=...) to 'intermediate' or 'advanced' after the beginner pass.",
    ]
    for line in (note.when_to_use or ())[:2]:
        items.append(f"Use when: {line}")
    for line in (note.when_not_to_use or ())[:2]:
        items.append(f"Avoid when: {line}")
    if ctx.get("idLike"):
        items.append(f"Audit id-like columns before fitting: {list_names(ctx.get('idLike') or [])}.")
    return items


def lesson_from_note(key: str) -> LessonSpec:
    """Build one full Academy lesson from a CONCEPT_NOTES key."""
    note = CONCEPT_NOTES[key]
    stage = stage_for_concept(key)
    tags = (
        "catalog",
        note.difficulty.value if note.difficulty else "core",
        f"stage-{stage:02d}",
    )
    search = (
        key,
        note.title,
        *(note.related_concepts or ())[:8],
        *(note.buildml_tools or ())[:6],
    )

    return lesson(
        slug=key,
        stage=stage,
        title=note.title or key,
        order=1000 + stage * 100,  # after handcrafted readiness ordering
        concept_key=key,
        tags=tags,
        search_terms=search,
        plain=_plain(note),
        technical=_technical(note),
        why=_why(note),
        formula=_formula(note),
        calculation=lambda ctx, k=key, n=note: _calculation(k, n, ctx),
        session_evidence=lambda ctx, k=key, n=note: _session_evidence(k, n, ctx),
        example_code=lambda ctx, k=key, n=note: _example_code(k, n, ctx),
        what_to_change=lambda ctx, n=note: _what_to_change(n, ctx),
        pitfalls=lambda _ctx, n=note: _pitfalls(n),
        decide=lambda _ctx, n=note: _decide(n),
        read_steps=lambda _ctx, n=note: _read_steps(n),
    )


def note_lessons(*, exclude_slugs: set[str] | frozenset[str] | None = None) -> list[LessonSpec]:
    """Lessons for every CONCEPT_NOTES key not already authored under exclude_slugs."""
    skip = set(exclude_slugs or ())
    out = [lesson_from_note(key) for key in sorted(CONCEPT_NOTES) if key not in skip]
    out.sort(key=lambda item: (item.stage, item.order, item.slug))
    return out


def catalog_concept_count() -> int:
    return len(CONCEPT_NOTES)


__all__ = [
    "catalog_concept_count",
    "lesson_from_note",
    "note_lessons",
]
