"""Formatting and context helpers for Academy adaptive binders."""

from __future__ import annotations

from typing import Any

from buildml.dashboard.gates import (
    CONCEPT_ALIASES,
    FINDING_CONCEPT_SLUG,
    build_gate_context,
)


def fmt_n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(rate: float, digits: int = 1) -> str:
    try:
        return f"{float(rate) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def fmt_compact(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    return f"{number:.2f}"


def fmt_dec(value: float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def plural(n: int, word: str) -> str:
    return word if int(n) == 1 else f"{word}s"


def list_names(items: list[Any], limit: int = 3) -> str:
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("column") or item.get("feature") or item.get("a")
            names.append(str(name))
        else:
            names.append(str(item))
    names = [n for n in names if n and n != "None"]
    if not names:
        return "<none>"
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" (+{len(names) - limit})"


def target_name(ctx: dict[str, Any], fallback: str = "<target>") -> str:
    target = ctx.get("target") or {}
    if isinstance(target, dict) and target.get("name"):
        return str(target["name"])
    return fallback


def is_classification(ctx: dict[str, Any]) -> bool:
    target = ctx.get("target") or {}
    return isinstance(target, dict) and str(target.get("task") or "") == "classification"


def is_regression(ctx: dict[str, Any]) -> bool:
    target = ctx.get("target") or {}
    return isinstance(target, dict) and str(target.get("task") or "") == "regression"


def first_numeric(ctx: dict[str, Any], fallback: str = "<numeric_column>") -> str:
    rows = ctx.get("numeric") or []
    if rows:
        return str(rows[0].get("name") or fallback)
    return fallback


def first_categorical(ctx: dict[str, Any], fallback: str = "<categorical_column>") -> str:
    cats = ctx.get("categorical") or []
    if cats:
        item = cats[0]
        return str(item.get("name") if isinstance(item, dict) else item)
    return fallback


def first_feature(ctx: dict[str, Any], fallback: str = "<feature>") -> str:
    features = ctx.get("features") or []
    if features:
        return str(features[0])
    if ctx.get("numeric"):
        return first_numeric(ctx, fallback)
    if ctx.get("categorical"):
        return first_categorical(ctx, fallback)
    cols = ctx.get("cols") or []
    for col in cols:
        name = str(col.get("name") if isinstance(col, dict) else col)
        if name and name != target_name(ctx, ""):
            return name
    return fallback


def first_missing(ctx: dict[str, Any], fallback: str = "<column>") -> str:
    missing = ctx.get("missing") or []
    if missing:
        return str(missing[0].get("name") or fallback)
    return fallback


def quote_list(names: list[str], indent: str = "    ") -> str:
    if not names:
        return f"{indent}# add column names here"
    return ",\n".join(f'{indent}"{name}"' for name in names)


def code_block(*lines: str) -> str:
    return "\n".join(line.rstrip() for line in lines if line is not None)


def build_academy_context(report: dict[str, Any]) -> dict[str, Any]:
    """Session-adaptive context for Academy binders (extends gate context)."""
    ctx = dict(build_gate_context(report))
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    bivariate = report.get("bivariate") or {}
    findings = list(report.get("findings") or [])

    # Normalise MI rows to {name, mi}.
    mi_norm: list[dict[str, Any]] = []
    for item in ctx.get("mi") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("feature") or item.get("column")
        try:
            score = float(item.get("mi") or item.get("score") or item.get("value") or 0)
        except (TypeError, ValueError):
            continue
        if name:
            mi_norm.append({"name": str(name), "mi": score})
    mi_norm.sort(key=lambda row: row["mi"], reverse=True)
    ctx["mi"] = mi_norm

    # Categorical as list of names (gate may already be strings).
    cats = ctx.get("categorical") or []
    if cats and isinstance(cats[0], dict):
        ctx["categorical"] = [str(c.get("name") or c) for c in cats]
    else:
        ctx["categorical"] = [str(c) for c in cats]

    features = list(overview.get("eligible_feature_columns") or overview.get("feature_columns") or [])
    if not features:
        target = target_name(ctx, "")
        id_like = set(ctx.get("idLike") or [])
        constants = set(ctx.get("constants") or [])
        features = [
            str(c.get("name") if isinstance(c, dict) else c)
            for c in (ctx.get("cols") or [])
            if str(c.get("name") if isinstance(c, dict) else c) not in id_like | constants | {target}
        ]
    ctx["features"] = [str(f) for f in features if f]

    cells = max(int(ctx.get("rows") or 0) * max(int(ctx.get("colCount") or 0), 1), 1)
    missing_cells = int(ctx.get("missingCells") or 0)
    ctx["completeness"] = max(0.0, min(1.0, 1.0 - (missing_cells / cells)))
    ctx["memoryMB"] = overview.get("memory_mb") or overview.get("approx_memory_mb")

    # Finding citations keyed by redesign curriculum slug and catalog aliases.
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for item in findings:
        slug = FINDING_CONCEPT_SLUG.get(str(item.get("key") or ""))
        if not slug:
            continue
        row = {
            "key": item.get("key"),
            "severity": item.get("severity") or item.get("level"),
            "title": item.get("title") or item.get("summary") or item.get("key"),
            "detail": item.get("detail") or item.get("message") or "",
        }
        by_slug.setdefault(slug, []).append(row)
        alias = CONCEPT_ALIASES.get(slug)
        if alias and alias != slug:
            by_slug.setdefault(alias, []).append(row)
    ctx["findings_by_slug"] = by_slug
    ctx["task"] = (
        str((ctx.get("target") or {}).get("task"))
        if isinstance(ctx.get("target"), dict)
        else "unsupervised"
    )
    ctx["has_target"] = bool(isinstance(ctx.get("target"), dict) and ctx["target"].get("name"))
    ctx["analyzers"] = {
        "quality": bool(quality),
        "bivariate": bool(bivariate),
        "vif": bool(ctx.get("vif")),
        "mi": bool(ctx.get("mi")),
        "drift": bool(ctx.get("drifted")),
        "outliers": bool(ctx.get("anomalies") or any(
            float(n.get("outlierRate") or 0) > 0 for n in (ctx.get("numeric") or []) if isinstance(n, dict)
        )),
    }
    return ctx


def finding_blurb(ctx: dict[str, Any], slug: str) -> str:
    rows = (ctx.get("findings_by_slug") or {}).get(slug) or []
    if not rows:
        return ""
    top = rows[0]
    title = str(top.get("title") or slug)
    severity = str(top.get("severity") or "")
    prefix = f"{severity} · " if severity else ""
    extra = f" (+{len(rows) - 1} more)" if len(rows) > 1 else ""
    return f"{prefix}{title}{extra}"
