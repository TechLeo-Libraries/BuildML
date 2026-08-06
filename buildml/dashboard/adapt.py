"""Dataset-adaptive copy helpers for the Industry EDA App.

Every surface that narrates a session (Cockpit sheet, Academy session lines,
Readiness Gates evidence) must bind language to the live report: task type,
declared target column, real column names, sampling disclosures, and skipped
analyzers. Nothing in this module assumes a demo/churn schema.

Academy / Gates agents — extension contract
------------------------------------------
Python (payload builders)::

    from buildml.dashboard.adapt import (
        build_adapt_context,
        list_names,
        plural,
        session_sentence,
        target_phrase,
        what_to_change,
    )

    ctx = build_adapt_context(report)
    # ctx["target_column"], ctx["task"], ctx["columns"], ctx["skipped_analyzers"], …
    line = session_sentence(report)
    bullets = what_to_change(report)

Frontend (presentation)::

    import {
      callout, codeBlock, calcBlock, whatToChange, sectionScaffold,
    } from "./learn_ui.js";

Wire ``learn_ui.js`` before any ``academy_view.js`` / ``gates_view.js`` /
``cockpit_view.js`` in ``templates/index.html``. Prefer these primitives over
ad-hoc markup so beginner tips, evidence callouts, and copyable code stay
Industry-consistent.

Do **not** hardcode demo column names (e.g. ``target_churn``, ``tenure``,
``monthly_charges``) in App payloads. Always resolve names from ``report``.
"""

from __future__ import annotations

from typing import Any

from buildml.dashboard.serialize import flagged_column_names

# Names that must never be required by App payloads (demo / launcher leftovers).
FORBIDDEN_REQUIRED_COLUMNS = frozenset(
    {
        "target_churn",
        "churned",
        "tenure",
        "monthly_charges",
        "total_charges",
        "customer_id",
    }
)


def fmt_n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "—"
    if n <= 1.0:
        n *= 100.0
    return f"{n:.{digits}f}%"


def plural(n: int, word: str, plural_word: str | None = None) -> str:
    if n == 1:
        return word
    return plural_word or f"{word}s"


def list_names(items: Any, limit: int = 3) -> str:
    """Join column / label names for prose; never invent demo names."""
    names: list[str] = []
    if isinstance(items, dict):
        items = list(items.keys())
    if not isinstance(items, list):
        return ""
    for item in items:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = str(
                item.get("name")
                or item.get("column")
                or item.get("feature")
                or item.get("key")
                or ""
            )
        else:
            name = str(item) if item is not None else ""
        if name:
            names.append(name)
    if not names:
        return ""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" (+{len(names) - limit})"


def _target_block(report: dict[str, Any]) -> dict[str, Any]:
    target = report.get("target") or {}
    summary = target.get("summary") if isinstance(target.get("summary"), dict) else {}
    column = target.get("column") or summary.get("column")
    raw_task = (
        target.get("task")
        or summary.get("task")
        or summary.get("type")
        or ""
    )
    task = _normalize_task(raw_task)
    class_counts = summary.get("class_counts") or target.get("class_balance") or {}
    n_classes = summary.get("n_classes")
    if n_classes is None and isinstance(class_counts, dict) and class_counts:
        n_classes = len(class_counts)
    return {
        "column": str(column) if column else None,
        "task": task,
        "task_raw": str(raw_task) if raw_task else None,
        "n_classes": int(n_classes) if n_classes is not None else None,
        "class_counts": (
            {str(k): int(v) for k, v in class_counts.items()}
            if isinstance(class_counts, dict)
            else {}
        ),
        "imbalance_ratio": summary.get("imbalance_ratio"),
    }


def _normalize_task(raw: Any) -> str | None:
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if "class" in text:
        return "classification"
    if "regress" in text:
        return "regression"
    if text in {"classification", "regression", "unsupervised", "unknown"}:
        return text
    return text


def analyzer_status(report: dict[str, Any]) -> list[dict[str, str]]:
    """Compact ran / skipped / n/a catalog derived from which sections exist."""
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    univariate = report.get("univariate") or {}
    bivariate = report.get("bivariate") or {}
    multivariate = report.get("multivariate") or {}
    target = report.get("target") or {}
    drift = report.get("drift") or {}
    outliers = report.get("outliers") or {}
    has_target = bool(target.get("column"))

    def status(ran: bool, *, na: bool = False) -> str:
        if na:
            return "not_applicable"
        return "ran" if ran else "skipped"

    cards = [
        {
            "family": "quality",
            "status": status(bool(quality)),
            "detail": (
                f"completeness {fmt_pct(quality.get('completeness_score'))}"
                if quality
                else "section absent"
            ),
        },
        {
            "family": "univariate",
            "status": status(bool(univariate.get("per_column"))),
            "detail": (
                f"{fmt_n(len(univariate.get('per_column') or {}))} columns profiled"
                if univariate.get("per_column")
                else "section absent"
            ),
        },
        {
            "family": "bivariate",
            "status": status(
                bool(
                    bivariate.get("pearson")
                    or bivariate.get("spearman")
                    or bivariate.get("mutual_information_vs_target")
                    or bivariate.get("cramers_v")
                )
            ),
            "detail": "associations among eligible features",
        },
        {
            "family": "multivariate",
            "status": status(bool(multivariate.get("vif") or multivariate.get("pca"))),
            "detail": "VIF / PCA when complete cases allow",
        },
        {
            "family": "target",
            "status": status(bool(target.get("summary") or target.get("column")), na=not has_target),
            "detail": (
                f"column={target.get('column')}"
                if has_target
                else "no target role declared"
            ),
        },
        {
            "family": "drift",
            "status": status(
                bool(drift.get("flagged_columns") is not None or drift.get("per_column")),
                na=not has_target,
            ),
            "detail": (
                f"{fmt_n(len(flagged_column_names(drift.get('flagged_columns'))))} flags"
                if drift
                else "section absent"
            ),
        },
        {
            "family": "outliers",
            "status": status(bool(outliers.get("per_column") or outliers.get("multivariate"))),
            "detail": "univariate / multivariate screens",
        },
        {
            "family": "adaptive_plan",
            "status": status(bool(report.get("adaptive_plan"))),
            "detail": f"{fmt_n(len(report.get('adaptive_plan') or []))} plot specs",
        },
    ]
    # Attach sampling disclosure when present.
    n_rows = overview.get("n_rows")
    analysis_rows = overview.get("analysis_rows") or n_rows
    if analysis_rows != n_rows:
        cards.insert(
            0,
            {
                "family": "sampling",
                "status": "ran",
                "detail": f"analysed {fmt_n(analysis_rows)} of {fmt_n(n_rows)} rows",
            },
        )
    return cards


def skipped_analyzers(report: dict[str, Any]) -> list[str]:
    return [
        f"{card['family']}: {card['detail']}"
        for card in analyzer_status(report)
        if card["status"] in {"skipped", "not_applicable"}
    ]


def build_adapt_context(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten live report facts used for adaptive copy across App surfaces.

    Returns
    -------
    dict
        JSON-safe adaptive fields. Academy / Gates / Cockpit should prefer these
        over any template defaults. Never requires demo column names.
    """
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    findings = report.get("findings") or []
    warnings = list(report.get("warnings") or [])
    target = _target_block(report)

    columns = list(overview.get("columns") or overview.get("analysis_columns") or [])
    eligible = list(overview.get("eligible_feature_columns") or [])
    numeric = list(overview.get("numeric_columns") or [])
    categorical = list(overview.get("categorical_columns") or [])
    constants = flagged_column_names(quality.get("constant_columns"))
    id_like = flagged_column_names(quality.get("id_like_columns"))

    n_rows = overview.get("n_rows")
    analysis_rows = overview.get("analysis_rows") or n_rows
    n_cols = overview.get("n_columns") or len(columns)

    blocking = [
        item
        for item in findings
        if str(item.get("severity", "")).lower() in {"high", "critical", "crit"}
    ]
    statuses = analyzer_status(report)
    skipped = [
        card["family"]
        for card in statuses
        if card["status"] in {"skipped", "not_applicable"}
    ]

    label = (
        overview.get("label")
        or overview.get("dataset")
        or overview.get("name")
        or "session"
    )

    return {
        "session_label": str(label),
        "engine": str(overview.get("engine") or "pandas"),
        "mode": str(overview.get("mode") or "eager"),
        "n_rows": n_rows,
        "analysis_rows": analysis_rows,
        "n_columns": n_cols,
        "columns": [str(c) for c in columns],
        "eligible_features": [str(c) for c in eligible],
        "numeric_columns": [str(c) for c in numeric],
        "categorical_columns": [str(c) for c in categorical],
        "constant_columns": constants,
        "id_like_columns": id_like,
        "target_column": target["column"],
        "task": target["task"],
        "task_raw": target["task_raw"],
        "n_classes": target["n_classes"],
        "class_counts": target["class_counts"],
        "imbalance_ratio": target["imbalance_ratio"],
        "has_target": target["column"] is not None,
        "sampled": bool(analysis_rows != n_rows or warnings),
        "warnings": warnings,
        "finding_count": len(findings),
        "blocking_count": len(blocking),
        "completeness": quality.get("completeness_score"),
        "missing_cells": quality.get("missing_cell_count"),
        "analyzers": statuses,
        "skipped_analyzers": skipped,
        "skipped_details": skipped_analyzers(report),
        "adaptive_plan_count": len(report.get("adaptive_plan") or []),
        # Convenience phrases for curriculum / gate writers.
        "target_phrase": target_phrase(report),
        "task_phrase": task_phrase(report),
        "scope_phrase": scope_phrase(report),
        "session_sentence": session_sentence(report),
    }


def target_phrase(report: dict[str, Any]) -> str:
    block = _target_block(report)
    if not block["column"]:
        return "no target column is declared"
    return f"target column `{block['column']}`"


def task_phrase(report: dict[str, Any]) -> str:
    block = _target_block(report)
    if not block["task"]:
        return "task type is undeclared"
    if block["task"] == "classification" and block["n_classes"]:
        return f"{block['task']} ({block['n_classes']} classes)"
    return str(block["task"])


def scope_phrase(report: dict[str, Any]) -> str:
    overview = report.get("overview") or {}
    n_rows = overview.get("n_rows")
    analysis_rows = overview.get("analysis_rows") or n_rows
    n_cols = overview.get("n_columns")
    if analysis_rows != n_rows:
        return (
            f"{fmt_n(analysis_rows)} of {fmt_n(n_rows)} rows · "
            f"{fmt_n(n_cols)} columns"
        )
    return f"{fmt_n(analysis_rows)} rows · {fmt_n(n_cols)} columns"


def session_sentence(report: dict[str, Any]) -> str:
    """One adaptive sentence for Academy 'In this session' / Cockpit lead-ins."""
    ctx_overview = report.get("overview") or {}
    target = _target_block(report)
    n_rows = ctx_overview.get("analysis_rows") or ctx_overview.get("n_rows")
    n_cols = ctx_overview.get("n_columns")
    eligible = len(ctx_overview.get("eligible_feature_columns") or [])
    parts = [
        f"{fmt_n(n_rows)} rows",
        f"{fmt_n(n_cols)} columns",
        f"{fmt_n(eligible)} eligible {plural(eligible, 'feature')}",
    ]
    if target["column"]:
        parts.append(f"{target_phrase(report)} as {task_phrase(report)}")
    else:
        parts.append("no target declared")
    skipped = skipped_analyzers(report)
    if skipped:
        parts.append(
            f"{len(skipped)} analyzer {plural(len(skipped), 'family')} skipped or n/a"
        )
    return "; ".join(parts) + "."


def what_to_change(report: dict[str, Any], *, limit: int = 6) -> list[dict[str, str]]:
    """Actionable 'what to change' rows bound to this report's findings / gaps."""
    items: list[dict[str, str]] = []
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    target = _target_block(report)

    if not target["column"]:
        items.append(
            {
                "change": "Declare a target role",
                "why": "Target screens, MI-vs-target, and drift stay unavailable until a target is set.",
                "api": 'session.set_roles({...: "target"})',
            }
        )

    id_like = flagged_column_names(quality.get("id_like_columns"))
    if id_like:
        items.append(
            {
                "change": f"Review identifier-like columns ({list_names(id_like)})",
                "why": "Near-unique columns can leak identity into the design matrix.",
                "api": 'session.set_roles({...: "id"})',
            }
        )

    constants = flagged_column_names(quality.get("constant_columns"))
    if constants:
        items.append(
            {
                "change": f"Drop or ignore constants ({list_names(constants)})",
                "why": "Zero-variance columns consume width without signal.",
                "api": 'session.set_roles({...: "ignore"})',
            }
        )

    for item in report.get("recommendation_details") or []:
        if len(items) >= limit:
            break
        action = item.get("action") or {}
        operation = action.get("operation") if isinstance(action, dict) else None
        title = item.get("title") or operation or "Follow recommendation"
        call = f"session.{operation}(...)" if operation else (item.get("api") or "")
        items.append(
            {
                "change": str(title),
                "why": str(item.get("rationale") or item.get("detail") or "Named by the readiness sheet."),
                "api": str(call),
            }
        )

    if overview.get("analysis_rows") != overview.get("n_rows"):
        items.append(
            {
                "change": "Re-run EDA without a row budget if tails matter",
                "why": scope_phrase(report) + " — sampling can hide rare levels.",
                "api": "session.eda(sample_rows=None)",
            }
        )

    # Deduplicate by change text while preserving order.
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in items:
        key = row["change"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= limit:
            break
    return unique


def assert_no_demo_column_requirements(payload: dict[str, Any]) -> list[str]:
    """Return forbidden demo column names that appear as *required* payload keys.

    Used in tests. Column *values* drawn from a real demo dataset are allowed;
    requiring those names as schema keys is not.
    """
    hits: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_s = str(key)
                here = f"{path}.{key_s}" if path else key_s
                if key_s in FORBIDDEN_REQUIRED_COLUMNS:
                    hits.append(here)
                # Explicit schema markers.
                if key_s in {"required_columns", "required", "must_have_columns"}:
                    names = value if isinstance(value, list) else [value]
                    for name in names:
                        if str(name) in FORBIDDEN_REQUIRED_COLUMNS:
                            hits.append(f"{here}={name}")
                walk(value, here)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(payload, "")
    return hits
