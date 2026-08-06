# ruff: noqa: E501
"""Readiness Gates for the Industry EDA App.

Statuses are computed from the live EDA report only. Human/"mark for this
session" toggles exist exclusively in the SPA client and are never accepted,
stored, or echoed by this module or any dashboard API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from buildml.dashboard.gate_teaching import enrich_gate_row
from buildml.dashboard.serialize import flagged_column_names
from buildml.explain.concepts import CONCEPT_NOTES

CLEAR = "clear"
OPEN = "open"
HUMAN = "human"
NA = "na"

GATE_STATUS_LABELS: dict[str, str] = {
    CLEAR: "settled by the frame",
    OPEN: "open — measurable",
    HUMAN: "needs a human judgment",
    NA: "not applicable",
}

GATE_STAGES: tuple[dict[str, Any], ...] = (
    {
        "key": 0,
        "n": "00",
        "label": "Framing",
        "blurb": "what the data would have to be before any statistic means anything",
    },
    {
        "key": 1,
        "n": "01",
        "label": "Data quality",
        "blurb": "what the frame is before anything is fitted",
    },
    {
        "key": 2,
        "n": "02",
        "label": "Relationships",
        "blurb": "how columns relate to the target and to each other",
    },
    {
        "key": 3,
        "n": "03",
        "label": "Validation",
        "blurb": "what the evidence is allowed to certify",
    },
    {
        "key": 4,
        "n": "04",
        "label": "Evaluation",
        "blurb": "what a score is worth",
    },
    {
        "key": 5,
        "n": "05",
        "label": "Interpretation",
        "blurb": "what a fitted model may be said to show, and what ships with it",
    },
)

# Redesign curriculum slug → BuildML concept key when a teaching note exists.
CONCEPT_ALIASES: dict[str, str] = {
    "problem-framing": "diagnostic-uncertainty",
    "unit-of-analysis": "column-roles",
    "population-and-sampling-frame": "diagnostic-uncertainty",
    "target-definition": "column-roles",
    "provenance-and-lineage": "feature-schema",
    "sensitive-attributes": "feature-schema",
    "dtypes-and-storage": "feature-schema",
    "missing-data": "missing-data",
    "missingness-mechanisms": "missing-data",
    "duplicate-records": "diagnostic-uncertainty",
    "constant-and-near-constant": "column-roles",
    "high-cardinality": "categorical-encoding",
    "measurement-units-and-ranges": "diagnostic-uncertainty",
    "text-hygiene": "text-features",
    "join-integrity": "feature-schema",
    "cross-field-consistency": "diagnostic-uncertainty",
    "datetime-parsing": "feature-schema",
    "precision-and-heaping": "diagnostic-uncertainty",
    "univariate-distributions": "normality-screens",
    "skew-and-transforms": "feature-scaling",
    "derived-and-redundant-columns": "feature-selection",
    "variance-inflation": "variance-inflation",
    "non-linearity-and-binning": "feature-binning",
    "confounding-and-subgroups": "causal-assumptions",
    "sparsity-and-dimensionality": "overfitting",
    "feature-scaling": "feature-scaling",
    "data-splitting": "data-splitting",
    "stratification": "data-splitting",
    "pipeline-order": "encoding-imputation-scaling",
    "leakage": "leakage-boundary",
    "temporal-structure": "data-splitting",
    "group-structure": "data-splitting",
    "nested-validation": "cross-validation",
    "sample-size-and-power": "diagnostic-uncertainty",
    "multiple-comparisons": "diagnostic-uncertainty",
    "reproducibility": "reproducibility",
    "dataset-drift": "dataset-drift",
    "outlier-screens": "outlier-handling",
    "metric-selection": "model-selection",
    "baselines": "baselines",
    "class-imbalance": "class-imbalance",
    "thresholds-and-costs": "thresholds",
    "uncertainty-intervals": "probabilistic-uncertainty",
    "slice-evaluation": "evaluation-partitions",
    "calibration": "probability-calibration",
    "feature-importance-methods": "feature-importance",
    "effect-shapes": "feature-importance",
    "learning-curves-and-capacity": "training-curves",
    "causal-caution": "causal-assumptions",
    "handoff-and-monitoring": "operation-history",
}

# Finding keys → curriculum concept slug (for gate citations + academy cited).
FINDING_CONCEPT_SLUG: dict[str, str] = {
    "eda.scope": "data-splitting",
    "quality.completeness": "missing-data",
    "quality.constants": "constant-and-near-constant",
    "quality.identifiers": "column-roles",
    "quality.duplicates": "duplicate-records",
    "quality.high_cardinality": "high-cardinality",
    "quality.near_constant": "constant-and-near-constant",
    "quality.text_hygiene": "text-hygiene",
    "quality.ranges": "measurement-units-and-ranges",
    "relationships.vif": "variance-inflation",
    "relationships.mi_leader": "mutual-information",
    "relationships.correlated_pairs": "derived-and-redundant-columns",
    "relationships.scaling": "feature-scaling",
    "outliers.univariate": "outlier-screens",
    "outliers.multivariate": "outlier-screens",
    "target.summary": "class-imbalance",
    "target.missing": "column-roles",
    "validation.drift": "dataset-drift",
    "validation.leakage": "leakage",
    "validation.temporal": "temporal-structure",
    "validation.grouping": "group-structure",
    "validation.sampling": "diagnostic-uncertainty",
    "distribution.skew": "skew-and-transforms",
    "evaluation.metric": "metric-selection",
    "evaluation.baseline": "baselines",
}


@dataclass(frozen=True, slots=True)
class _GateDef:
    gate_id: str
    stage: int
    question: str
    concept: str
    resolve: Callable[[dict[str, Any]], dict[str, str]]


def resolve_concept_key(slug: str) -> str | None:
    """Map a readiness curriculum slug to a BuildML concept key, if taught."""
    if slug in CONCEPT_NOTES:
        return slug
    alias = CONCEPT_ALIASES.get(slug)
    if alias and alias in CONCEPT_NOTES:
        return alias
    return None


def build_gates_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Compute readiness gates for the Industry EDA App Gates board.

    Returns rows, counts, and stage groups. Does not accept or return any
    human decision marks — those stay in the browser for the open App session.
    """
    ctx = build_gate_context(report)
    findings = list(report.get("findings") or [])
    rows: list[dict[str, Any]] = []
    for gate in _GATE_DEFS:
        resolved = gate.resolve(ctx)
        status = str(resolved["status"])
        concept_key = resolve_concept_key(gate.concept)
        cited = [
            {"key": item.get("key"), "label": item.get("key")}
            for item in findings
            if FINDING_CONCEPT_SLUG.get(str(item.get("key", ""))) == gate.concept
        ]
        row = {
            "id": gate.gate_id,
            "key": gate.gate_id,
            "stage": gate.stage,
            "question": gate.question,
            "concept": gate.concept,
            "concept_key": concept_key,
            "status": status,
            "status_label": GATE_STATUS_LABELS.get(status, status),
            "evidence": resolved["evidence"],
            "closes": resolved["closes"],
            "findings": cited,
            "is_clear": status == CLEAR,
            "is_open": status == OPEN,
            "is_human": status == HUMAN,
            "is_na": status == NA,
            "session_mark_eligible": status in {HUMAN, OPEN},
        }
        rows.append(enrich_gate_row(row, ctx))

    def _count(status: str) -> int:
        return sum(1 for row in rows if row["status"] == status)

    answerable = sum(1 for row in rows if row["status"] != NA)
    clear_n = _count(CLEAR)
    counts = {
        "clear": clear_n,
        "open": _count(OPEN),
        "human": _count(HUMAN),
        "na": _count(NA),
        "total": len(rows),
        "answerable": answerable,
    }
    settled_pct = f"{round(100 * clear_n / max(1, answerable))}%"
    groups: list[dict[str, Any]] = []
    for stage in GATE_STAGES:
        mine = [row for row in rows if row["stage"] == stage["key"]]
        if not mine:
            continue
        outstanding = sum(1 for row in mine if row["is_open"] or row["is_human"])
        groups.append(
            {
                "key": stage["key"],
                "n": stage["n"],
                "label": stage["label"],
                "blurb": stage["blurb"],
                "rows": mine,
                "count_label": (
                    f"{len(mine)} {'gate' if len(mine) == 1 else 'gates'} · "
                    f"{outstanding} outstanding"
                ),
            }
        )
    return {
        "rows": rows,
        "counts": counts,
        "settled_pct": settled_pct,
        "stages": list(GATE_STAGES),
        "groups": groups,
        "ephemeral_notice": (
            "Marks you set here stay in this browser tab only. Refreshing the "
            "App clears them. BuildML never saves gate judgments to the session, "
            "history, disk, or any dataset copy."
        ),
        "persistence": {
            "human_decisions": False,
            "session_api": False,
            "disk": False,
            "reason": "ui_only_privacy",
        },
        "teaching": {
            "sidebar": True,
            "depth": "beginner_to_advanced",
            "worked_examples": True,
            "adaptive": True,
            "persistence": False,
        },
    }


def build_gate_context(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten an EDA report dict into the resolver context used by gate rules."""
    overview = report.get("overview") or {}
    quality = report.get("quality") or {}
    univariate = report.get("univariate") or {}
    bivariate = report.get("bivariate") or {}
    multivariate = report.get("multivariate") or {}
    target = report.get("target") or {}
    drift = report.get("drift") or {}
    outliers = report.get("outliers") or {}
    warnings = list(report.get("warnings") or [])

    n_rows = int(overview.get("n_rows") or overview.get("analysis_rows") or 0)
    analysis_rows = int(overview.get("analysis_rows") or n_rows)
    columns = list(overview.get("columns") or overview.get("analysis_columns") or [])
    dtypes = overview.get("dtypes") or {}
    per_column = univariate.get("per_column") or {}

    missing_rate = quality.get("missing_rate_by_column") or {}
    missing_list: list[dict[str, Any]] = []
    if isinstance(missing_rate, dict):
        for name, rate in missing_rate.items():
            try:
                rate_f = float(rate)
            except (TypeError, ValueError):
                continue
            if rate_f > 0:
                missing_list.append({"name": str(name), "missingRate": rate_f})
        missing_list.sort(key=lambda item: item["missingRate"], reverse=True)

    numeric: list[dict[str, Any]] = []
    profiled_numeric: list[dict[str, Any]] = []
    unprofiled: list[str] = []
    skewed: list[dict[str, Any]] = []
    for name, stats in per_column.items():
        if not isinstance(stats, dict):
            continue
        kind = str(stats.get("kind") or "")
        if kind != "numeric":
            continue
        row = {
            "name": str(name),
            "negatives": int(stats.get("negatives") or 0),
            "outlierRate": float(
                (outliers.get("per_column") or {}).get(name, {}).get("iqr_outlier_rate") or 0
            )
            if isinstance((outliers.get("per_column") or {}).get(name), dict)
            else 0.0,
            "skew": float(stats.get("skew") or 0),
            "min": stats.get("min"),
            "max": stats.get("max"),
        }
        numeric.append(row)
        if stats.get("min") is not None and stats.get("max") is not None:
            profiled_numeric.append(row)
        else:
            unprofiled.append(str(name))
        if abs(float(stats.get("skew") or 0)) > 1:
            skewed.append({"name": str(name), "skew": float(stats.get("skew") or 0)})

    cols = []
    mixed = set(str(x) for x in (quality.get("mixed_type_suspect_columns") or []))
    for name in columns:
        cols.append(
            {
                "name": str(name),
                "dtype": str(dtypes.get(name, "unknown")),
                "mixedType": str(name) in mixed,
                "caseVariants": False,
            }
        )

    high_card = []
    for item in quality.get("high_cardinality_columns") or []:
        if isinstance(item, dict):
            high_card.append(
                {
                    "name": str(item.get("column") or item.get("name") or ""),
                    "distinct": int(item.get("n_unique") or item.get("distinct") or 0),
                }
            )
        else:
            stats = per_column.get(str(item)) or {}
            distinct = int(stats.get("n_unique") or stats.get("unique") or 0)
            high_card.append({"name": str(item), "distinct": distinct})

    corr_pairs: list[dict[str, Any]] = []
    for item in bivariate.get("top_abs_pearson_pairs") or []:
        if not isinstance(item, dict):
            continue
        corr_pairs.append(
            {
                "a": str(item.get("a") or item.get("column_a") or item.get("left") or ""),
                "b": str(item.get("b") or item.get("column_b") or item.get("right") or ""),
                "r": float(item.get("r") or item.get("corr") or item.get("correlation") or 0),
            }
        )
    pearson = bivariate.get("pearson")
    has_corr = bool(corr_pairs) or bool(pearson)

    mi_rows = []
    mi_raw = bivariate.get("mutual_information_vs_target") or []
    if isinstance(mi_raw, list):
        mi_rows = mi_raw
    elif isinstance(mi_raw, dict):
        mi_rows = [{"feature": k, "mi": v} for k, v in mi_raw.items()]

    vif_rows: list[dict[str, Any]] = []
    vif_raw = multivariate.get("vif") or []
    if isinstance(vif_raw, list):
        for item in vif_raw:
            if isinstance(item, dict):
                vif_rows.append(
                    {
                        "name": str(item.get("column") or item.get("feature") or item.get("name") or ""),
                        "vif": float(item.get("vif") or 0),
                    }
                )
    elif isinstance(vif_raw, dict):
        for name, value in vif_raw.items():
            if isinstance(value, dict):
                vif_rows.append({"name": str(name), "vif": float(value.get("vif") or 0)})
            else:
                try:
                    vif_rows.append({"name": str(name), "vif": float(value)})
                except (TypeError, ValueError):
                    continue
    vif_rows.sort(key=lambda item: item["vif"], reverse=True)

    target_summary = target.get("summary") or {}
    target_payload = None
    if target.get("column"):
        task = "regression"
        classes = None
        if str(target_summary.get("type") or "").startswith("classification"):
            task = "classification"
            counts = target_summary.get("class_counts") or {}
            classes = [
                {"label": str(label), "count": int(count)} for label, count in counts.items()
            ]
        stats = None
        if task == "regression":
            stats = {
                "median": target_summary.get("median") or target_summary.get("p50"),
            }
        target_payload = {
            "name": str(target.get("column")),
            "task": task,
            "classes": classes,
            "stats": stats,
        }

    mv = outliers.get("multivariate") or {}
    anomalies = None
    if isinstance(mv, dict) and (mv.get("flagged_count") or mv.get("n_flagged") is not None):
        anomalies = {
            "flagged": int(mv.get("flagged_count") or mv.get("n_flagged") or 0),
            "scored": int(mv.get("n_scored") or mv.get("scored") or analysis_rows),
            "contamination": float(mv.get("contamination") or 0.05),
        }

    dup_count = quality.get("duplicate_row_count")
    duplicates = {
        "rows": int(dup_count or 0),
        "keyDupes": 0,
        "keyColumn": None,
    }

    datetime_cols = list(overview.get("datetime_columns") or [])
    time_col = None
    if datetime_cols:
        name = str(datetime_cols[0])
        time_col = {"name": name, "min": None, "max": None, "gaps": 0}

    return {
        "rows": analysis_rows,
        "rowsTotal": n_rows,
        "colCount": len(columns) or int(overview.get("n_columns") or 0),
        "sampled": analysis_rows < n_rows or bool(warnings),
        "idLike": [str(x) for x in (quality.get("id_like_columns") or [])],
        "missingCells": int(quality.get("missing_cell_count") or 0),
        "missing": missing_list,
        "constants": [str(x) for x in (quality.get("constant_columns") or [])],
        "nearConstant": [str(x) for x in (quality.get("quasi_constant_columns") or [])],
        "highCard": high_card,
        "numeric": numeric,
        "profiledNumeric": profiled_numeric,
        "unprofiled": unprofiled,
        "skewed": skewed,
        "cols": cols,
        "timeCol": time_col,
        "groupCol": None,
        "duplicates": duplicates,
        "corrPairs": corr_pairs,
        "hasCorr": has_corr,
        "mi": mi_rows,
        "vif": vif_rows,
        "vifThreshold": 5.0,
        "eligible": len(overview.get("eligible_feature_columns") or []),
        "completeRows": int(multivariate.get("complete_case_rows") or analysis_rows),
        "categorical": [str(x) for x in (overview.get("categorical_columns") or [])],
        "target": target_payload,
        "leakage": [],
        "drifted": flagged_column_names(drift.get("flagged_columns")),
        "anomalies": anomalies,
        "ds": {
            "engine": str(overview.get("engine") or "pandas"),
            "version": str(overview.get("mode") or "memory"),
        },
    }


def _fmt_n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(rate: float, digits: int = 1) -> str:
    return f"{rate * 100:.{digits}f}%"


def _fmt_compact(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _plural(n: int, word: str) -> str:
    return word if n == 1 else f"{word}s"


def _list_names(items: list[Any], limit: int = 3) -> str:
    names = [str(x if not isinstance(x, dict) else x.get("name") or x.get("column") or x) for x in items]
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" (+{len(names) - limit})"


def _gate(
    gate_id: str,
    stage: int,
    question: str,
    concept: str,
    resolve: Callable[[dict[str, Any]], dict[str, str]],
) -> _GateDef:
    return _GateDef(gate_id, stage, question, concept, resolve)


def _r00_1(c: dict[str, Any]) -> dict[str, str]:
    return {
        "status": HUMAN,
        "evidence": (
            f"{_fmt_n(c['rows'])} rows and {c['colCount']} columns were profiled; "
            "no decision statement is attached to the session."
        ),
        "closes": "One sentence: who acts, on what output, at what moment.",
    }


def _r00_2(c: dict[str, Any]) -> dict[str, str]:
    id_like = c.get("idLike") or []
    if id_like:
        return {
            "status": OPEN,
            "evidence": (
                f"{_list_names(id_like, 3)} "
                f"{'is' if len(id_like) == 1 else 'are'} near-unique across "
                f"{_fmt_n(c['rows'])} rows, but no uniqueness assertion ran."
            ),
            "closes": "A named key with a uniqueness check that passes.",
        }
    return {
        "status": HUMAN,
        "evidence": "No near-unique column was observed, so no candidate key is visible in the frame.",
        "closes": "A named key with a uniqueness check that passes.",
    }


def _r00_3(c: dict[str, Any]) -> dict[str, str]:
    sampled = " under sampling" if c.get("sampled") else ""
    return {
        "status": HUMAN,
        "evidence": (
            f"{_fmt_n(c['rows'])} of {_fmt_n(c['rowsTotal'])} rows examined{sampled}; "
            "upstream filters are invisible here."
        ),
        "closes": "The extract query, the window, and what was excluded.",
    }


def _r00_4(c: dict[str, Any]) -> dict[str, str]:
    target = c.get("target")
    if target:
        return {
            "status": HUMAN,
            "evidence": (
                f"{target['name']} is used as a {target['task']} target; "
                "how it was constructed is not recorded."
            ),
            "closes": "The label rule with its anchor, horizon and censoring policy.",
        }
    return {
        "status": OPEN,
        "evidence": "No target is declared, so this is not yet a supervised problem.",
        "closes": "A declared target column.",
    }


def _r00_5(c: dict[str, Any]) -> dict[str, str]:
    ds = c.get("ds") or {}
    return {
        "status": HUMAN,
        "evidence": (
            f"{c['colCount']} columns profiled on {ds.get('engine')} {ds.get('version')}; "
            "no lineage or refresh schedule travels with them."
        ),
        "closes": "Source, derivation and known-at time per column.",
    }


def _r00_6(c: dict[str, Any]) -> dict[str, str]:
    return {
        "status": HUMAN,
        "evidence": f"No sensitivity classification is attached to any of the {c['colCount']} columns.",
        "closes": "An inventory naming personal, protected and neither — plus which are kept for evaluation only.",
    }


def _r01_1(c: dict[str, Any]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for col in c.get("cols") or []:
        dtype = str(col.get("dtype") or "unknown")
        counts[dtype] = counts.get(dtype, 0) + 1
    blob = ", ".join(f"{n} {t}" for t, n in sorted(counts.items())) or "no columns"
    return {
        "status": HUMAN,
        "evidence": f"{blob} — as loaded, not as asserted.",
        "closes": "An explicit dtype per column at load time.",
    }


def _r01_2(c: dict[str, Any]) -> dict[str, str]:
    if c.get("missingCells", 0) == 0:
        return {
            "status": CLEAR,
            "evidence": f"No missing cells across {c['colCount']} columns.",
            "closes": "One in-fold strategy per gappy column, plus indicators where the gap may be informative.",
        }
    missing = c.get("missing") or []
    worst = missing[0] if missing else {"name": "?", "missingRate": 0}
    return {
        "status": OPEN,
        "evidence": (
            f"{_fmt_n(c['missingCells'])} cells missing across {len(missing)} "
            f"{_plural(len(missing), 'column')}; worst is {worst['name']} at "
            f"{_fmt_pct(float(worst['missingRate']), 1)}."
        ),
        "closes": "One in-fold strategy per gappy column, plus indicators where the gap may be informative.",
    }


def _r01_3(c: dict[str, Any]) -> dict[str, str]:
    missing = c.get("missing") or []
    if missing:
        return {
            "status": OPEN,
            "evidence": (
                f"{len(missing)} {_plural(len(missing), 'column')} carry gaps; "
                "no mechanism was inferred for any of them."
            ),
            "closes": "MCAR / MAR / MNAR recorded per column, on evidence.",
        }
    return {
        "status": NA,
        "evidence": "Nothing is missing in this extract.",
        "closes": "MCAR / MAR / MNAR recorded per column, on evidence.",
    }


def _r01_4(c: dict[str, Any]) -> dict[str, str]:
    d = c.get("duplicates")
    if not d:
        return {
            "status": OPEN,
            "evidence": "No duplicate screen ran on this frame.",
            "closes": "A stated grain, deduplication before the split, and near-duplicate checks after string cleaning.",
        }
    ok = not d.get("rows") and not d.get("keyDupes")
    if ok:
        return {
            "status": CLEAR,
            "evidence": f"No exact duplicates and no repeated keys across {_fmt_n(c['rows'])} rows.",
            "closes": "A stated grain, deduplication before the split, and near-duplicate checks after string cleaning.",
        }
    key_bit = (
        f" and {_fmt_n(d['keyDupes'])} repeated {_plural(int(d['keyDupes']), 'key')} on {d['keyColumn']}"
        if d.get("keyDupes")
        else ""
    )
    return {
        "status": OPEN,
        "evidence": f"{_fmt_n(d['rows'])} exact duplicate {_plural(int(d['rows']), 'row')}{key_bit}.",
        "closes": "A stated grain, deduplication before the split, and near-duplicate checks after string cleaning.",
    }


def _r01_5(c: dict[str, Any]) -> dict[str, str]:
    constants = c.get("constants") or []
    near = c.get("nearConstant") or []
    n = len(constants) + len(near)
    if n:
        const_bit = f" ({_list_names(constants, 3)})" if constants else ""
        return {
            "status": OPEN,
            "evidence": (
                f"{len(constants)} constant and {len(near)} near-constant "
                f"{_plural(n, 'column')}{const_bit}."
            ),
            "closes": "Constants ignored, near-constants coarsened or knowingly kept.",
        }
    return {
        "status": CLEAR,
        "evidence": "No constant or near-constant columns were observed.",
        "closes": "Constants ignored, near-constants coarsened or knowingly kept.",
    }


def _r01_6(c: dict[str, Any]) -> dict[str, str]:
    high = c.get("highCard") or []
    if high:
        added = sum(int(x.get("distinct") or 0) for x in high)
        return {
            "status": OPEN,
            "evidence": (
                f"{len(high)} categorical {_plural(len(high), 'column')} exceed 20 levels; "
                f"one-hot would add about {_fmt_n(added)} columns."
            ),
            "closes": "Group-rare, in-fold target encoding or attribute replacement — plus an unseen-level policy.",
        }
    return {
        "status": CLEAR,
        "evidence": "No categorical column exceeds 20 observed levels.",
        "closes": "Group-rare, in-fold target encoding or attribute replacement — plus an unseen-level policy.",
    }


def _r01_7(c: dict[str, Any]) -> dict[str, str]:
    negs = [n for n in (c.get("numeric") or []) if int(n.get("negatives") or 0) > 0]
    if negs:
        return {
            "status": OPEN,
            "evidence": (
                f"{len(negs)} numeric {_plural(len(negs), 'column')} contain negatives "
                f"({_list_names(negs, 3)}); legitimacy is a domain question."
            ),
            "closes": "A min/max assertion per numeric column and sentinels converted to missing.",
        }
    return {
        "status": HUMAN,
        "evidence": "No negative values observed; no range assertion has been declared either.",
        "closes": "A min/max assertion per numeric column and sentinels converted to missing.",
    }


def _r01_8(c: dict[str, Any]) -> dict[str, str]:
    mixed = sum(1 for x in (c.get("cols") or []) if x.get("mixedType"))
    varia = sum(1 for x in (c.get("cols") or []) if x.get("caseVariants"))
    total = mixed + varia
    if total:
        return {
            "status": OPEN,
            "evidence": (
                f"{mixed} mixed-type and {varia} case-variant {_plural(total, 'column')}; "
                "level counts on this sheet use raw strings."
            ),
            "closes": "Strip, normalise, case-fold, recount — and the same transform at prediction time.",
        }
    return {
        "status": HUMAN,
        "evidence": "No mixed-type or case-variant columns were observed; no normalisation was applied either.",
        "closes": "Strip, normalise, case-fold, recount — and the same transform at prediction time.",
    }


def _r01_9(c: dict[str, Any]) -> dict[str, str]:
    return {
        "status": HUMAN,
        "evidence": f"{_fmt_n(c['rows'])} rows are present; no join history travels with the frame.",
        "closes": "Expected cardinality and a minimum match rate per join.",
    }


def _r01_10(c: dict[str, Any]) -> dict[str, str]:
    return {
        "status": HUMAN,
        "evidence": f"{c['colCount']} columns were profiled independently; no cross-field constraint was tested.",
        "closes": "One assertion per known relationship, run on every extract.",
    }


def _r01_11(c: dict[str, Any]) -> dict[str, str]:
    time_col = c.get("timeCol")
    if time_col:
        return {
            "status": OPEN,
            "evidence": (
                f"{time_col['name']} spans {time_col.get('min')} to {time_col.get('max')}; "
                "format and timezone are not recorded."
            ),
            "closes": "An explicit parse with format and tz, and a span checked against reality.",
        }
    return {
        "status": HUMAN,
        "evidence": "No column is typed as a date, which is not the same as the rows being order-free.",
        "closes": "An explicit parse with format and tz, and a span checked against reality.",
    }


def _r01_12(c: dict[str, Any]) -> dict[str, str]:
    n = len(c.get("numeric") or [])
    return {
        "status": HUMAN,
        "evidence": (
            f"{n} numeric {_plural(n, 'column')} were summarised; "
            "digit preference and edge pile-ups were not tested."
        ),
        "closes": "True precision per column and edge masses classified as caps or tails.",
    }


def _r02_1(c: dict[str, Any]) -> dict[str, str]:
    numeric = c.get("numeric") or []
    if not numeric:
        return {
            "status": NA,
            "evidence": "This frame has no numeric columns.",
            "closes": "Quartiles plus a histogram per column, with the shape named.",
        }
    unprofiled = c.get("unprofiled") or []
    if unprofiled:
        return {
            "status": OPEN,
            "evidence": (
                f"{len(numeric)} numeric {_plural(len(numeric), 'column')} present, but "
                f"{len(unprofiled)} of them carry no quartiles or range in this profile."
            ),
            "closes": "Quartiles plus a histogram per column, with the shape named.",
        }
    return {
        "status": CLEAR,
        "evidence": f"{len(numeric)} numeric {_plural(len(numeric), 'column')} summarised with quartiles and ranges.",
        "closes": "Quartiles plus a histogram per column, with the shape named.",
    }


def _r02_2(c: dict[str, Any]) -> dict[str, str]:
    skewed = c.get("skewed") or []
    if skewed:
        sample = ", ".join(f"{x['name']} ({x['skew']:.2f})" for x in skewed[:3])
        return {
            "status": OPEN,
            "evidence": f"{len(skewed)} {_plural(len(skewed), 'column')} exceed |skew| 1: {sample}.",
            "closes": "Transform or not, per column, decided against the model family and fitted in-fold.",
        }
    return {
        "status": CLEAR,
        "evidence": "No numeric column exceeds |skew| 1.",
        "closes": "Transform or not, per column, decided against the model family and fitted in-fold.",
    }


def _r02_3(c: dict[str, Any]) -> dict[str, str]:
    pairs = c.get("corrPairs") or []
    if not c.get("hasCorr"):
        return {
            "status": OPEN,
            "evidence": "No pairwise correlations were supplied in this profile, so redundancy has not been screened.",
            "closes": "The derived member removed, the measured one kept.",
        }
    near = [p for p in pairs if abs(float(p.get("r") or 0)) >= 0.95]
    if near:
        sample = "; ".join(f"{p['a']} × {p['b']}" for p in near[:2])
        return {
            "status": OPEN,
            "evidence": f"{len(near)} {_plural(len(near), 'pair')} correlate above |0.95| ({sample}).",
            "closes": "The derived member removed, the measured one kept.",
        }
    strongest = pairs[0]["r"] if pairs else 0
    return {
        "status": CLEAR,
        "evidence": f"No pair correlates above |0.95|; strongest is {float(strongest):.3f}.",
        "closes": "The derived member removed, the measured one kept.",
    }


def _r02_4(c: dict[str, Any]) -> dict[str, str]:
    vif = c.get("vif") or []
    threshold = float(c.get("vifThreshold") or 5)
    if not vif:
        return {
            "status": NA,
            "evidence": "No eligible numeric feature set to compute VIF against.",
            "closes": "One member removed at a time, recomputed, until coefficients can be read.",
        }
    over = [v for v in vif if float(v.get("vif") or 0) >= threshold]
    if over:
        return {
            "status": OPEN,
            "evidence": (
                f"{len(over)} {_plural(len(over), 'feature')} above the {threshold:.1f} threshold, "
                f"led by {over[0]['name']} at {float(over[0]['vif']):.2f}."
            ),
            "closes": "One member removed at a time, recomputed, until coefficients can be read.",
        }
    return {
        "status": CLEAR,
        "evidence": (
            f"All features below the {threshold:.1f} threshold; "
            f"highest is {float(vif[0]['vif']):.2f}."
        ),
        "closes": "One member removed at a time, recomputed, until coefficients can be read.",
    }


def _r02_5(c: dict[str, Any]) -> dict[str, str]:
    pairs = c.get("corrPairs") or []
    mi = c.get("mi") or []
    if c.get("hasCorr"):
        return {
            "status": OPEN,
            "evidence": (
                f"Every relationship figure here is linear or monotone ({len(pairs)} pairs, "
                f"{len(mi)} MI estimates); a saturating or reversing shape would not appear."
            ),
            "closes": "Target mean per decile for each candidate feature, with the shape named.",
        }
    mi_bit = f", only {len(mi)} MI estimates" if mi else ""
    return {
        "status": OPEN,
        "evidence": f"No pairwise correlations were supplied in this profile{mi_bit}; shape has not been examined at all.",
        "closes": "Target mean per decile for each candidate feature, with the shape named.",
    }


def _r02_6(c: dict[str, Any]) -> dict[str, str]:
    cats = c.get("categorical") or []
    if cats:
        return {
            "status": OPEN,
            "evidence": (
                f"All figures are pooled across {_fmt_n(c['rows'])} rows; {len(cats)} categorical "
                f"{_plural(len(cats), 'column')} were available as stratifiers and none was used."
            ),
            "closes": "Each headline relationship recomputed within at least one plausible confounder.",
        }
    return {
        "status": NA,
        "evidence": "No categorical column is available to stratify by.",
        "closes": "Each headline relationship recomputed within at least one plausible confounder.",
    }


def _r02_7(c: dict[str, Any]) -> dict[str, str]:
    eligible = max(1, int(c.get("eligible") or 1))
    ratio = float(c["rows"]) / eligible
    return {
        "status": CLEAR if ratio >= 10 else OPEN,
        "evidence": (
            f"{_fmt_n(c['rows'])} rows over {c['eligible']} eligible features — about "
            f"{round(ratio)} rows per feature before encoding."
        ),
        "closes": "Above about ten after encoding, reached by removing redundancy and coarsening categories first.",
    }


def _r02_8(c: dict[str, Any]) -> dict[str, str]:
    numeric = c.get("numeric") or []
    if not numeric:
        return {
            "status": NA,
            "evidence": "No numeric columns, so no scaling decision arises.",
            "closes": "A named scaler fitted in-fold, or a recorded decision that the model does not need one.",
        }
    spans = []
    for n in c.get("profiledNumeric") or []:
        try:
            span = abs(float(n["max"]) - float(n["min"]))
        except (TypeError, ValueError, KeyError):
            continue
        if span > 0:
            spans.append(span)
    if spans:
        factor = max(spans) / max(min(spans), 1e-12)
        return {
            "status": HUMAN,
            "evidence": (
                f"Numeric ranges differ by a factor of about {_fmt_compact(factor)}; "
                "no scaler is declared."
            ),
            "closes": "A named scaler fitted in-fold, or a recorded decision that the model does not need one.",
        }
    return {
        "status": HUMAN,
        "evidence": (
            f"{len(numeric)} numeric {_plural(len(numeric), 'column')} present but unranged "
            "in this profile; no scaler is declared."
        ),
        "closes": "A named scaler fitted in-fold, or a recorded decision that the model does not need one.",
    }


def _r03_1(c: dict[str, Any]) -> dict[str, str]:
    if c.get("timeCol"):
        return {
            "status": OPEN,
            "evidence": f"{c['timeCol']['name']} orders the rows, so a random split would train on the future.",
            "closes": "A split drawn on the structure the data has, with the test set touched once.",
        }
    if c.get("groupCol"):
        return {
            "status": OPEN,
            "evidence": (
                f"{c['groupCol']['name']} repeats across {_fmt_n(c['rows'])} rows, "
                "so a row-level split would straddle entities."
            ),
            "closes": "A split drawn on the structure the data has, with the test set touched once.",
        }
    return {
        "status": HUMAN,
        "evidence": "No time or group structure is declared; row independence is assumed rather than verified.",
        "closes": "A split drawn on the structure the data has, with the test set touched once.",
    }


def _r03_2(c: dict[str, Any]) -> dict[str, str]:
    target = c.get("target")
    if not target:
        return {"status": NA, "evidence": "No target is declared.", "closes": "A declared target."}
    if target.get("classes"):
        small = min(int(k["count"]) for k in target["classes"])
        return {
            "status": OPEN,
            "evidence": (
                f"Smallest class holds {_fmt_n(small)} rows; a 20% unstratified test set "
                f"would carry about {_fmt_n(round(small * 0.2))}."
            ),
            "closes": "Stratified split with the resulting shares verified per membership.",
        }
    return {
        "status": OPEN,
        "evidence": f"{target['name']} is continuous, so stratification would need binning.",
        "closes": "Stratified split with the resulting shares verified per membership.",
    }


def _r03_3(c: dict[str, Any]) -> dict[str, str]:
    anomaly = ", anomaly scores" if c.get("anomalies") else ""
    return {
        "status": OPEN,
        "evidence": (
            f"Every statistic here — medians, correlations, VIF, MI{anomaly} — "
            f"was computed on the full {_fmt_n(c['rows'])} rows."
        ),
        "closes": "A pipeline object below the split that the cross-validator refits per fold.",
    }


def _r03_4(c: dict[str, Any]) -> dict[str, str]:
    leakage = c.get("leakage") or []
    if leakage:
        return {
            "status": OPEN,
            "evidence": (
                f"{len(leakage)} {_plural(len(leakage), 'suspect')} flagged by heuristic: "
                f"{_list_names(leakage, 3)}."
            ),
            "closes": "A known-at time per column and every fitted step below the split.",
        }
    id_bit = (
        f", though {_list_names(c.get('idLike') or [], 2)} must stay out of the matrix"
        if c.get("idLike")
        else ""
    )
    return {
        "status": HUMAN,
        "evidence": f"No suspect flagged{id_bit}; column timing was not verified.",
        "closes": "A known-at time per column and every fitted step below the split.",
    }


def _r03_5(c: dict[str, Any]) -> dict[str, str]:
    time_col = c.get("timeCol")
    if time_col:
        gaps = time_col.get("gaps")
        gap_bit = (
            f" with {_fmt_n(gaps)} coverage {_plural(int(gaps), 'gap')}" if gaps else ""
        )
        return {
            "status": OPEN,
            "evidence": f"{time_col['name']} spans {time_col.get('min')} to {time_col.get('max')}{gap_bit}.",
            "closes": "A forward split with a horizon-length gap, and every window closed before the prediction moment.",
        }
    return {
        "status": NA,
        "evidence": "No time column is declared in this frame.",
        "closes": "A forward split with a horizon-length gap, and every window closed before the prediction moment.",
    }


def _r03_6(c: dict[str, Any]) -> dict[str, str]:
    group = c.get("groupCol")
    if group:
        groups = max(1, int(group.get("groups") or 1))
        return {
            "status": OPEN,
            "evidence": (
                f"{group['name']} identifies {_fmt_n(groups)} groups — about "
                f"{(c['rows'] / groups):.1f} rows each."
            ),
            "closes": "A group-aware split and cross-validation, or a recorded finding that rows are independent.",
        }
    id_bit = (
        f"; {_list_names(c.get('idLike') or [], 2)} looks near-unique in this extract"
        if c.get("idLike")
        else ""
    )
    return {
        "status": HUMAN,
        "evidence": f"No group column is declared{id_bit}.",
        "closes": "A group-aware split and cross-validation, or a recorded finding that rows are independent.",
    }


def _r03_7(c: dict[str, Any]) -> dict[str, str]:
    thin = ", thin enough to prefer nested cross-validation" if c["rows"] < 3000 else ""
    return {
        "status": HUMAN,
        "evidence": (
            f"{_fmt_n(c['rows'])} rows would leave about {_fmt_n(round(c['rows'] * 0.2))} "
            f"per 20% slice{thin}."
        ),
        "closes": "Nested CV or a three-way split, with the reported number from rows that influenced nothing.",
    }


def _r03_8(c: dict[str, Any]) -> dict[str, str]:
    import math

    se = 1.96 * math.sqrt(0.25 / max(1, c["rows"])) * 100
    return {
        "status": OPEN if c["rows"] < 1000 else HUMAN,
        "evidence": (
            f"At {_fmt_n(c['rows'])} rows a proportion metric carries roughly "
            f"±{se:.1f} points before splitting."
        ),
        "closes": "The smallest actionable effect stated, and the interval width compared against it.",
    }


def _r03_9(c: dict[str, Any]) -> dict[str, str]:
    tests = len(c.get("corrPairs") or []) + len(c.get("mi") or [])
    return {
        "status": OPEN if tests > 20 else CLEAR,
        "evidence": (
            f"{tests} screening {_plural(tests, 'statistic')} were computed across "
            f"{c['colCount']} columns with no multiplicity correction."
        ),
        "closes": "A false-discovery correction on reported p-values, and survivors re-checked out of sample.",
    }


def _r03_10(c: dict[str, Any]) -> dict[str, str]:
    ds = c.get("ds") or {}
    sampled = " under sampling" if c.get("sampled") else ""
    return {
        "status": OPEN,
        "evidence": (
            f"Run on {ds.get('engine')} {ds.get('version')} over {_fmt_n(c['rows'])} of "
            f"{_fmt_n(c['rowsTotal'])} rows{sampled}; no seed is reported."
        ),
        "closes": "A seed per random step, a sensitivity check across seeds, and a pinned extract.",
    }


def _r03_11(c: dict[str, Any]) -> dict[str, str]:
    drifted = c.get("drifted") or []
    if drifted:
        return {
            "status": OPEN,
            "evidence": (
                f"{len(drifted)} {_plural(len(drifted), 'column')} met the configured thresholds: "
                f"{_list_names(drifted, 5)}."
            ),
            "closes": "The split, the ingestion and the population ruled out in that order, and the shift named.",
        }
    return {
        "status": CLEAR,
        "evidence": "No column met the configured thresholds — an absence of flags at your threshold.",
        "closes": "The split, the ingestion and the population ruled out in that order, and the shift named.",
    }


def _r03_12(c: dict[str, Any]) -> dict[str, str]:
    uni = sum(1 for n in (c.get("numeric") or []) if float(n.get("outlierRate") or 0) > 0)
    anomalies = c.get("anomalies")
    if anomalies:
        uni_bit = (
            f", plus {uni} {_plural(uni, 'column')} with IQR-fence values" if uni else ""
        )
        return {
            "status": OPEN,
            "evidence": (
                f"{_fmt_n(anomalies['flagged'])} of {_fmt_n(anomalies['scored'])} scored rows "
                f"marked at a {_fmt_pct(float(anomalies['contamination']), 0)} configured "
                f"contamination{uni_bit}."
            ),
            "closes": "Each flag classified as error, rare event, subpopulation or sentinel.",
        }
    if uni:
        return {
            "status": OPEN,
            "evidence": f"{uni} numeric {_plural(uni, 'column')} carry values beyond the IQR fences.",
            "closes": "Each flag classified as error, rare event, subpopulation or sentinel.",
        }
    return {
        "status": CLEAR,
        "evidence": "No univariate or multivariate outlier flags were raised.",
        "closes": "Each flag classified as error, rare event, subpopulation or sentinel.",
    }


def _r04_1(c: dict[str, Any]) -> dict[str, str]:
    target = c.get("target")
    if not target:
        return {"status": NA, "evidence": "No target is declared.", "closes": "One headline metric plus diagnostics, with the population and threshold stated."}
    return {
        "status": HUMAN,
        "evidence": f"{target['name']} is {target['task']}; no metric is declared in this session.",
        "closes": "One headline metric plus diagnostics, with the population and threshold stated.",
    }


def _r04_2(c: dict[str, Any]) -> dict[str, str]:
    target = c.get("target")
    if not target:
        return {"status": NA, "evidence": "No target is declared.", "closes": "Trivial and incumbent baselines scored on the same rows and the same metric."}
    if target.get("task") == "regression" and target.get("stats"):
        return {
            "status": OPEN,
            "evidence": (
                f"Predicting the median ({_fmt_compact(float(target['stats'].get('median') or 0))}) "
                "is the floor and has not been scored."
            ),
            "closes": "Trivial and incumbent baselines scored on the same rows and the same metric.",
        }
    if target.get("classes"):
        maj = max(int(k["count"]) for k in target["classes"]) / max(1, c["rows"])
        return {
            "status": OPEN,
            "evidence": f"Predicting the majority class scores {_fmt_pct(maj, 1)} and catches nothing.",
            "closes": "Trivial and incumbent baselines scored on the same rows and the same metric.",
        }
    return {
        "status": OPEN,
        "evidence": "Predicting the majority class scores the base rate and catches nothing.",
        "closes": "Trivial and incumbent baselines scored on the same rows and the same metric.",
    }


def _r04_3(c: dict[str, Any]) -> dict[str, str]:
    target = c.get("target")
    if not target or target.get("task") == "regression" or not target.get("classes"):
        return {
            "status": NA,
            "evidence": "No classification target in this frame.",
            "closes": "A ranking metric plus precision and recall at a stated threshold; class weights preferred to resampling.",
        }
    parts = ", ".join(
        f"{k['label']} {_fmt_pct(int(k['count']) / max(1, c['rows']), 1)}"
        for k in target["classes"]
    )
    return {
        "status": OPEN,
        "evidence": f"{parts} — accuracy is uninformative at this balance.",
        "closes": "A ranking metric plus precision and recall at a stated threshold; class weights preferred to resampling.",
    }


def _r04_4(c: dict[str, Any]) -> dict[str, str]:
    target = c.get("target")
    classes = (target or {}).get("classes") if target else None
    if not classes or len(classes) != 2:
        return {
            "status": NA,
            "evidence": "No binary target, so no single cut arises.",
            "closes": "A cut derived from C_FP/(C_FP+C_FN), tuned on validation and frozen.",
        }
    pos = int(classes[1]["count"]) / max(1, c["rows"])
    return {
        "status": OPEN,
        "evidence": (
            f"Positive rate {_fmt_pct(pos, 1)}; the default 0.5 cut assumes equal costs "
            "and an even base rate."
        ),
        "closes": "A cut derived from C_FP/(C_FP+C_FN), tuned on validation and frozen.",
    }


def _r04_5(c: dict[str, Any]) -> dict[str, str]:
    complete = c.get("completeRows", c["rows"])
    complete_bit = (
        f" ({_fmt_n(complete)} complete cases for some)" if complete != c["rows"] else ""
    )
    return {
        "status": OPEN,
        "evidence": f"Every figure on the sheet is a point estimate from {_fmt_n(c['rows'])} rows{complete_bit}.",
        "closes": "Bootstrap intervals at the level of independence, and rounding that matches them.",
    }


def _r04_6(c: dict[str, Any]) -> dict[str, str]:
    cats = c.get("categorical") or []
    if cats:
        time_bit = ", plus periods" if c.get("timeCol") else ""
        return {
            "status": OPEN,
            "evidence": (
                f"{len(cats)} categorical {_plural(len(cats), 'column')} could define slices"
                f"{time_bit}; all figures here are pooled."
            ),
            "closes": "Predefined slices with support and intervals, and a named response to every material gap.",
        }
    return {
        "status": HUMAN,
        "evidence": "No categorical column is available to slice by.",
        "closes": "Predefined slices with support and intervals, and a named response to every material gap.",
    }


def _r04_7(c: dict[str, Any]) -> dict[str, str]:
    target = c.get("target")
    if not target or target.get("task") == "regression" or not target.get("classes"):
        return {
            "status": NA,
            "evidence": "No probabilistic classification target here.",
            "closes": "Either a mark for this session that ranking suffices, or a reliability curve on held-out data.",
        }
    return {
        "status": HUMAN,
        "evidence": (
            f"{target['name']} would produce scores; whether they are used as probabilities "
            "is a product decision, not a data one."
        ),
        "closes": "Either a mark for this session that ranking suffices, or a reliability curve on held-out data.",
    }


def _r05_1(c: dict[str, Any]) -> dict[str, str]:
    hot = [p for p in (c.get("corrPairs") or []) if abs(float(p.get("r") or 0)) >= 0.8]
    hot_bit = (
        f", of which {len(hot)} correlated {_plural(len(hot), 'pair')} will share credit"
        if hot
        else ""
    )
    return {
        "status": OPEN,
        "evidence": (
            f"{c['eligible']} eligible {_plural(int(c['eligible'] or 0), 'feature')} "
            f"would enter the calculation{hot_bit}."
        ),
        "closes": "Permutation importance on held-out rows, repeated, with redundant groups noted.",
    }


def _r05_2(c: dict[str, Any]) -> dict[str, str]:
    numeric = c.get("numeric") or []
    if not numeric:
        return {
            "status": NA,
            "evidence": "No numeric features to sweep.",
            "closes": "Partial-dependence and ICE curves clipped to the data's own percentiles, with density shown.",
        }
    skewed = c.get("skewed") or []
    skew_bit = (
        f", {len(skewed)} of them skewed enough to narrow the supported range sharply"
        if skewed
        else ""
    )
    return {
        "status": OPEN,
        "evidence": f"{len(numeric)} numeric {_plural(len(numeric), 'feature')} could be swept{skew_bit}.",
        "closes": "Partial-dependence and ICE curves clipped to the data's own percentiles, with density shown.",
    }


def _r05_3(c: dict[str, Any]) -> dict[str, str]:
    eligible = max(1, int(c.get("eligible") or 1))
    ratio = c["rows"] / eligible
    regime = " — the regime where variance usually dominates" if ratio < 20 else ""
    return {
        "status": OPEN,
        "evidence": f"About {round(ratio)} rows per feature{regime}; no curve has been drawn.",
        "closes": "A learning curve with fold error bars, read before any hyper-parameter search.",
    }


def _r05_4(c: dict[str, Any]) -> dict[str, str]:
    pairs = c.get("corrPairs") or []
    mi = c.get("mi") or []
    mi_bit = f" and {len(mi)} MI {_plural(len(mi), 'estimate')}" if mi else ""
    return {
        "status": OPEN,
        "evidence": (
            f"{len(pairs)} correlation {_plural(len(pairs), 'pair')}{mi_bit} from observational "
            "rows; no assignment mechanism is recorded."
        ),
        "closes": "Associations reported as associations, with the confounders that would need measuring listed.",
    }


def _r05_5(c: dict[str, Any]) -> dict[str, str]:
    parts = [f"VIF {float(c.get('vifThreshold') or 5):.1f}"]
    if c.get("anomalies"):
        parts.append(
            f"contamination {_fmt_pct(float(c['anomalies']['contamination']), 0)}"
        )
    return {
        "status": OPEN,
        "evidence": (
            f"Configured thresholds include {' and '.join(parts)}; "
            "no owner or review date is attached."
        ),
        "closes": "Assumptions ledger, decision log, pinned extract and a monitoring plan with named owners.",
    }


_GATE_DEFS: tuple[_GateDef, ...] = (
    _gate("00.1", 0, "Is there one written sentence saying who acts on this model’s output, and when?", "problem-framing", _r00_1),
    _gate("00.2", 0, "Is it written down what one row represents, and does a key prove it is unique?", "unit-of-analysis", _r00_2),
    _gate("00.3", 0, "Do we know which rows the extract filtered out before we saw it?", "population-and-sampling-frame", _r00_3),
    _gate("00.4", 0, "Is the label’s exact rule recorded — what counts as positive, measured from when, over how long?", "target-definition", _r00_4),
    _gate("00.5", 0, "For each column, do we know its source system and whether its value exists at prediction time?", "provenance-and-lineage", _r00_5),
    _gate("00.6", 0, "Has someone listed which columns are personal or legally protected?", "sensitive-attributes", _r00_6),
    _gate("01.1", 1, "Was each column’s type set deliberately, rather than guessed by the CSV loader?", "dtypes-and-storage", _r01_1),
    _gate("01.2", 1, "Does each column with gaps have a chosen fill strategy, fitted on training rows only?", "missing-data", _r01_2),
    _gate("01.3", 1, "For each gappy column, do we know whether the gaps are random or systematic?", "missingness-mechanisms", _r01_3),
    _gate("01.4", 1, "Have exact duplicate rows and repeated keys been counted and resolved?", "duplicate-records", _r01_4),
    _gate("01.5", 1, "Have columns with one dominant value been dropped, coarsened, or knowingly kept?", "constant-and-near-constant", _r01_5),
    _gate("01.6", 1, "Does each category column with many levels have an encoding plan and an unseen-level policy?", "high-cardinality", _r01_6),
    _gate("01.7", 1, "Does each numeric column have an allowed min/max, with codes like -999 turned into missing?", "measurement-units-and-ranges", _r01_7),
    _gate("01.8", 1, "Were text columns trimmed and case-folded before their category levels were counted?", "text-hygiene", _r01_8),
    _gate("01.9", 1, "For every join that built this table, do we know it added or dropped no rows unexpectedly?", "join-integrity", _r01_9),
    _gate("01.10", 1, "Are contradictions between columns tested — end before start, parts not summing to total?", "cross-field-consistency", _r01_10),
    _gate("01.11", 1, "Were date columns parsed with an explicit format and timezone, and does the span look right?", "datetime-parsing", _r01_11),
    _gate("01.12", 1, "Do we know how coarsely each number was recorded, and whether values pile up at a cap?", "precision-and-heaping", _r01_12),
    _gate("02.1", 2, "Has each numeric column been read as a distribution — quartiles and histogram — not just a mean?", "univariate-distributions", _r02_1),
    _gate("02.2", 2, "For each skewed column, is there a decision to transform it or not, and why?", "skew-and-transforms", _r02_2),
    _gate("02.3", 2, "Have columns that are just re-expressions of other columns been removed?", "derived-and-redundant-columns", _r02_3),
    _gate("02.4", 2, "Are features independent enough that a model’s coefficients can be trusted?", "variance-inflation", _r02_4),
    _gate("02.5", 2, "Has each feature’s relationship to the target been checked for curves and reversals, not just straight lines?", "non-linearity-and-binning", _r02_5),
    _gate("02.6", 2, "Was each headline relationship re-checked inside subgroups, in case it reverses?", "confounding-and-subgroups", _r02_6),
    _gate("02.7", 2, "Are there enough rows per feature — after encoding — for a model to learn rather than memorise?", "sparsity-and-dimensionality", _r02_7),
    _gate("02.8", 2, "Is there a recorded decision on whether to scale features, based on the model chosen?", "feature-scaling", _r02_8),
    _gate("03.1", 3, "Does the train/test split respect time order and repeated entities, rather than splitting at random?", "data-splitting", _r03_1),
    _gate("03.2", 3, "Will each split keep the target’s class balance, and was that verified after splitting?", "stratification", _r03_2),
    _gate("03.3", 3, "Is every step that learns from data — imputer, encoder, scaler — fitted after the split, not before?", "pipeline-order", _r03_3),
    _gate("03.4", 3, "Has every column been confirmed knowable at prediction time, with no post-outcome values?", "leakage", _r03_4),
    _gate("03.5", 3, "Do the split and every window feature look only backwards in time?", "temporal-structure", _r03_5),
    _gate("03.6", 3, "If rows repeat the same entity, does the split keep that entity on one side?", "group-structure", _r03_6),
    _gate("03.7", 3, "Are the rows used to pick the model different from the rows used to report its score?", "nested-validation", _r03_7),
    _gate("03.8", 3, "Is this sample large enough to detect a difference small enough to matter?", "sample-size-and-power", _r03_8),
    _gate("03.9", 3, "Given how many statistics were screened, are the strongest results corrected for chance?", "multiple-comparisons", _r03_9),
    _gate("03.10", 3, "Could someone else re-run this and get the same numbers — seed, library versions, data snapshot?", "reproducibility", _r03_10),
    _gate("03.11", 3, "For each drift flag, was the split and the pipeline ruled out before blaming the data?", "dataset-drift", _r03_11),
    _gate("03.12", 3, "Is each outlier explained as an error, a rare true event, a subgroup, or a sentinel code?", "outlier-screens", _r03_12),
    _gate("04.1", 4, "Was the scoring metric written down before the first model was fitted?", "metric-selection", _r04_1),
    _gate("04.2", 4, "Do we know what the dumbest predictor, and today’s existing process, would score?", "baselines", _r04_2),
    _gate("04.3", 4, "Given the class balance, is the metric one that a majority-class guess cannot win?", "class-imbalance", _r04_3),
    _gate("04.4", 4, "Does the cut-off come from the relative cost of a false alarm versus a miss, not from 0.5?", "thresholds-and-costs", _r04_4),
    _gate("04.5", 4, "Does every number someone will act on carry an uncertainty range?", "uncertainty-intervals", _r04_5),
    _gate("04.6", 4, "Will performance be reported per segment, not only as one overall figure?", "slice-evaluation", _r04_6),
    _gate("04.7", 4, "If the output is used as a probability, has it been checked against observed rates?", "calibration", _r04_7),
    _gate("05.1", 5, "Is the feature-importance method named, run on held-out rows, and reported with its variability?", "feature-importance-methods", _r05_1),
    _gate("05.2", 5, "Are feature-effect curves drawn only across the range where rows actually exist?", "effect-shapes", _r05_2),
    _gate("05.3", 5, "Do we know whether more data or better features is the lever, before spending on either?", "learning-curves-and-capacity", _r05_3),
    _gate("05.4", 5, "Is every finding stated as an association, without implying that changing the feature changes the outcome?", "causal-caution", _r05_4),
    _gate("05.5", 5, "Do the assumptions, chosen thresholds and monitoring owners exist as a written handoff?", "handoff-and-monitoring", _r05_5),
)
