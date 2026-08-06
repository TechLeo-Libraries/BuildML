"""Concept Academy learning-hub: full CONCEPT_NOTES coverage, adaptivity, examples."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.academy import build_academy_payload
from buildml.dashboard.academy_curriculum import (
    all_lessons,
    catalog_concept_count,
    curriculum_slugs,
    readiness_slugs,
)
from buildml.explain.concepts import CONCEPT_NOTES

pytest.importorskip("fastapi")

# Exact catalog inventory the Academy must teach at full depth.
EXPECTED_CATALOG_COUNT = 204

# Redesign pack ~67 concepts (must all be present as curriculum slugs).
REDESIGN_SLUGS = {
    # 00
    "problem-framing",
    "unit-of-analysis",
    "population-and-sampling-frame",
    "target-definition",
    "provenance-and-lineage",
    "sensitive-attributes",
    # 01
    "column-roles",
    "dtypes-and-storage",
    "missing-data",
    "missingness-mechanisms",
    "duplicate-records",
    "constant-and-near-constant",
    "high-cardinality",
    "categorical-encoding",
    "measurement-units-and-ranges",
    "text-hygiene",
    "join-integrity",
    "cross-field-consistency",
    "datetime-parsing",
    "precision-and-heaping",
    "nested-and-multivalued",
    # 02
    "univariate-distributions",
    "skew-and-transforms",
    "correlation",
    "mutual-information",
    "variance-inflation",
    "interaction-effects",
    "dimensionality-reduction",
    "feature-scaling",
    "categorical-association",
    "non-linearity-and-binning",
    "confounding-and-subgroups",
    "derived-and-redundant-columns",
    "time-feature-engineering",
    "sparsity-and-dimensionality",
    # 03
    "data-splitting",
    "stratification",
    "cross-validation",
    "dataset-drift",
    "leakage",
    "temporal-structure",
    "group-structure",
    "diagnostic-uncertainty",
    "outlier-screens",
    "pipeline-order",
    "nested-validation",
    "sample-size-and-power",
    "multiple-comparisons",
    "reproducibility",
    "shift-taxonomy",
    # 04
    "class-imbalance",
    "target-distribution",
    "metric-selection",
    "thresholds-and-costs",
    "baselines",
    "calibration",
    "confusion-matrix",
    "ranking-curves",
    "multiclass-and-averaging",
    "residual-diagnostics",
    "uncertainty-intervals",
    "slice-evaluation",
    # 05
    "feature-importance-methods",
    "effect-shapes",
    "learning-curves-and-capacity",
    "causal-caution",
    "handoff-and-monitoring",
}

GAP_SLUGS = {
    "batch-leakage",
    "evaluation-partitions",
    "early-stopping-partition",
    "overfitting",
    "feature-selection",
    "model-selection",
    "mi-vs-correlation",
    "train-serve-parity",
}

DEMO_HARDCODE_FORBIDDEN = (
    "churn rate",
    "ops-delay",
    "telecom churn",
    "this churn dataset",
    "customer_id_demo",
)

REQUIRED_SECTION_KEYS = (
    "what_it_means",
    "technical_depth",
    "why_it_matters",
    "calculation",
    "worked_example",
    "pitfalls",
    "evidence",
    "how_to_read",
    "decide",
)


def _classification_session() -> Session:
    frame = pd.DataFrame(
        {
            "age": [21, 25, 30, 35, 40, 45, None, 55, 60, 22, 28, 33, 41, 50],
            "income": [40, 55, 60, 80, 50, 70, 65, 90, 95, 42, 48, 58, 77, 88],
            "city": ["a", "b", "a", "b", "a", "b", "a", "a", "b", "a", "b", "a", "b", "a"],
            "const": [1] * 14,
            "id_like": list(range(14)),
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    return (
        Session.ingest(frame)
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "city": "feature",
                "y": "target",
                "id_like": "id",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=0)
    )


def _regression_session() -> Session:
    n = 40
    frame = pd.DataFrame(
        {
            "sqft": [800 + 35 * i for i in range(n)],
            "beds": [1 + (i % 5) for i in range(n)],
            "price": [100.5 + 7.3 * i + (i % 3) * 0.25 for i in range(n)],
        }
    )
    return (
        Session.ingest(frame)
        .set_roles({"sqft": "feature", "beds": "feature", "price": "target"})
        .split(test_size=0.25, random_state=0)
    )


def test_catalog_inventory_is_204() -> None:
    assert len(CONCEPT_NOTES) == EXPECTED_CATALOG_COUNT
    assert catalog_concept_count() == EXPECTED_CATALOG_COUNT


def test_curriculum_covers_redesign_gaps_and_full_catalog() -> None:
    slugs = curriculum_slugs()
    assert REDESIGN_SLUGS.issubset(slugs)
    assert GAP_SLUGS.issubset(slugs)
    assert set(CONCEPT_NOTES).issubset(slugs)
    assert len(readiness_slugs()) >= 67 + len(GAP_SLUGS)
    assert len(all_lessons()) == len(slugs)
    # Union of redesign spine + full catalog (no thin extended tier).
    assert len(slugs) >= EXPECTED_CATALOG_COUNT
    assert len(slugs) == EXPECTED_CATALOG_COUNT + len(slugs - set(CONCEPT_NOTES))


def test_academy_payload_covers_catalog_at_full_depth() -> None:
    report = _classification_session().eda(include_plots=False, show=False).to_dict()
    academy = build_academy_payload(report)

    assert academy["catalog_count"] == EXPECTED_CATALOG_COUNT
    assert academy["catalog_covered"] == EXPECTED_CATALOG_COUNT
    assert academy["concept_count"] >= EXPECTED_CATALOG_COUNT
    assert academy["curriculum_count"] == academy["concept_count"]
    assert academy["extended_count"] == 0
    assert academy["adaptivity"]["target"] == "y"
    assert academy["adaptivity"]["task"] == "classification"
    assert academy["stages"]

    by_slug = {c["slug"]: c for c in academy["concepts"]}
    # Every CONCEPT_NOTES key is a first-class curriculum lesson.
    for key in CONCEPT_NOTES:
        item = by_slug[key]
        assert item["curriculum"] is True
        assert item.get("catalog") is True
        sections = item["sections"]
        for section_key in REQUIRED_SECTION_KEYS:
            assert section_key in sections
        assert sections["what_it_means"]
        assert sections["technical_depth"]
        assert sections["calculation"]["walkthrough"]
        code = sections["worked_example"]["code"]
        assert code.strip()
        assert "TODO" not in code
        assert "Session" in code or "session" in code
        assert sections["worked_example"]["what_to_change"]
        assert sections["evidence"]["session"]
        assert sections["how_to_read"]
        assert sections["decide"]
        assert "extended" not in (item.get("tags") or [])


def test_academy_payload_richness_and_real_session_api() -> None:
    report = _classification_session().eda(include_plots=False, show=False).to_dict()
    academy = build_academy_payload(report)

    missing = next(c for c in academy["concepts"] if c["slug"] == "missing-data")
    sections = missing["sections"]
    assert sections["what_it_means"]
    assert sections["calculation"]["walkthrough"]
    assert "completeness" in sections["calculation"]["walkthrough"].lower() or "%" in sections[
        "calculation"
    ]["walkthrough"]
    code = sections["worked_example"]["code"]
    assert "Session.ingest" in code or "session =" in code
    assert "session.split" in code or ".split(" in code
    assert "impute" in code
    assert "<--" in code or "change" in code.lower()
    assert sections["worked_example"]["what_to_change"]
    assert sections["evidence"]["session"]
    assert "y" in sections["evidence"]["session"] or "missing" in sections["evidence"]["session"].lower()

    blob = " ".join(
        [
            missing["session"],
            missing["example"],
            missing.get("calculation") or "",
            " ".join(missing.get("prose") or []),
        ]
    ).lower()
    for forbidden in DEMO_HARDCODE_FORBIDDEN:
        assert forbidden not in blob


def test_academy_adapts_classification_vs_regression() -> None:
    class_report = _classification_session().eda(include_plots=False, show=False).to_dict()
    reg_report = _regression_session().eda(include_plots=False, show=False).to_dict()
    class_academy = build_academy_payload(class_report)
    reg_academy = build_academy_payload(reg_report)

    assert class_academy["adaptivity"]["task"] == "classification"
    assert reg_academy["adaptivity"]["task"] == "regression"
    assert class_academy["adaptivity"]["target"] == "y"
    assert reg_academy["adaptivity"]["target"] == "price"

    class_imb = next(c for c in class_academy["concepts"] if c["slug"] == "class-imbalance")
    reg_imb = next(c for c in reg_academy["concepts"] if c["slug"] == "class-imbalance")
    assert "classification" in class_imb["session"].lower() or "%" in class_imb["session"]
    assert "not a classification" in reg_imb["session"].lower() or "regression" in reg_imb[
        "session"
    ].lower() or "n/a" in reg_imb["session"].lower()

    roles = next(c for c in class_academy["concepts"] if c["slug"] == "column-roles")
    assert '"y": "target"' in roles["example"]
    reg_roles = next(c for c in reg_academy["concepts"] if c["slug"] == "column-roles")
    assert '"price": "target"' in reg_roles["example"]


def test_academy_lessons_have_no_todo_stubs() -> None:
    report = _classification_session().eda(include_plots=False, show=False).to_dict()
    academy = build_academy_payload(report)
    thin_markers = (
        "Extended reference — not part of the core EDA readiness path",
        "extended library",
    )
    for item in academy["concepts"]:
        code = item["sections"]["worked_example"]["code"]
        assert code.strip()
        assert "TODO" not in code
        assert "pass  #" not in code
        assert item["sections"]["what_it_means"]
        assert item["sections"]["evidence"]["session"]
        assert item["sections"]["worked_example"]["what_to_change"]
        session = item["sections"]["evidence"]["session"]
        for marker in thin_markers:
            assert marker not in session


def test_academy_view_module_wired() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "buildml" / "dashboard" / "static" / "js"
    app_js = (root / "app.js").read_text(encoding="utf-8")
    view_js = (root / "academy_view.js").read_text(encoding="utf-8")
    assert 'from "./academy_view.js"' in app_js
    assert "export async function renderAcademy" in view_js
    assert 'from "./learn_ui.js"' in view_js
    assert "wireLearnUi" in view_js
    assert "calcBlock" in view_js
    assert "codeBlock" in view_js
    assert "What it means" in view_js
    assert "Open lesson" in view_js
    assert "extended library" not in view_js.lower()
    assert 'data-academy-mode' in view_js
    assert "catalog" in view_js
