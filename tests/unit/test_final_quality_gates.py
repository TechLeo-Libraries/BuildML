from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.eda.html_report import export_eda_html
from buildml.explain import OPERATION_CATALOG
from buildml.reporting import render_table

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_reports"
SECTION_IDS = (
    "findings-register",
    "assumptions",
    "ledger",
    "recommended-sequence",
    "figures",
    "methods",
    "degraded",
    "appendix",
)


def _case(name: str) -> tuple[Session, dict[str, object]]:
    if name == "dirty_classification":
        frame = pd.DataFrame(
            {
                "age<unsafe>": [20, 22, None, 35, 35, 42] * 8,
                "income": [30, 40, 50, 80, 80, 120] * 8,
                "city": ["north", "south", "north", "", "", "west"] * 8,
                "constant": ["same"] * 48,
                "customer_id": [f"customer-{index}" for index in range(48)],
                "target": [0, 1, 0, 1, 0, 1] * 8,
            }
        )
        session = Session.ingest(frame).set_roles(
            {
                "age<unsafe>": "feature",
                "income": "feature",
                "city": "feature",
                "constant": "feature",
                "customer_id": "id",
                "target": "target",
            }
        )
        session.split(test_size=0.25, stratify=True, random_state=7)
        return session, {"sample_rows": 24, "max_columns": 20, "max_plots": 0}
    if name == "regression":
        frame = pd.DataFrame(
            {
                "x": list(range(40)),
                "x2": [value * 1.5 + value % 3 for value in range(40)],
                "group": ["a", "b", "c", "a"] * 10,
                "target": [value * 2.25 + (value % 4) for value in range(40)],
            }
        )
        session = Session.ingest(frame).set_roles(
            {"x": "feature", "x2": "feature", "group": "feature", "target": "target"}
        )
        return session, {"max_columns": 20, "max_plots": 0}
    if name == "high_cardinality":
        frame = pd.DataFrame(
            {
                "category": [f"level-{index}" for index in range(120)],
                "value": [index % 17 for index in range(120)],
                "target": [index % 2 for index in range(120)],
            }
        )
        session = Session.ingest(frame).set_roles(
            {"category": "feature", "value": "feature", "target": "target"}
        )
        return session, {"max_columns": 20, "max_plots": 0}
    if name == "drift":
        frame = pd.DataFrame(
            {
                "shifted": [float(index % 7) for index in range(60)]
                + [100.0 + float(index % 7) for index in range(20)],
                "stable": [index % 3 for index in range(80)],
                "target": [index % 2 for index in range(80)],
            }
        )
        session = Session.ingest(frame).set_roles(
            {"shifted": "feature", "stable": "feature", "target": "target"}
        )
        session.inject_split(train_indices=list(range(60)), test_indices=list(range(60, 80)))
        return session, {"max_columns": 20, "max_plots": 0}
    if name == "degraded":
        return Session.ingest(pd.DataFrame({"label": ["a", "b", None]})), {
            "max_columns": 5,
            "max_plots": 0,
        }
    raise AssertionError(name)


def _normalized_snapshot(name: str, report: object, html: str) -> dict[str, object]:
    payload = report.to_dict()
    quality = payload["quality"]
    target = payload["target"]
    drift = payload["drift"]
    recommendations = payload["recommendation_details"]
    return {
        "case": name,
        "shape": [
            payload["overview"]["n_rows"],
            payload["overview"]["n_columns"],
        ],
        "analysis_rows": payload["overview"]["analysis_rows"],
        "analysis_column_count": payload["overview"]["analysis_column_count"],
        "target_type": (target.get("summary") or {}).get("type"),
        "quality_flags": {
            key: sorted(quality.get(key) or [])
            for key in (
                "constant_columns",
                "high_cardinality_columns",
                "id_like_columns",
                "mixed_type_suspect_columns",
            )
        },
        "drift_available": bool(drift.get("available")),
        "drift_flag_count": len(drift.get("flagged_columns") or []),
        "recommendations": [
            {"key": item["key"], "based_on": item["based_on"]}
            for item in recommendations
        ],
        "section_ids": re.findall(r'<section id="([^"]+)"', html),
        "has_sampling_disclosure": "Heavy EDA sections used a sample" in html,
        "has_skipped_disclosure": "Skipped and degraded analyses" in html,
    }


@pytest.mark.parametrize(
    "name",
    ["dirty_classification", "regression", "high_cardinality", "drift", "degraded"],
)
def test_compact_golden_reports(name: str, tmp_path: Path) -> None:
    session, options = _case(name)
    destination = tmp_path / f"{name}.html"
    report = session.eda(export_html=destination, html_format="research", **options)
    actual = _normalized_snapshot(name, report, destination.read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
    assert actual == expected


class _TinyFigure:
    def savefig(self, stream: object, **_: object) -> None:
        stream.write(b"\x89PNG\r\n\x1a\ncompact-fixture")


def test_eda_html_contract_is_cockpit_accessible_escaped_and_offline(
    tmp_path: Path,
) -> None:
    session, options = _case("dirty_classification")
    report = session.eda(**options)
    destination = tmp_path / "contract.html"
    export_eda_html(
        report.to_dict(),
        destination,
        title='Unsafe <title> & "quote"',
        figures={"compact": _TinyFigure()},
        max_figures=1,
    )
    html = destination.read_text(encoding="utf-8")

    assert tuple(re.findall(r'<section id="([^"]+)"', html)) == SECTION_IDS
    for marker in (
        'role="banner"',
        'role="search"',
        'aria-label="Search report sections"',
        '<main id="main-content" tabindex="-1">',
        'role="contentinfo"',
        'aria-live="polite"',
        'aria-controls="main-content"',
        "<details>",
        "bml-section-search",
        "bml-table-search",
        'addEventListener("input"',
        'addEventListener("keydown"',
        "rows.sort",
        "--color-accent: #5980a6",
        "--color-bg: #f2f2f3",
        "blueprint",
        "Findings register",
        "Ledger — every computed number",
        "Recommended sequence",
        "What each finding assumes",
        "What this means",
        "Why it matters",
        "What to check next",
        "om-header__title",
        "Offline HTML",
        'id="bml-offline-html"',
        "@media print",
        "data:image/png;base64,",
        "Skipped and degraded analyses",
        "om-methods__grid",
        "om-callout",
        "om-assumption-card",
        "om-ledger-group",
        "om-search-field",
        "Type to filter sections by title or keywords",
        'data-filter-target="assumptions"',
        'data-filter-target="ledger"',
        "Methods and limitations",
        "Caveats",
    ):
        assert marker in html
    assert "Static EDA — readiness sheet" not in html
    assert "PDF briefing" not in html
    assert 'id="bml-csv"' not in html
    assert "bml-csv-payload" not in html

    default_destination = tmp_path / "default-title.html"
    export_eda_html(report.to_dict(), default_destination, max_figures=0)
    default_html = default_destination.read_text(encoding="utf-8")
    assert "BUILDML STATIC EDA" in default_html
    assert "<h1>BUILDML STATIC EDA</h1>" in default_html
    assert "Static EDA — readiness sheet" not in default_html
    assert "Readiness Gates" not in html
    assert "Concept Academy" not in html
    assert "Unsafe &lt;title&gt; &amp; &quot;quote&quot;" in html
    assert "age&lt;unsafe&gt;" in html
    assert not re.search(r'(?:src|href)=["\']https?://', html, flags=re.IGNORECASE)


def test_report_budgets_bound_wide_output_and_disclose_truncation(
    tmp_path: Path,
) -> None:
    columns = {f"feature_{index:03d}": [index + row for row in range(60)] for index in range(140)}
    columns["target"] = [row % 2 for row in range(60)]
    session = Session.ingest(pd.DataFrame(columns)).set_roles({"target": "target"})
    destination = tmp_path / "wide.html"
    report = session.eda(
        sample_rows=30,
        max_columns=12,
        max_plots=0,
        export_html=destination,
        html_format="research",
    )
    html = destination.read_text(encoding="utf-8")

    assert report.overview["analysis_rows"] == 30
    assert report.overview["analysis_column_count"] == 12
    assert "target" in report.overview["analysis_columns"]
    assert "Column budget limited detailed analyzers" in html
    assert len(html.encode("utf-8")) < 2_000_000

    bounded = render_table(
        [{f"column_{column}": row for column in range(20)} for row in range(100)],
        max_rows=7,
        max_columns=5,
    )
    assert bounded.count("<tr>") == 8
    assert bounded.count('<th scope="col">') == 5
    assert "93 additional rows and 15 additional columns" in bounded

    payload = report.to_dict()
    payload["raw_appendix_stress"] = "x" * 300_000
    size_bounded = tmp_path / "size-bounded.html"
    export_eda_html(
        payload,
        size_bounded,
        max_figures=0,
        max_html_bytes=250_000,
    )
    size_html = size_bounded.read_text(encoding="utf-8")
    assert "Raw appendix omitted by output budget" in size_html
    assert len(size_html.encode("utf-8")) <= 250_000


def test_limited_dependency_report_records_degraded_visuals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import buildml.eda.visualize as visualize

    def unavailable(*_: object, **__: object) -> dict[str, object]:
        raise MissingExtraError("viz", "EDA visualization")

    monkeypatch.setattr(visualize, "render_adaptive_plots", unavailable)
    destination = tmp_path / "limited-dependency.html"
    report = Session.ingest(pd.DataFrame({"value": [1, 2, 3]})).eda(
        include_plots=True,
        export_html=destination,
        html_format="research",
    )
    html = destination.read_text(encoding="utf-8")

    assert "visualization_unavailable" in report.figures
    assert "requires the optional extra" in html
    assert "Skipped and degraded analyses" in html


def test_catalog_parameters_match_public_session_signatures() -> None:
    for operation, spec in OPERATION_CATALOG.items():
        signature = inspect.signature(getattr(Session, operation))
        available = set(signature.parameters)
        documented = {parameter.name for parameter in spec.parameters}
        assert documented <= available, f"{operation}: {sorted(documented - available)}"
