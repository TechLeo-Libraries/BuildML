"""Documentation and user-copy contract tests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from buildml import Session
from scripts.lint_user_copy import COPY_RULES, STALE_API

ROOT = Path(__file__).resolve().parents[2]


def test_user_copy_lint_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/lint_user_copy.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_documented_v2_methods_exist() -> None:
    documented_methods = {
        "calibration",
        "checkpoint_load",
        "checkpoint_save",
        "eda",
        "evaluate",
        "explain",
        "feature_importance",
        "fit",
        "impute",
        "inject_split",
        "save_model",
        "scale",
        "set_roles",
        "split",
        "tune_threshold",
        "walkthrough",
        "workflow",
    }
    missing = sorted(name for name in documented_methods if not hasattr(Session, name))
    assert not missing


def test_readme_states_alpha_and_legacy_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "BuildML 2.2 alpha" in readme
    assert "public 2.x entry point is `buildml.Session`" in readme
    assert "BuildML 1.x legacy boundary" in readme
    assert "There is no compatibility shim" in readme


def test_sphinx_current_path_uses_session() -> None:
    current_pages = ("readme.rst", "usage.rst", "features.rst", "package.rst")
    text = "\n".join(
        (ROOT / "docs" / page).read_text(encoding="utf-8") for page in current_pages
    )
    assert "buildml.Session" in text
    assert "buildml.automate" not in text


def test_copy_lint_rejects_approved_banned_filler_and_stale_apis() -> None:
    banned = (
        "Executive narrative",
        "Actionable recommendations",
        "This is research-grade output",
        "Unlock seamless workflows",
    )
    for text in banned:
        assert any(pattern.search(text) for _, pattern in COPY_RULES), text
    assert STALE_API.search("Call buildml.preprocessing before fitting")


def test_generated_prose_matches_human_tone_fixture_without_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 22, None, 35, 35, 42] * 8,
            "constant": ["same"] * 48,
            "customer_id": [f"customer-{index}" for index in range(48)],
            "target": [0, 1, 0, 1, 0, 1] * 8,
        }
    )
    report = Session.ingest(frame).set_roles(
        {
            "age": "feature",
            "constant": "feature",
            "customer_id": "id",
            "target": "target",
        }
    ).eda(max_plots=0)
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "human_tone.json").read_text(encoding="utf-8")
    )
    details = [finding.detail for finding in report.findings]
    titles = [recommendation.title for recommendation in report.recommendation_details]
    assert all(expected in details for expected in fixture["findings"])
    assert all(expected in titles for expected in fixture["recommendation_titles"])

    def normalize(value: str) -> str:
        return re.sub(r"\W+", " ", value.casefold()).strip()

    narrative = {normalize(value) for value in report.narrative}
    recommendations = {normalize(value) for value in report.recommendations}
    assert narrative.isdisjoint(recommendations)
    assert len(narrative) == len(report.narrative)
    assert len(recommendations) == len(report.recommendations)
    assert all(recommendation.based_on for recommendation in report.recommendation_details)
