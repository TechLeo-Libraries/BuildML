"""Teaching Studio / walkthrough surface for nested CV warm_start_studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.dashboard.teaching import build_teaching_studios
from buildml.session.walkthrough import warm_start_studies_status


def _cls_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.2 * x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _has_optuna() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


def test_warm_start_status_empty_without_history() -> None:
    status = warm_start_studies_status([])
    assert status["enabled"] is False
    assert status["disclosures"] == []


def test_warm_start_status_from_history_record() -> None:
    history = [
        {
            "operation_id": "nested_cv_score",
            "sequence": 4,
            "parameters": {"warm_start_studies": True, "search_method": "optuna"},
            "result_summary": {
                "warm_start_studies": True,
                "search_method": "optuna",
                "n_outer_splits": 2,
            },
        }
    ]
    status = warm_start_studies_status(history)
    assert status["enabled"] is True
    assert status["search_method"] == "optuna"
    assert status["n_outer_splits"] == 2
    assert status["shared"] == "optuna_study_trial_history"
    assert any("trial history" in note.lower() for note in status["disclosures"])
    assert any("test" in note.lower() for note in status["disclosures"])


@pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")
def test_walkthrough_and_teaching_surface_warm_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.model import selection as selection_mod

    session = (
        Session.ingest(_cls_frame(120))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )

    real_optuna = selection_mod.optuna_search

    def _fast(*args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["n_trials"] = 2
        return real_optuna(*args, **kwargs)

    monkeypatch.setattr(selection_mod, "optuna_search", _fast)
    session.nested_cv_score(
        DecisionTreeClassifier(random_state=0),
        inner_search="optuna",
        param_space={"max_depth": {"type": "int", "low": 2, "high": 3}},
        n_trials=2,
        outer_cv=2,
        inner_cv=2,
        random_state=0,
        warm_start_studies=True,
    )

    walk = session.walkthrough()
    assert walk.warm_start_status["enabled"] is True
    assert walk.warm_start_status["shared"] == "optuna_study_trial_history"
    doc = walk.export_html(tmp_path / "walk_warm.html").read_text(encoding="utf-8")
    assert "warm_start_studies" in doc
    assert "optuna" in doc.lower()

    eda = session.eda(include_plots=False)
    assert eda.overview["warm_start_status"]["enabled"] is True
    studios = build_teaching_studios(eda.to_dict())
    cockpit = studios["cockpit"]
    assert cockpit["worked_example"]["values"]["warm_start_studies"] is True
    assert any("warm_start_studies" in line for line in cockpit["interpretation"])
    assert "cross-validation" in cockpit["concepts"]
