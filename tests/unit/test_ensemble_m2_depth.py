"""Depth coverage for native ensembles (leakage, explain, AI allowlist)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.ai.tools import registered_tool_names
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.sync import REQUIRED_AI_TOOL_SESSION_METHODS


def _frame(n: int = 100, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.3 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _bases():
    return {
        "lr": LogisticRegression(max_iter=400),
        "rf": RandomForestClassifier(n_estimators=15, random_state=0),
    }


def test_soft_voting_requires_predict_proba() -> None:
    from sklearn.svm import LinearSVC

    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0, stratify=True)
    )
    with pytest.raises(ValidationError, match="predict_proba"):
        session.fit_voting(
            {"svc": LinearSVC(), "rf": RandomForestClassifier(n_estimators=10, random_state=0)},
            voting="soft",
        )


def test_blending_discloses_train_inner_holdout() -> None:
    session = (
        Session.ingest(_frame(160))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    fit = session.fit_blending(_bases(), holdout_fraction=0.2, random_state=0)
    joined = " ".join(fit.disclosures).lower()
    assert "holdout_fraction" in joined
    assert "validation/test" in joined or "session test" in joined
    assert session.ensemble_plan is not None
    assert session.ensemble_plan.refit_bases_on_full_train is True


def test_stacking_sets_classical_fit_result_for_predict() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0, stratify=True)
        .scale(method="standard")
    )
    session.fit_stacking(_bases(), cv=3)
    preds = session.predict(partition="test")
    assert len(preds) > 0


def test_explain_prerequisites_for_ensemble_ops() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0, stratify=True)
    )
    before = session.explain("fit_voting", moment="before")
    assert before.operation == "fit_voting"
    assert before.prerequisite_status.get("split") is True

    session.fit_voting(_bases(), voting="hard")
    after_save = session.explain("save_ensemble_bundle", moment="before")
    assert after_save.prerequisite_status.get("ensemble-plan") is True


def test_ai_allowlist_includes_ensemble_surface() -> None:
    names = set(registered_tool_names())
    for tool in (
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "evaluate_ensemble",
        "save_ensemble_bundle",
    ):
        assert tool in names
    assert "fit_voting" in REQUIRED_AI_TOOL_SESSION_METHODS
    assert "fit_stacking" in REQUIRED_AI_TOOL_SESSION_METHODS
    assert "fit_blending" in REQUIRED_AI_TOOL_SESSION_METHODS
    assert "evaluate_ensemble" in REQUIRED_AI_TOOL_SESSION_METHODS


def test_fit_without_split_raises_leakage() -> None:
    session = Session.ingest(_frame()).set_roles(
        {"x1": "feature", "x2": "feature", "y": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_stacking(_bases(), cv=3)


def test_walkthrough_includes_ensemble_status() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0, stratify=True)
        .scale(method="standard")
    )
    session.fit_voting(_bases())
    report = session.walkthrough()
    status = report.ensemble_status
    assert status.get("has_ensemble_plan") is True
    assert status.get("strategy") == "voting"
