"""Session-facing slice tests for symbolic / neuro-symbolic ML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG


def _clf_session() -> Session:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    return (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )


def test_core_import_and_catalog() -> None:
    import buildml.symbolic as sym

    assert hasattr(sym, "fit_symbolic")
    assert hasattr(sym, "fit_neuro_symbolic")
    assert hasattr(Session, "fit_symbolic")
    assert hasattr(Session, "fit_neuro_symbolic")
    for op in (
        "fit_symbolic",
        "evaluate_symbolic",
        "predict_symbolic",
        "fit_neuro_symbolic",
        "evaluate_neuro_symbolic",
        "predict_neuro_symbolic",
        "save_symbolic_bundle",
        "load_symbolic_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "symbolic-rules" in OPERATION_CATALOG["fit_symbolic"].concept_links
    assert "neuro-symbolic-hybrid" in OPERATION_CATALOG["fit_neuro_symbolic"].concept_links

    registry = build_default_registry()
    assert registry.get("fit_symbolic") is not None
    assert registry.get("fit_neuro_symbolic") is not None
    assert registry.get("evaluate_symbolic") is not None


def test_session_fit_predict_eval_bundle(tmp_path: Path) -> None:
    session = _clf_session()
    fit = session.fit_symbolic(source="decision_tree", task="classification")
    assert session.symbolic_plan is not None
    assert fit.n_rules >= 1
    assert fit.provenance == "induced_tree"

    pred = session.predict_symbolic(partition="test", return_traces=True)
    assert len(pred.predictions) == pred.n_rows
    assert len(pred.traces) == pred.n_rows
    assert pred.traces[0].chosen_rule_id is not None or pred.traces[0].notes

    ev = session.evaluate_symbolic(partition="validation")
    assert "accuracy" in ev.metrics
    assert ev.rule_coverage is not None

    out = tmp_path / "symbolic_bundle"
    session.save_symbolic_bundle(out)
    assert (out / "meta.json").is_file()
    assert (out / "symbolic_plan.joblib").is_file()

    other = _clf_session()
    other.load_symbolic_bundle(out)
    assert other.symbolic_plan is not None
    assert other.symbolic_plan.source == "decision_tree"
    reloaded = other.evaluate_symbolic(partition="test")
    assert "accuracy" in reloaded.metrics


def test_declared_rules_and_decision_list() -> None:
    session = _clf_session()
    rules = [
        {
            "if": [{"column": "a", "op": ">", "value": 0.0}],
            "then": 1,
            "priority": 10,
        },
        {
            "if": [{"column": "a", "op": "<=", "value": 0.0}],
            "then": 0,
            "priority": 5,
        },
    ]
    fit = session.fit_symbolic(
        source="declared", task="classification", rules=rules
    )
    assert fit.provenance == "declared"
    pred = session.predict_symbolic(partition="test")
    assert len(pred.predictions) > 0

    session2 = _clf_session()
    fit2 = session2.fit_symbolic(source="decision_list", task="classification")
    assert fit2.provenance == "induced_list"
    assert fit2.n_rules >= 1


def test_neuro_symbolic_overlay_and_bundle(tmp_path: Path) -> None:
    session = _clf_session()
    hard_rules = [
        {
            "rule_id": "force_pos",
            "if": [{"column": "a", "op": ">", "value": 1.5}],
            "then": 1,
            "hardness": "hard",
            "kind": "constraint",
            "priority": 100,
        }
    ]
    fit = session.fit_neuro_symbolic(
        mode="constraint_overlay",
        base_estimator="logistic_regression",
        task="classification",
        rules=hard_rules,
        rule_source="declared",
    )
    assert session.neuro_symbolic_plan is not None
    assert fit.mode == "constraint_overlay"
    assert fit.rule_provenance == "declared"

    pred = session.predict_neuro_symbolic(partition="test")
    assert pred.neural_predictions is not None
    assert len(pred.traces) == pred.n_rows

    ev = session.evaluate_neuro_symbolic(partition="validation")
    assert "accuracy" in ev.metrics

    # rules_as_features path
    session2 = _clf_session()
    fit2 = session2.fit_neuro_symbolic(
        mode="rules_as_features",
        rule_source="decision_tree",
        task="classification",
    )
    assert fit2.n_rules >= 1
    ev2 = session2.evaluate_neuro_symbolic(partition="test")
    assert "accuracy" in ev2.metrics

    out = tmp_path / "neuro_bundle"
    session2.save_symbolic_bundle(out)
    other = _clf_session()
    other.load_symbolic_bundle(out)
    assert other.neuro_symbolic_plan is not None
    assert other.neuro_symbolic_plan.mode == "rules_as_features"


def test_leakage_refuses_fit_without_split() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            "b": rng.normal(size=40),
            "y": rng.integers(0, 2, size=40),
        }
    )
    session = Session.ingest(frame).set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    with pytest.raises((ValidationError, LeakageError)):
        session.fit_symbolic(source="decision_tree")
    with pytest.raises((ValidationError, LeakageError)):
        session.fit_neuro_symbolic(mode="constraint_overlay")


def test_walkthrough_exposes_symbolic_status() -> None:
    session = _clf_session()
    session.fit_symbolic(source="decision_tree")
    report = session.walkthrough()
    payload = report.to_dict()
    assert "symbolic_status" in payload
    assert payload["symbolic_status"]["enabled"] is True
    assert payload["symbolic_status"]["has_symbolic_plan"] is True
