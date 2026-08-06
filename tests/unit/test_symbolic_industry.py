"""Industry backend tests for symbolic / neuro-symbolic ML."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_spec_available
from buildml.symbolic.catalog import symbolic_capability_matrix
from buildml.symbolic.extras import (
    imodels_available,
    skope_rules_available,
    symbolic_industry_available,
)


def _clf_session() -> Session:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(220, 3))
    y = ((x[:, 0] + 0.4 * x[:, 1]) > 0).astype(int)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": y})
    return (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=1, stratify=True)
        .scale(method="standard")
    )


def test_capability_matrix_sklearn_always_available() -> None:
    matrix = symbolic_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "decision_tree" in matrix["backends"]["sklearn"]["sources"]
    assert matrix["neuro_symbolic_backends"]["sklearn"]["available"] is True


def test_session_capability_matrix() -> None:
    matrix = Session.symbolic_capability_matrix()
    assert "backends" in matrix
    assert "neuro_symbolic_backends" in matrix


def test_sklearn_backend_explicit() -> None:
    session = _clf_session()
    fit = session.fit_symbolic(backend="sklearn", source="decision_tree")
    assert fit.backend == "sklearn"
    assert fit.n_rules >= 1
    ev = session.evaluate_symbolic(partition="test")
    assert ev.metrics.get("accuracy") is not None
    pred = session.predict_symbolic(partition="test", return_traces=True)
    assert len(pred.traces) == pred.n_rows


@pytest.mark.skipif(not skope_rules_available(), reason="skope-rules not installed")
def test_skope_rules_industry_backend() -> None:
    session = _clf_session()
    fit = session.fit_symbolic(backend="industry", method="skope_rules")
    assert fit.backend == "industry"
    assert fit.method == "skope_rules"
    assert fit.provenance == "induced_skope"
    assert fit.n_rules >= 1


@pytest.mark.skipif(not imodels_available(), reason="imodels not installed")
def test_rulefit_industry_backend() -> None:
    session = _clf_session()
    fit = session.fit_symbolic(backend="industry", method="rulefit", max_depth=3)
    assert fit.backend == "industry"
    assert fit.method == "rulefit"
    assert fit.n_rules >= 1


def test_industry_backend_raises_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    if symbolic_industry_available():
        pytest.skip("symbolic-industry packages installed in this environment")
    monkeypatch.setattr(
        "buildml.symbolic.catalog.symbolic_industry_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "buildml.symbolic.catalog.backend_available",
        lambda name: name == "sklearn",
    )
    session = _clf_session()
    with pytest.raises(MissingExtraError, match="symbolic-industry"):
        session.fit_symbolic(backend="industry", method="skope_rules")


def test_torch_concept_bottleneck_neuro_symbolic() -> None:
    if not torch_spec_available():
        pytest.skip("torch not installed")
    session = _clf_session()
    try:
        fit = session.fit_neuro_symbolic(
            backend="torch",
            base_estimator="concept_bottleneck_lite",
            mode="constraint_overlay",
            torch_epochs=8,
        )
    except MissingExtraError:
        pytest.skip("torch installed but not importable in this environment")
    assert fit.backend == "torch"
    assert fit.torch_method == "concept_bottleneck_lite"
    ev = session.evaluate_neuro_symbolic(partition="test")
    assert ev.metrics.get("accuracy") is not None
    pred = session.predict_neuro_symbolic(partition="test", return_traces=True)
    assert pred.neural_predictions is not None


def test_verify_constraints_skipped_without_z3() -> None:
    session = _clf_session()
    fit = session.fit_symbolic(
        backend="sklearn",
        source="declared",
        rules=[
            {
                "rule_id": "r1",
                "if": [{"column": "a", "op": ">", "value": 0}],
                "then": 1,
                "hardness": "hard",
                "kind": "constraint",
            }
        ],
        verify_constraints=True,
    )
    assert fit.n_rules == 1
    assert any("Z3" in d or "constraint" in d.lower() for d in fit.disclosures)
