"""Session-facing slice tests for case-based reasoning."""

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
    import buildml.cbr as cbr

    assert hasattr(cbr, "fit_cbr")
    assert hasattr(cbr, "retrieve_cases")
    assert hasattr(Session, "fit_cbr")
    assert hasattr(Session, "retain_cbr")
    for op in (
        "fit_cbr",
        "retrieve_cases",
        "predict_cbr",
        "evaluate_cbr",
        "retain_cbr",
        "save_cbr_bundle",
        "load_cbr_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "cbr-case-memory" in OPERATION_CATALOG["fit_cbr"].concept_links
    assert "cbr-vs-rag" in OPERATION_CATALOG["fit_cbr"].concept_links

    registry = build_default_registry()
    assert registry.get("fit_cbr") is not None
    assert registry.get("retrieve_cases") is not None
    assert registry.get("evaluate_cbr") is not None


def test_session_fit_predict_eval_bundle(tmp_path: Path) -> None:
    session = _clf_session()
    fit = session.fit_cbr(
        task="classification",
        metric="euclidean",
        reuse="distance_weighted",
        k=5,
    )
    assert session.cbr_plan is not None
    assert fit.n_cases >= 1
    assert fit.metric == "euclidean"

    retrieved = session.retrieve_cases(partition="test", k=3)
    assert retrieved.n_queries == len(retrieved.traces)
    assert len(retrieved.traces[0].neighbor_case_ids) == 3

    pred = session.predict_cbr(partition="test", return_traces=True)
    assert len(pred.predictions) == pred.n_rows
    assert len(pred.traces) == pred.n_rows
    assert pred.traces[0].prediction is not None
    assert len(pred.traces[0].weights) == 5

    ev = session.evaluate_cbr(partition="validation")
    assert "accuracy" in ev.metrics
    assert ev.mean_neighbor_distance is not None

    out = tmp_path / "cbr_bundle"
    session.save_cbr_bundle(out)
    assert (out / "meta.json").is_file()
    assert (out / "cbr_plan.joblib").is_file()

    other = _clf_session()
    other.load_cbr_bundle(out)
    assert other.cbr_plan is not None
    assert other.cbr_plan.metric == "euclidean"
    reloaded = other.evaluate_cbr(partition="test")
    assert "accuracy" in reloaded.metrics


def test_regression_and_mixed_metric() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=160),
            "b": rng.normal(size=160),
            "cat": rng.choice(["x", "y", "z"], size=160),
            "y": rng.normal(size=160),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {"a": "feature", "b": "feature", "cat": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard", columns=["a", "b"])
    )
    fit = session.fit_cbr(
        task="regression",
        metric="mixed",
        reuse="distance_weighted",
        categorical_columns=["cat"],
        k=4,
    )
    assert fit.n_cases >= 1
    ev = session.evaluate_cbr(partition="test")
    assert "rmse" in ev.metrics

    session2 = (
        Session.ingest(frame)
        .set_roles(
            {"a": "feature", "b": "feature", "cat": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=1)
        .scale(method="standard", columns=["a", "b"])
    )
    fit2 = session2.fit_cbr(
        task="regression",
        metric="euclidean",
        reuse="local_ridge",
        k=5,
        columns=["a", "b"],
    )
    assert fit2.reuse == "local_ridge"
    assert "r2" in session2.evaluate_cbr(partition="validation").metrics


def test_retain_refuses_holdout_and_accepts_external() -> None:
    session = _clf_session()
    session.fit_cbr(task="classification", k=3)
    split = session._split_plan
    assert split is not None
    frame = session.dataset._ensure_pandas()
    test_label = frame.index[list(split.test_indices)[0]]

    with pytest.raises(ValidationError, match="validation/test"):
        session.retain_cbr(
            row_indices=[test_label],
            source_disclosure="should fail — test index",
        )

    # External labeled rows with fresh index labels (not in holdout), in the
    # same post-scale feature space as the live Session frame.
    train_frame = frame.iloc[list(split.train_indices)[:2]].copy()
    train_frame.index = [20_000, 20_001]
    before = session.cbr_plan.case_base.n_cases
    result = session.retain_cbr(
        labeled_frame=train_frame,
        source_disclosure="synthetic external labels for unit test",
    )
    assert result.n_added == 2
    assert session.cbr_plan.case_base.n_cases == before + 2
    assert session.cbr_plan.case_base.n_retained == 2


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
        session.fit_cbr(task="classification")


def test_walkthrough_exposes_cbr_status() -> None:
    session = _clf_session()
    session.fit_cbr(task="classification", k=3)
    report = session.walkthrough()
    payload = report.to_dict()
    assert "cbr_status" in payload
    assert payload["cbr_status"]["enabled"] is True
    assert payload["cbr_status"]["has_cbr_plan"] is True
