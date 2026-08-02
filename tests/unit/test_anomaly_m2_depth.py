"""Deeper anomaly coverage: modes, walkthrough, AI allowlist, low-level API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.anomaly.evaluate import evaluate_anomaly
from buildml.anomaly.fit import fit_detector
from buildml.anomaly.score import score_anomalies
from buildml.core.errors import ValidationError


def _frame(n_normal: int = 120, n_fraud: int = 15, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(n_normal, 2))
    fraud = rng.normal(5.0, 0.4, size=(n_fraud, 2))
    frame = pd.DataFrame(np.vstack([normal, fraud]), columns=["a", "b"])
    frame["is_fraud"] = [0] * n_normal + [1] * n_fraud
    return frame


def test_low_level_fit_score_evaluate(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_detector(
        session.dataset,
        session.split_plan,
        method="isolation_forest",
        contamination=0.12,
        random_state=0,
        prefer_reduce_components=False,
    )
    assert fit.n_fit_rows == fit.n_train_rows
    _, scored = score_anomalies(
        session.dataset, plan, session.split_plan, partition="test"
    )
    ev = evaluate_anomaly(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
    )
    assert scored.n_rows == ev.n_rows
    assert ev.labeled_metrics["average_precision"] >= 0.0

    from buildml.anomaly.checkpoint import save_anomaly_bundle

    out = save_anomaly_bundle(tmp_path / "direct", plan, fit_result=fit, eval_result=ev)
    assert (out / "meta.json").is_file()


def test_novelty_requires_normal_definition() -> None:
    # No target and no normal_label_column → novelty cannot define normals.
    frame = _frame().drop(columns=["is_fraud"])
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="novelty mode requires"):
        session.fit_anomaly(method="lof", mode="novelty", contamination=0.1)


def test_supervised_requires_binary_target() -> None:
    frame = _frame().drop(columns=["is_fraud"])
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="binary target"):
        session.fit_anomaly(method="supervised_hgb", mode="supervised")


def test_null_features_refused() -> None:
    frame = _frame()
    frame.loc[0, "a"] = np.nan
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="non-null"):
        session.fit_anomaly(method="isolation_forest", contamination=0.1)


def test_validation_fallback_and_explain() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.fit_anomaly(method="isolation_forest", contamination=0.1)
    # No validation partition → evaluate_anomaly falls back to test
    ev = session.evaluate_anomaly(partition="validation")
    assert ev.partition == "test"
    before = session.explain("fit_anomaly", moment="before")
    assert before.operation == "fit_anomaly"
    assert before.prerequisite_status.get("split") is True


def test_walkthrough_and_ai_tools_include_anomaly() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.fit_anomaly(method="isolation_forest", contamination=0.1)
    session.evaluate_anomaly(partition="test")
    report = session.walkthrough()
    payload = report.to_dict()
    status = payload["anomaly_status"]
    assert status["enabled"] is True
    assert status["method"] == "isolation_forest"
    assert any("AnomalyPlan" in d or "anomaly" in d.lower() for d in status["disclosures"])

    registry = build_default_registry()
    for name in (
        "fit_anomaly",
        "score_anomalies",
        "evaluate_anomaly",
        "save_anomaly_bundle",
        "load_anomaly_bundle",
    ):
        assert registry.get(name) is not None


def test_score_threshold_policy() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    # First fit to learn a score scale, then refit with absolute threshold.
    probe = session.fit_anomaly(method="isolation_forest", contamination=0.1)
    thr = float(probe.train_score_stats["p90"])
    fit = session.fit_anomaly(
        method="isolation_forest",
        contamination=0.1,
        threshold_policy="score_threshold",
        score_threshold=thr,
    )
    assert fit.threshold == thr
    assert fit.threshold_policy == "score_threshold"
