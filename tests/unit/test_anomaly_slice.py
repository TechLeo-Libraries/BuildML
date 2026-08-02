"""Unit coverage for the anomaly / fraud thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.anomaly.checkpoint import (
    BUNDLE_FORMAT,
    load_anomaly_bundle,
    save_anomaly_bundle,
)
from buildml.anomaly.evaluate import evaluate_anomaly
from buildml.anomaly.fit import fit_detector
from buildml.anomaly.score import score_anomalies
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG


def _anomaly_frame(n_normal: int = 160, n_fraud: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal([0.0, 0.0], 0.8, size=(n_normal, 2))
    fraud = rng.normal([4.5, -4.0], 0.5, size=(n_fraud, 2))
    frame = pd.DataFrame(np.vstack([normal, fraud]), columns=["x", "y"])
    frame["is_fraud"] = [0] * n_normal + [1] * n_fraud
    return frame


def _ready_session(*, validation: bool = False) -> Session:
    # Keep is_fraud as target so scale/encode do not transform labels; anomaly
    # feature resolution already excludes protected target roles.
    roles = {"x": "feature", "y": "feature", "is_fraud": "target"}
    kwargs: dict = {"test_size": 0.25, "random_state": 0, "stratify": True}
    if validation:
        kwargs["validation_size"] = 0.2
    return Session.ingest(_anomaly_frame()).set_roles(roles).split(**kwargs).scale(
        method="standard"
    )


def test_core_import_does_not_require_extra() -> None:
    import buildml
    import buildml.anomaly as anomaly

    assert hasattr(buildml, "Session")
    assert hasattr(anomaly, "fit_detector")
    assert hasattr(Session, "fit_anomaly")


def test_catalog_covers_anomaly_operations() -> None:
    for name in (
        "fit_anomaly",
        "score_anomalies",
        "evaluate_anomaly",
        "save_anomaly_bundle",
        "load_anomaly_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert (
        "anomaly-train-fit-holdout-score"
        in OPERATION_CATALOG["fit_anomaly"].concept_links
    )
    assert "anomaly-threshold-alert-rate" in OPERATION_CATALOG["fit_anomaly"].concept_links
    assert "anomaly-imbalance-metrics" in OPERATION_CATALOG["evaluate_anomaly"].concept_links
    assert (
        "anomaly-bundle-boundary"
        in OPERATION_CATALOG["save_anomaly_bundle"].concept_links
    )
    assert "anomaly-eda-boundary" in OPERATION_CATALOG["fit_anomaly"].concept_links


def test_fit_requires_split() -> None:
    session = Session.ingest(_anomaly_frame()).set_roles(
        {"x": "feature", "y": "feature", "is_fraud": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_anomaly(method="isolation_forest")


def test_isolation_forest_fit_score_evaluate_and_bundle(tmp_path: Path) -> None:
    session = _ready_session()
    fit = session.fit_anomaly(
        method="isolation_forest",
        mode="unsupervised",
        contamination=0.1,
        random_state=0,
    )
    assert fit.method == "isolation_forest"
    assert fit.mode == "unsupervised"
    assert session.anomaly_plan is not None
    assert fit.threshold_policy == "contamination"
    assert 0.0 < fit.train_alert_rate < 0.5
    assert "is_fraud" not in fit.columns

    scored = session.score_anomalies(partition="test")
    assert scored.n_rows > 0
    assert scored.n_flagged >= 0
    assert scored.threshold == fit.threshold
    assert len(scored.scores) == scored.n_rows
    assert set(scored.flags).issubset({0, 1})

    metrics = session.evaluate_anomaly(partition="test", positive_label=1)
    assert "alert_rate" in metrics.metrics
    assert "average_precision" in metrics.labeled_metrics
    assert "precision_at_k" in metrics.labeled_metrics
    assert metrics.positive_rate is not None

    path = session.save_anomaly_bundle(tmp_path / "anomaly")
    assert (path / "meta.json").is_file()
    assert (path / "anomaly_plan.joblib").is_file()
    plan = load_anomaly_bundle(path)
    assert plan.method == "isolation_forest"
    assert plan.columns == session.anomaly_plan.columns

    restored = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "y": "feature", "is_fraud": "target"}
    )
    restored.split(test_size=0.25, stratify=True, random_state=0).scale(method="standard")
    restored.load_anomaly_bundle(path)
    again = restored.score_anomalies(partition="test")
    assert again.flags == scored.flags

    with pytest.raises(ValidationError, match=BUNDLE_FORMAT):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "meta.json").write_text('{"format": "buildml.rag_bundle.v1"}', encoding="utf-8")
        (bad / "anomaly_plan.joblib").write_bytes(b"not-a-real-joblib")
        load_anomaly_bundle(bad)


def test_novelty_and_supervised_modes() -> None:
    session = _ready_session()
    nov = session.fit_anomaly(
        method="lof",
        mode="novelty",
        normal_label_value=0,
        contamination=0.1,
        n_neighbors=10,
    )
    assert nov.mode == "novelty"
    assert nov.n_fit_rows < nov.n_train_rows
    assert any("normal-only" in d.lower() or "novelty" in d.lower() for d in nov.disclosures)
    scored = session.score_anomalies(partition="test")
    assert scored.n_rows > 0

    supervised = _ready_session()
    fit = supervised.fit_anomaly(method="supervised_hgb", mode="supervised")
    assert fit.mode == "supervised"
    assert fit.method == "supervised_hgb"
    ev = supervised.evaluate_anomaly(partition="test", k=5)
    assert "average_precision" in ev.labeled_metrics


def test_one_class_svm_decision_zero() -> None:
    session = _ready_session()
    fit = session.fit_anomaly(
        method="one_class_svm",
        mode="unsupervised",
        threshold_policy="decision_zero",
        nu=0.1,
    )
    assert fit.threshold == 0.0
    assert fit.threshold_policy == "decision_zero"
    scored = session.score_anomalies(partition="test")
    assert scored.n_rows > 0


def test_attach_requires_all_partition() -> None:
    session = _ready_session()
    session.fit_anomaly(method="isolation_forest", contamination=0.1)
    with pytest.raises(ValidationError, match="partition='all'"):
        session.score_anomalies(partition="test", attach=True)
    attached = session.score_anomalies(partition="all", attach=True)
    assert attached.attached is True
    assert "is_anomaly" in session.dataset.columns
    assert "anomaly_score" in session.dataset.columns


def test_prefer_reduce_components() -> None:
    session = (
        Session.ingest(_anomaly_frame())
        .set_roles({"x": "feature", "y": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=2, prefix="pc")
    )
    fit = session.fit_anomaly(
        method="isolation_forest",
        contamination=0.1,
        prefer_reduce_components=True,
    )
    assert fit.used_reduce_components is True
    assert all(c.startswith("pc_") for c in fit.columns)


def test_low_level_fit_detector_matches_session(tmp_path: Path) -> None:
    session = _ready_session()
    plan, fit = fit_detector(
        session.dataset,
        session.split_plan,
        method="isolation_forest",
        contamination=0.1,
        random_state=0,
        prefer_reduce_components=False,
    )
    _, scored = score_anomalies(
        session.dataset, plan, session.split_plan, partition="test"
    )
    eval_result = evaluate_anomaly(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
    )
    assert fit.method == "isolation_forest"
    assert scored.n_rows == eval_result.n_rows
    path = save_anomaly_bundle(tmp_path / "direct", plan, fit_result=fit)
    restored = load_anomaly_bundle(path)
    assert restored.method == plan.method
    assert restored.columns == plan.columns
