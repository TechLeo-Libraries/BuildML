"""Industry-depth anomaly tests (PyOD/torch skipped when extras missing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.anomaly.catalog import anomaly_capability_matrix, list_anomaly_methods
from buildml.anomaly.extras import pyod_available
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import torch_spec_available


def _frame(n_normal: int = 120, n_fraud: int = 15, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(n_normal, 2))
    fraud = rng.normal(5.0, 0.4, size=(n_fraud, 2))
    frame = pd.DataFrame(np.vstack([normal, fraud]), columns=["a", "b"])
    frame["is_fraud"] = [0] * n_normal + [1] * n_fraud
    return frame


def test_capability_matrix_core() -> None:
    matrix = anomaly_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "isolation_forest" in matrix["backends"]["sklearn"]["methods"]
    assert matrix["backends"]["pyod"]["available"] == pyod_available()


def test_list_anomaly_methods_supervised() -> None:
    methods = list_anomaly_methods(mode="supervised")
    assert "supervised_hgb" in methods


@pytest.mark.skipif(not pyod_available(), reason="pyod not installed")
def test_pyod_ecod_fit_evaluate() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.anomaly.fit(backend="pyod", method="ecod", contamination=0.1)
    assert fit.backend == "pyod"
    ev = session.anomaly.evaluate(partition="test")
    assert ev.labeled_metrics["average_precision"] >= 0.0


@pytest.mark.skipif(not pyod_available(), reason="pyod not installed")
@pytest.mark.parametrize("method", ["hbos", "copod"])
def test_pyod_catalog_methods(method: str) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.anomaly.fit(backend="pyod", method=method, contamination=0.1)
    assert fit.method == method


def test_pyod_missing_extra_raises() -> None:
    if pyod_available():
        pytest.skip("pyod installed")
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(MissingExtraError):
        session.anomaly.fit(backend="pyod", method="ecod")


@pytest.mark.skipif(not torch_spec_available(), reason="torch not installed")
def test_torch_autoencoder_fit() -> None:
    session = (
        Session.ingest(_frame(n_normal=40, n_fraud=8))
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.anomaly.fit(
        backend="torch", method="autoencoder", ae_epochs=2,
        ae_batch_size=16, latent_dim=1, random_state=0,
    )
    assert fit.backend == "torch"
    evaluation = session.anomaly.evaluate(partition="test")
    assert np.isfinite(evaluation.labeled_metrics["average_precision"])


def test_tune_anomaly_threshold_validation_only() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.anomaly.fit(method="isolation_forest", contamination=0.1)
    tuned = session.anomaly.tune_threshold(partition="validation", metric="f1")
    assert tuned.threshold != tuned.old_threshold or tuned.metric == "f1"
    assert session.anomaly_plan.threshold_policy == "validation_tuned"
    with pytest.raises(ValidationError, match="Refusing to tune"):
        session.anomaly.tune_threshold(partition="test")


def test_supervised_xgb_skip_without_extra() -> None:
    from buildml.anomaly.extras import xgboost_available

    if xgboost_available():
        pytest.skip("xgboost installed")
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(MissingExtraError):
        session.anomaly.fit(method="supervised_xgb", mode="supervised")


def test_anomaly_bundle_roundtrip_backend(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "is_fraud": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.anomaly.fit(method="isolation_forest", contamination=0.1)
    path = session.anomaly.save_bundle(tmp_path / "bundle")
    fresh = Session.ingest(session.to_pandas()).set_roles(
        {"a": "feature", "b": "feature", "is_fraud": "target"}
    )
    fresh.split(test_size=0.25, stratify=True, random_state=0).scale(method="standard")
    fresh.anomaly.load_bundle(path, trusted=True)
    assert fresh.anomaly_plan.backend == "sklearn"
