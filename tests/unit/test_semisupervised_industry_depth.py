"""Industry-depth tests for semi-supervised backends (R6.1)."""

from __future__ import annotations

from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from buildml.semisupervised.catalog import (
    list_semisupervised_methods,
    resolve_backend_method,
    semisupervised_capability_matrix,
)
from buildml.semisupervised.extras import xgboost_available


def _torch_spec_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def _frame(n: int = 160, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -0.5], 0.7, size=(n // 2, 2))
    x1 = rng.normal([1.2, 0.9], 0.7, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask(session: Session, fraction: float = 0.6) -> Session:
    rng = np.random.default_rng(4)
    full = session.to_pandas().copy()
    idx = list(session.split_plan.train_indices)
    blank = rng.choice(idx, size=max(1, int(fraction * len(idx))), replace=False)
    full.loc[blank, "y"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session


def test_capability_matrix_sklearn_always_available() -> None:
    matrix = semisupervised_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "label_propagation" in matrix["backends"]["sklearn"]["methods"]
    assert "ssl_integration" in matrix


def test_list_semisupervised_methods_includes_sklearn() -> None:
    methods = list_semisupervised_methods()
    assert "label_propagation" in methods
    assert "self_training" in methods


def test_resolve_backend_method_defaults() -> None:
    backend, method = resolve_backend_method(backend=None, method="label_spreading")
    assert backend == "sklearn"
    assert method == "label_spreading"


def test_resolve_industry_requires_extra_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "buildml.semisupervised.catalog.backend_available",
        lambda name: name == "sklearn",
    )
    with pytest.raises(MissingExtraError):
        resolve_backend_method(backend="industry", method="pseudo_label_xgb")


@pytest.mark.skipif(not xgboost_available(), reason="xgboost not installed")
def test_pseudo_label_xgb_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session = _mask(session)
    fit = session.fit_semisupervised(
        backend="industry",
        method="pseudo_label_xgb",
        prefer_reduce_components=False,
        threshold=0.65,
    )
    assert fit.backend == "industry"
    assert fit.n_unlabeled_train > 0
    ev = session.evaluate_semisupervised(partition="test")
    assert ev.metrics["accuracy"] >= 0.5


@pytest.mark.skipif(not _torch_spec_present(), reason="torch not installed")
def test_fixmatch_tabular_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session = _mask(session)
    try:
        fit = session.fit_semisupervised(
            backend="torch",
            method="fixmatch_tabular",
            prefer_reduce_components=False,
            epochs=12,
            batch_size=32,
        )
    except (MissingExtraError, ValidationError) as exc:
        if "torch" in str(exc).lower():
            pytest.skip("torch installed but not importable on this host")
        raise
    assert fit.backend == "torch"
    ev = session.evaluate_semisupervised(partition="test")
    assert "accuracy" in ev.metrics


def test_semisupervised_status_includes_capability_matrix() -> None:
    from buildml.semisupervised.explain_hooks import semisupervised_status

    status = semisupervised_status()
    assert "capability_matrix" in status
    assert status["capability_matrix"]["backends"]["sklearn"]["available"]


def test_bundle_roundtrip_with_backend(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session = _mask(session)
    session.fit_semisupervised(method="label_spreading", prefer_reduce_components=False)
    out = session.save_semisupervised_bundle(tmp_path / "bundle")
    session2 = Session.ingest(_frame()).set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    session2.load_semisupervised_bundle(out, trusted=True)
    assert session2.semisupervised_plan is not None
    assert session2.semisupervised_plan.method == "label_spreading"


def test_invalid_backend_method_pairing() -> None:
    with pytest.raises(ValidationError):
        resolve_backend_method(backend="sklearn", method="fixmatch_tabular")
