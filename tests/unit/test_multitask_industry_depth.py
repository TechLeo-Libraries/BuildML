"""Industry-depth tests for multi-task backends (R6.4)."""

from __future__ import annotations

from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.multitask.catalog import (
    list_multitask_methods,
    multitask_capability_matrix,
    resolve_backend_method,
)
from buildml.multitask.extras import xgboost_available


def _torch_spec_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def _cls_frame(n: int = 180, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    frame = pd.DataFrame(x, columns=["x", "y"])
    frame["t1"] = (x[:, 0] > 0).astype(int)
    frame["t2"] = (x[:, 1] > 0).astype(int)
    return frame


def _reg_frame(n: int = 180, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    frame = pd.DataFrame(x, columns=["x", "y"])
    frame["t1"] = x[:, 0] * 1.5 + rng.normal(0, 0.1, size=n)
    frame["t2"] = x[:, 1] * -0.8 + rng.normal(0, 0.1, size=n)
    return frame


def _mixed_frame(n: int = 180, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    frame = pd.DataFrame(x, columns=["x", "y"])
    frame["t_cls"] = (x[:, 0] > 0).astype(int)
    frame["t_reg"] = x[:, 1] * 2.0 + rng.normal(0, 0.05, size=n)
    return frame


def test_capability_matrix_sklearn_always_available() -> None:
    matrix = multitask_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "multi_output" in matrix["backends"]["sklearn"]["methods"]
    assert "evaluation" in matrix


def test_list_multitask_methods_includes_sklearn() -> None:
    methods = list_multitask_methods()
    assert "multi_output" in methods
    assert "classifier_chain" in methods


def test_resolve_backend_method_defaults() -> None:
    backend, method = resolve_backend_method(backend=None, method="multi_output")
    assert backend == "sklearn"
    assert method == "multi_output"


def test_resolve_industry_requires_extra_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "buildml.multitask.catalog.backend_available",
        lambda name: name == "sklearn",
    )
    with pytest.raises(MissingExtraError):
        resolve_backend_method(backend="industry", method="multi_output_xgb")


@pytest.mark.skipif(not xgboost_available(), reason="xgboost not installed")
def test_industry_xgb_classification_session_path() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles(
            {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_multitask(
        backend="industry",
        method="multi_output_xgb",
        task="classification",
        prefer_reduce_components=False,
    )
    assert fit.backend == "industry"
    ev = session.evaluate_multitask(partition="test")
    assert "mean_accuracy" in ev.metrics
    assert set(ev.per_task_metrics) == {"t1", "t2"}


@pytest.mark.skipif(not _torch_spec_present(), reason="torch not installed")
def test_torch_shared_trunk_mixed_targets() -> None:
    session = (
        Session.ingest(_mixed_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "t_cls": "target",
                "t_reg": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    try:
        fit = session.fit_multitask(
            backend="torch",
            method="shared_trunk_multihead",
            task="auto",
            prefer_reduce_components=False,
            epochs=20,
            batch_size=32,
        )
    except (MissingExtraError, ValidationError) as exc:
        if "torch" in str(exc).lower():
            pytest.skip("torch installed but not importable on this host")
        raise
    assert fit.backend == "torch"
    assert fit.task == "mixed"
    ev = session.evaluate_multitask(partition="test")
    assert "mean_accuracy" in ev.metrics
    assert "mean_mae" in ev.metrics


def test_sklearn_refuses_mixed_targets() -> None:
    session = (
        Session.ingest(_mixed_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "t_cls": "target",
                "t_reg": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="Mixed classification/regression"):
        session.fit_multitask(backend="sklearn", task="auto")


def test_multitask_status_includes_capability_matrix() -> None:
    from buildml.multitask.explain_hooks import multitask_status

    status = multitask_status()
    assert "capability_matrix" in status
    assert status["capability_matrix"]["backends"]["sklearn"]["available"]


def test_bundle_roundtrip_with_backend(tmp_path: Path) -> None:
    session = (
        Session.ingest(_reg_frame())
        .set_roles(
            {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    session.fit_multitask(
        method="multi_output",
        task="regression",
        base_estimator="ridge",
        prefer_reduce_components=False,
    )
    out = session.save_multitask_bundle(tmp_path / "bundle")
    session2 = Session.ingest(_reg_frame()).set_roles(
        {"x": "feature", "y": "feature", "t1": "target", "t2": "target"}
    )
    session2.load_multitask_bundle(out, trusted=True)
    assert session2.multitask_plan is not None
    assert session2.multitask_plan.method == "multi_output"


def test_invalid_backend_method_pairing() -> None:
    with pytest.raises(ValidationError):
        resolve_backend_method(backend="sklearn", method="shared_trunk_multihead")
