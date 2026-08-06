"""Industry-depth tests for online / continual backends (R6.3)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.online.catalog import (
    list_online_estimators,
    online_capability_matrix,
    resolve_backend_estimator,
)
from buildml.online.extras import river_available
from buildml.online.fit import fit_online


def _torch_spec_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def _frame(n: int = 180, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.8, size=(n // 2, 2))
    x1 = rng.normal(2.5, 0.8, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def test_capability_matrix_sklearn_always_available() -> None:
    matrix = online_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "sgd_classifier" in matrix["backends"]["sklearn"]["estimators"]
    assert "chunk_ingestion" in matrix


def test_list_online_estimators_includes_sklearn() -> None:
    estimators = list_online_estimators()
    assert "sgd_classifier" in estimators


def test_resolve_backend_estimator_defaults() -> None:
    backend, estimator = resolve_backend_estimator(
        backend=None, estimator="passive_aggressive_classifier"
    )
    assert backend == "sklearn"
    assert estimator == "passive_aggressive_classifier"


def test_resolve_industry_requires_river_when_missing() -> None:
    if river_available():
        backend, estimator = resolve_backend_estimator(
            backend="industry", estimator="river_logistic"
        )
        assert backend == "industry"
        assert estimator == "river_logistic"
    else:
        with pytest.raises(MissingExtraError):
            resolve_backend_estimator(backend="industry", estimator="river_logistic")


@pytest.mark.skipif(not river_available(), reason="river not installed")
def test_river_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_online(
        backend="industry",
        estimator="river_logistic",
        chunk_size=35,
        n_init=35,
        classes=[0, 1],
        drift_detector="adwin",
        prefer_reduce_components=False,
    )
    assert fit.backend == "industry"
    session.partial_fit_online(n_rows=35)
    ev = session.evaluate_online(partition="test")
    assert ev.metrics["accuracy"] >= 0.0
    assert session.online_plan is not None
    assert session.online_plan.backend == "industry"


@pytest.mark.skipif(not _torch_spec_present(), reason="torch not installed")
def test_replay_mlp_torch_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    try:
        fit = session.fit_online(
            backend="torch",
            estimator="replay_mlp",
            chunk_size=30,
            n_init=30,
            classes=[0, 1],
            epochs_per_update=3,
            prefer_reduce_components=False,
        )
    except (MissingExtraError, ValidationError, OSError) as exc:
        if "torch" in str(exc).lower():
            pytest.skip("torch not runnable on this host")
        raise
    assert fit.backend == "torch"
    session.partial_fit_online(n_rows=30)
    ev = session.evaluate_online(partition="test")
    assert "accuracy" in ev.metrics


def test_adwin_on_sklearn_backend_refused() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="requires backend='industry'"):
        session.fit_online(
            backend="sklearn",
            drift_detector="adwin",
            classes=[0, 1],
            chunk_size=30,
            n_init=30,
        )


def test_low_level_fit_with_backend(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    plan, fit = fit_online(
        session.dataset,
        session.split_plan,
        backend="sklearn",
        estimator="sgd_classifier",
        chunk_size=35,
        n_init=35,
        classes=[0, 1],
        prefer_reduce_components=False,
    )
    assert plan.backend == "sklearn"
    assert fit.backend == "sklearn"
