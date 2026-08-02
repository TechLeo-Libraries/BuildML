"""Industry-depth tests for active-learning backends (R6.2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from buildml.activelearning.catalog import (
    activelearning_capability_matrix,
    list_activelearning_strategies,
    resolve_backend_strategy,
)
from buildml.activelearning.extras import activelearning_industry_available


def _torch_spec_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def _frame(n: int = 160, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -0.5], 0.7, size=(n // 2, 2))
    x1 = rng.normal([1.2, 0.9], 0.7, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask(session: Session, fraction: float = 0.75) -> tuple[Session, pd.Series]:
    rng = np.random.default_rng(4)
    full = session.to_pandas().copy()
    truth = full["y"].copy()
    idx = list(session.split_plan.train_indices)
    blank = rng.choice(idx, size=max(1, int(fraction * len(idx))), replace=False)
    full.loc[blank, "y"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session, truth


def test_capability_matrix_sklearn_always_available() -> None:
    matrix = activelearning_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "margin" in matrix["backends"]["sklearn"]["strategies"]
    assert "human_label_boundary" in matrix
    assert "vs_semisupervised" in matrix


def test_list_activelearning_strategies_includes_sklearn() -> None:
    strategies = list_activelearning_strategies()
    assert "margin" in strategies
    assert "entropy" in strategies


def test_resolve_backend_strategy_defaults() -> None:
    backend, strategy = resolve_backend_strategy(backend=None, strategy="entropy")
    assert backend == "sklearn"
    assert strategy == "entropy"


def test_resolve_industry_backend_always_available() -> None:
    backend, strategy = resolve_backend_strategy(backend="industry", strategy="core_set")
    assert backend == "industry"
    assert strategy == "core_set"


def test_core_set_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session, truth = _mask(session)
    try:
        fit = session.fit_active_learner(
            backend="industry",
            strategy="core_set",
            prefer_reduce_components=False,
            label_budget=10,
        )
        q = session.suggest_query(batch_size=4)
    except (MissingExtraError, ValidationError, OSError) as exc:
        if any(token in str(exc).lower() for token in ("torch", "dll")):
            pytest.skip("backend not runnable on this host")
        raise
    assert fit.backend == "industry"
    assert len(q.indices) == 4
    labels = [int(truth.loc[i]) for i in q.indices]
    session.label_rows(indices=q.indices, labels=labels)
    ev = session.evaluate_active_learning(partition="test")
    assert ev.metrics["accuracy"] >= 0.0


def test_qbc_kl_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session, _ = _mask(session)
    try:
        fit = session.fit_active_learner(
            backend="industry",
            strategy="qbc_kl",
            committee_size=4,
            label_budget=8,
        )
        q = session.suggest_query(batch_size=3)
    except (MissingExtraError, ValidationError, OSError) as exc:
        if any(token in str(exc).lower() for token in ("torch", "dll")):
            pytest.skip("backend not runnable on this host")
        raise
    assert fit.backend == "industry"
    assert len(q.indices) == 3


@pytest.mark.skipif(not _torch_spec_present(), reason="torch not installed")
def test_bald_torch_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session, truth = _mask(session)
    try:
        fit = session.fit_active_learner(
            backend="torch",
            strategy="bald",
            prefer_reduce_components=False,
            label_budget=10,
            epochs=15,
            batch_size=32,
        )
    except (MissingExtraError, ValidationError) as exc:
        if "torch" in str(exc).lower():
            pytest.skip("torch installed but not importable on this host")
        raise
    assert fit.backend == "torch"
    q = session.suggest_query(batch_size=4)
    assert len(q.indices) == 4
    labels = [int(truth.loc[i]) for i in q.indices]
    session.label_rows(indices=q.indices, labels=labels)
    ev = session.evaluate_active_learning(partition="test")
    assert "accuracy" in ev.metrics


def test_activelearning_status_includes_capability_matrix() -> None:
    from buildml.activelearning.explain_hooks import activelearning_status

    status = activelearning_status()
    assert "capability_matrix" in status
    assert status["capability_matrix"]["backends"]["sklearn"]["available"]


def test_label_rows_not_ai_allowlisted() -> None:
    from buildml.ai.tools import build_default_registry

    assert build_default_registry().get("label_rows") is None


def test_invalid_backend_strategy_pairing() -> None:
    with pytest.raises(ValidationError):
        resolve_backend_strategy(backend="sklearn", strategy="bald")


def test_bundle_roundtrip_with_backend(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session, _ = _mask(session)
    session.fit_active_learner(strategy="margin", prefer_reduce_components=False)
    out = session.save_active_learning_bundle(tmp_path / "bundle")
    session2 = Session.ingest(_frame()).set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    session2.load_active_learning_bundle(out)
    assert session2.activelearning_plan is not None
    assert session2.activelearning_plan.backend == "sklearn"
