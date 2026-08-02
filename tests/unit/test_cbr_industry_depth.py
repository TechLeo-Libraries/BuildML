"""Industry-depth tests for CBR backends (R6.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.cbr.catalog import (
    cbr_capability_matrix,
    list_cbr_backends,
    resolve_backend_metric,
)
from buildml.cbr.extras import cbr_industry_available, hnswlib_available, text_embedding_available
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import torch_spec_available


def _frame(n: int = 160, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.7, size=(n // 2, 2))
    x1 = rng.normal(2.0, 0.7, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def test_capability_matrix_sklearn_always_available() -> None:
    matrix = cbr_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "euclidean" in matrix["backends"]["sklearn"]["metrics"]
    assert "cbr_vs_rag" in matrix


def test_list_cbr_backends_includes_sklearn() -> None:
    assert "sklearn" in list_cbr_backends()


def test_resolve_backend_metric_defaults() -> None:
    backend, metric = resolve_backend_metric(backend=None, metric="euclidean")
    assert metric == "euclidean"
    assert backend in {"sklearn", "industry", "embedding", "torch"}


def test_resolve_industry_requires_ann_when_missing() -> None:
    if cbr_industry_available():
        backend, metric = resolve_backend_metric(backend="industry", metric="euclidean")
        assert backend == "industry"
        assert metric == "euclidean"
    else:
        with pytest.raises(MissingExtraError):
            resolve_backend_metric(backend="industry", metric="euclidean")


def test_embedding_requires_text_columns() -> None:
    if not text_embedding_available():
        pytest.skip("sentence-transformers not installed")
    with pytest.raises(ValidationError, match="text_columns"):
        resolve_backend_metric(backend="embedding", metric="cosine", text_columns=None)


@pytest.mark.skipif(not hnswlib_available(), reason="hnswlib not installed")
def test_industry_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_cbr(
        backend="industry",
        task="classification",
        metric="euclidean",
        prefer_reduce_components=False,
        k=5,
    )
    assert fit.backend == "industry"
    pred = session.predict_cbr(partition="test", return_traces=True)
    assert pred.traces
    assert pred.traces[0].neighbor_case_ids
    ev = session.evaluate_cbr(partition="test")
    assert ev.metrics["accuracy"] >= 0.0
    assert session.cbr_plan is not None
    assert session.cbr_plan.backend == "industry"


@pytest.mark.skipif(not torch_spec_available(), reason="torch not installed")
def test_torch_learned_metric_session_path() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_cbr(
        backend="torch",
        task="classification",
        metric="euclidean",
        prefer_reduce_components=False,
        k=5,
        torch_epochs=15,
    )
    assert fit.backend == "torch"
    pred = session.predict_cbr(partition="test", return_traces=True)
    assert len(pred.predictions) > 0
    assert pred.traces[0].weights


def test_sklearn_fallback_always_runs() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    fit = session.fit_cbr(backend="sklearn", metric="mixed", k=3)
    assert fit.backend == "sklearn"
    neighbors = session.retrieve_cases(partition="test", k=3)
    assert neighbors.traces[0].distances
