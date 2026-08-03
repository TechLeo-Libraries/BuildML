"""Session-facing slice tests for Topological Data Analysis."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, MissingExtraError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.tda.extras import tda_available


def _demo_session(n: int = 160) -> Session:
    rng = np.random.default_rng(11)
    a = rng.normal(size=(n // 2, 3))
    b = rng.normal(size=(n - n // 2, 3)) * 1.5 + np.array([2.0, 0.0, 0.0])
    x = np.vstack([a, b])
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    frame["y"] = y
    return (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )


def test_core_import_and_catalog() -> None:
    import buildml.tda as tda

    assert hasattr(tda, "fit_tda")
    assert hasattr(Session, "fit_tda")
    for op in (
        "fit_tda",
        "transform_tda",
        "predict_tda",
        "evaluate_tda",
        "save_tda_bundle",
        "load_tda_bundle",
    ):
        assert op in OPERATION_CATALOG
    assert "tda-persistent-homology" in OPERATION_CATALOG["fit_tda"].concept_links
    assert "tda-vectorization" in OPERATION_CATALOG["fit_tda"].concept_links
    assert "tda-bundle-boundary" in OPERATION_CATALOG["save_tda_bundle"].concept_links

    registry = build_default_registry()
    for name in ("fit_tda", "transform_tda", "evaluate_tda", "save_tda_bundle"):
        assert name in registry


def test_fit_requires_split() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0], "y": [0, 1, 0, 1]})
    session = Session.ingest(frame).set_roles(
        {"a": "feature", "b": "feature", "y": "target"}
    )
    with pytest.raises((LeakageError, ValidationError)):
        session.fit_tda()


@pytest.mark.skipif(not tda_available(), reason="buildml[tda] missing")
def test_session_fit_transform_evaluate_bundle(tmp_path: Path) -> None:
    session = _demo_session()
    fit = session.fit_tda(
        vectorization="persistence_image",
        knn=10,
        n_bins=10,
        head="logistic_regression",
        random_state=0,
    )
    assert fit.feature_dim > 0
    assert session.tda_plan is not None
    tr = session.transform_tda(partition="test")
    assert tr.features.shape == (tr.n_rows, fit.feature_dim)
    pred = session.predict_tda(partition="test")
    assert pred.n_rows == tr.n_rows
    ev = session.evaluate_tda(partition="validation")
    assert "accuracy" in ev.metrics

    session.save_tda_bundle(tmp_path / "tda")
    other = _demo_session()
    other.load_tda_bundle(tmp_path / "tda", trusted=True)
    assert other.tda_plan is not None
    assert other.evaluate_tda(partition="test").n_rows > 0


@pytest.mark.skipif(not tda_available(), reason="buildml[tda] missing")
@pytest.mark.parametrize("vectorization", ["landscape", "silhouette"])
def test_vectorization_variants(vectorization: str) -> None:
    session = _demo_session(n=120)
    fit = session.fit_tda(
        vectorization=vectorization,  # type: ignore[arg-type]
        knn=8,
        n_bins=8,
        n_layers=2,
        head="random_forest",
        random_state=0,
    )
    assert fit.vectorization == vectorization
    assert session.evaluate_tda(partition="validation").metrics


def test_missing_extra_raises() -> None:
    session = _demo_session(n=80)
    with mock.patch("buildml.tda.extras.require_tda_stack", side_effect=MissingExtraError("tda", "fit_tda")):
        # Patch at the call site used by fit
        with mock.patch(
            "buildml.tda.fit.require_tda_stack",
            side_effect=MissingExtraError("tda", "fit_tda"),
        ):
            with pytest.raises(MissingExtraError, match="buildml\\[tda\\]"):
                session.fit_tda()


def test_head_none_blocks_evaluate() -> None:
    if not tda_available():
        pytest.skip("buildml[tda] missing")
    session = _demo_session(n=100)
    session.fit_tda(head="none", knn=8, n_bins=8, random_state=0)
    with pytest.raises(ValidationError, match="head"):
        session.evaluate_tda()
    tr = session.transform_tda(partition="test")
    assert tr.feature_dim > 0
