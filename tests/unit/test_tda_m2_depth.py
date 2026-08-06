"""Algorithm-depth tests for TDA (outside Session facade)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.tda.extras import tda_available
from buildml.tda.features import infer_tda_task
from buildml.tda.fit import fit_tda
from buildml.tda.homology import compute_rips_diagrams, finite_diagram
from buildml.tda.transform import transform_tda
from buildml.tda.vectorize import fit_vectorizer_state, vectorize_diagrams

pytestmark = pytest.mark.skipif(not tda_available(), reason="buildml[tda] missing")


def test_finite_diagram_drops_inf_and_zero_pers() -> None:
    dgm = np.array([[0.0, np.inf], [0.1, 0.1], [0.0, 0.5]], dtype=float)
    out = finite_diagram(dgm)
    assert out.shape == (1, 2)
    assert np.isclose(out[0, 1] - out[0, 0], 0.5)


def test_rips_and_vectorizers_smoke() -> None:
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(18, 3))
    dgms = compute_rips_diagrams(cloud, maxdim=1)
    assert len(dgms) >= 2
    train = [dgms, dgms]
    for kind in ("persistence_image", "landscape", "silhouette"):
        state = fit_vectorizer_state(
            train,
            vectorization=kind,
            homology_dims=(0, 1),
            n_bins=8,
            n_layers=2,
        )
        vec = vectorize_diagrams(dgms, state)
        assert vec.shape == (state["feature_dim"],)
        assert np.isfinite(vec).all()


def test_holdout_does_not_change_vectorizer_state() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(90, 3))
    b = rng.normal(size=(90, 3)) + np.array([2.0, 0, 0])
    x = np.vstack([a, b])
    y = np.array([0] * 90 + [1] * 90)
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    frame["y"] = y
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
    )
    plan, fit = fit_tda(
        session.dataset,
        session._split_plan,
        knn=8,
        n_bins=8,
        head="logistic_regression",
        random_state=0,
    )
    state_before = dict(plan.vectorizer_state_)
    _ = transform_tda(session.dataset, plan, session._split_plan, partition="test")
    assert plan.vectorizer_state_ == state_before
    assert fit.feature_dim == plan.feature_dim


def test_infer_task() -> None:
    assert infer_tda_task(pd.Series([0, 1, 0, 1, 0, 1])) == "classification"
    rng = np.random.default_rng(0)
    cont = pd.Series(rng.normal(size=80))
    assert infer_tda_task(cont) == "regression"
