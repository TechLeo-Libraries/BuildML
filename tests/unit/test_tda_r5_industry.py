"""Industry-depth tests for TDA backends (giotto optional)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.tda.catalog import tda_capability_matrix
from buildml.tda.extras import giotto_available, tda_available
from buildml.tda.subsample import apply_train_subsample


pytestmark = pytest.mark.skipif(not tda_available(), reason="buildml[tda] missing")


def test_capability_matrix_native() -> None:
    matrix = tda_capability_matrix()
    assert "native" in matrix["backends"]
    assert matrix["backends"]["native"]["available"] is True
    assert "persistence_image" in matrix["backends"]["native"]["vectorizations"]


def test_subsample_random_disclosure() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    frame["y"] = [0, 1] * 25
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )
    plan = session._split_plan
    assert plan is not None
    train = frame.iloc[list(plan.train_indices)]
    _, _, disclosures, _ = apply_train_subsample(
        plan,
        train,
        max_points=20,
        strategy="random",
        target_column="y",
        random_state=0,
    )
    assert any("Random train subsample" in d for d in disclosures)


def test_evaluate_diagram_distances_smoke() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(80, 3))
    b = rng.normal(size=(80, 3)) + 2.0
    x = np.vstack([a, b])
    y = np.array([0] * 80 + [1] * 80)
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    frame["y"] = y
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
    )
    session.fit_tda(knn=8, n_bins=8, head="logistic_regression", random_state=0)
    ev = session.evaluate_tda(
        partition="validation",
        compare_diagram_distances=True,
        diagram_distance_metric="wasserstein",
    )
    assert ev.metrics
    assert isinstance(ev.diagram_distances, dict)


@pytest.mark.skipif(not giotto_available(), reason="buildml[tda-industry] missing")
def test_giotto_backend_smoke() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 4))
    x[50:, 0] += 3.0
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = [0] * 50 + [1] * 50
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )
    fit = session.fit_tda(
        backend="giotto",
        vectorization="betti_curve",
        knn=10,
        n_bins=10,
        head="logistic_regression",
        mapper=True,
        random_state=0,
    )
    assert fit.backend == "giotto"
    assert session.tda_plan is not None
