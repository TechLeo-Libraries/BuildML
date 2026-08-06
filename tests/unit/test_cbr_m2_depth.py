"""Low-level depth tests for CBR distances / reuse / retain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.cbr.cases import pairwise_distances, top_k_indices
from buildml.cbr.features import standardize_fit
from buildml.cbr.predict import reuse_solutions
from buildml.core.errors import ValidationError


def test_pairwise_metrics_shapes() -> None:
    memory = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    query = np.array([[0.0, 0.0]], dtype=float)
    d_e = pairwise_distances(query, memory, metric="euclidean")
    assert d_e.shape == (1, 3)
    assert d_e[0, 0] == pytest.approx(0.0)
    d_m = pairwise_distances(query, memory, metric="manhattan")
    assert d_m[0, 1] == pytest.approx(1.0)
    d_c = pairwise_distances(query, memory, metric="cosine")
    assert d_c.shape == (1, 3)

    q_cat = np.array([[0]], dtype=int)
    m_cat = np.array([[0], [1], [0]], dtype=int)
    d_mix = pairwise_distances(
        query,
        memory,
        metric="mixed",
        query_cat=q_cat,
        memory_cat=m_cat,
        numeric_ranges=np.array([1.0, 1.0]),
    )
    assert d_mix.shape == (1, 3)
    assert d_mix[0, 0] < d_mix[0, 1]


def test_top_k_and_reuse_modes() -> None:
    d = np.array([0.5, 0.1, 0.9, 0.2])
    order = top_k_indices(d, 2)
    assert list(order) == [1, 3]

    pred_maj, _ = reuse_solutions(
        neighbors=[0, 1, 1],
        weights=np.array([1.0, 1.0, 1.0]),
        neighbor_features=np.zeros((3, 1)),
        query_features=np.zeros(1),
        task="classification",
        reuse="majority",
        adapt="none",
    )
    assert pred_maj == 1

    pred_w, notes = reuse_solutions(
        neighbors=[0, 1],
        weights=np.array([10.0, 0.1]),
        neighbor_features=np.zeros((2, 1)),
        query_features=np.zeros(1),
        task="classification",
        reuse="distance_weighted",
        adapt="none",
    )
    assert pred_w == 0
    assert notes

    pred_r, _ = reuse_solutions(
        neighbors=[1.0, 3.0, 5.0],
        weights=np.array([1.0, 1.0, 1.0]),
        neighbor_features=np.array([[0.0], [1.0], [2.0]]),
        query_features=np.array([1.0]),
        task="regression",
        reuse="local_ridge",
        adapt="none",
    )
    assert isinstance(pred_r, float)


def test_standardize_fit_stable() -> None:
    x = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=float)
    xs, mean, scale = standardize_fit(x)
    assert mean[0] == pytest.approx(2.0)
    assert xs.mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-10)


def test_majority_rejects_regression_reuse() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=80),
            "b": rng.normal(size=80),
            "y": rng.normal(size=80),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    with pytest.raises(ValidationError, match="classification-only"):
        session.fit_cbr(task="regression", reuse="majority")
