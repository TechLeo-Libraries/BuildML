"""Giotto's homology/grid axes form features, not additional samples."""

import numpy as np
import pytest

from buildml.tda.adapters.giotto import fit_giotto_vectorizer_state, vectorize_giotto_diagrams


@pytest.mark.parametrize(
    ("kind", "width"),
    [("betti_curve", 8), ("persistence_image", 32), ("persistence_landscape", 16)],
)
def test_giotto_preserves_one_feature_row_per_diagram(kind, width):
    pytest.importorskip("gtda")
    diagrams = [
        [np.array([[0.0, 1.0 + i / 10]]), np.array([[0.2, 0.8 + i / 10]])]
        for i in range(3)
    ]
    state = fit_giotto_vectorizer_state(
        diagrams, vectorization=kind, homology_dims=(0, 1), n_bins=4, n_layers=2,
    )
    rows = np.stack([vectorize_giotto_diagrams(diagram, state) for diagram in diagrams])
    assert state["feature_dim"] == width
    assert rows.shape == (len(diagrams), width)
    assert np.isfinite(rows).all()


@pytest.mark.parametrize(
    ("kind", "width"),
    [("betti_curve", 8), ("persistence_image", 32), ("persistence_landscape", 16)],
)
def test_giotto_retains_empty_training_homology_axes(kind, width):
    pytest.importorskip("gtda")
    from buildml.tda.vectorize import feature_names_from_state

    empty = np.empty((0, 2))
    diagrams = [[np.array([[0.0, 1.0 + i / 10]]), empty] for i in range(3)]
    state = fit_giotto_vectorizer_state(
        diagrams, vectorization=kind, homology_dims=(0, 1), n_bins=4, n_layers=2,
    )
    assert state["feature_dim"] == width
    names = feature_names_from_state(state)
    assert len(names) == width
    assert sum("_H0_" in name for name in names) == width // 2
    assert sum("_H1_" in name for name in names) == width // 2
    for diagram in (diagrams[0], [empty, empty]):
        row = vectorize_giotto_diagrams(diagram, state)
        assert row.shape == (width,)
        assert np.isfinite(row).all()
        np.testing.assert_array_equal(row[width // 2:], 0)
