"""giotto-tda batch packing must equalize homology counts per dimension."""

from __future__ import annotations

import numpy as np

from buildml.tda.adapters.giotto import _diagrams_to_giotto_batch


def test_giotto_batch_pads_per_homology_dimension() -> None:
    """Unequal H0/H1 counts must not share a global pad across dimensions.

    giotto-tda raises ValueError when samples disagree on how many
    points sit in each homology dimension. Pad trivial birth==death
    bars inside each dim so the stacked tensor is rectangular per dim.
    """
    sample_a = [
        np.array([[0.1, 0.4], [0.2, 0.9]]),  # H0: 2
        np.array([[0.3, 0.7]]),  # H1: 1
    ]
    sample_b = [
        np.array([[0.05, 0.2]]),  # H0: 1
        np.array([[0.15, 0.5], [0.25, 0.8], [0.35, 0.95]]),  # H1: 3
    ]
    batch = _diagrams_to_giotto_batch([sample_a, sample_b], dims=(0, 1))
    assert batch.shape == (2, 5, 3)

    h0_a = batch[0, batch[0, :, 2] == 0.0]
    h1_a = batch[0, batch[0, :, 2] == 1.0]
    h0_b = batch[1, batch[1, :, 2] == 0.0]
    h1_b = batch[1, batch[1, :, 2] == 1.0]
    assert h0_a.shape[0] == h0_b.shape[0] == 2
    assert h1_a.shape[0] == h1_b.shape[0] == 3

    trivial_h0_b = h0_b[np.isclose(h0_b[:, 0], h0_b[:, 1])]
    trivial_h1_a = h1_a[np.isclose(h1_a[:, 0], h1_a[:, 1])]
    assert trivial_h0_b.shape[0] == 1
    assert trivial_h1_a.shape[0] == 2


def test_giotto_batch_empty_diagrams_stay_rectangular() -> None:
    empty = [np.zeros((0, 2)), np.zeros((0, 2))]
    batch = _diagrams_to_giotto_batch([empty, empty], dims=(0, 1))
    assert batch.shape == (2, 1, 3)
