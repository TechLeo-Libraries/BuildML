"""Vietoris–Rips persistent homology via ripser (optional ``buildml[tda]``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.tda.extras import require_ripser


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """Drop infinite death times (common for H0 essential class)."""
    dgm = np.asarray(diagram, dtype=float)
    if dgm.size == 0:
        return np.zeros((0, 2), dtype=float)
    if dgm.ndim != 2 or dgm.shape[1] != 2:
        raise ValidationError(f"Persistence diagram must be (n, 2); got {dgm.shape}.")
    mask = np.isfinite(dgm).all(axis=1)
    # Also drop zero-persistence points
    mask &= (dgm[:, 1] - dgm[:, 0]) > 1e-12
    return dgm[mask]


def compute_rips_diagrams(
    points: np.ndarray,
    *,
    maxdim: int = 1,
    thresh: float | None = None,
) -> list[np.ndarray]:
    """Compute Vietoris–Rips persistence diagrams for a point cloud.

    Returns a list of diagrams indexed by homology dimension (H0, H1, …),
    each shaped ``(n_points, 2)`` as ``(birth, death)``.
    """
    ripser_mod = require_ripser(feature="fit_tda / transform_tda (ripser)")
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2:
        raise ValidationError(f"Point cloud must be 2-D; got shape {cloud.shape}.")
    if cloud.shape[0] < 2:
        # Degenerate: return empty diagrams
        return [np.zeros((0, 2), dtype=float) for _ in range(int(maxdim) + 1)]
    kwargs: dict[str, Any] = {"maxdim": int(maxdim)}
    if thresh is not None:
        kwargs["thresh"] = float(thresh)
    out = ripser_mod.ripser(cloud, **kwargs)
    diagrams = out.get("dgms")
    if diagrams is None:
        raise ValidationError("ripser returned no diagrams.")
    return [finite_diagram(d) for d in diagrams]


def local_point_cloud(
    query: np.ndarray,
    neighbor_index: Any,
    train_x: np.ndarray,
    *,
    knn: int,
) -> np.ndarray:
    """Build a local cloud from ``knn`` nearest **train** neighbors of ``query``."""
    k = int(min(max(knn, 2), len(train_x)))
    dists, idxs = neighbor_index.kneighbors(
        np.asarray(query, dtype=float).reshape(1, -1), n_neighbors=k
    )
    _ = dists
    return np.asarray(train_x[idxs[0]], dtype=float)
