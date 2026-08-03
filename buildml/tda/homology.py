"""Vietoris–Rips persistent homology via ripser (optional ``buildml[tda]``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.tda.extras import require_ripser


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """Remove infinite death times and zero-persistence noise from one diagram.

    ripser often emits an essential H0 class with infinite death; downstream
    vectorizers need finite ``(birth, death)`` pairs only.

    Parameters
    ----------
    diagram:
        Persistence diagram shaped ``(n, 2)`` with columns ``birth`` and ``death``.

    Returns
    -------
    numpy.ndarray
        Filtered diagram with finite pairs where ``death - birth > 1e-12``.

    Raises
    ------
    ValidationError
        When ``diagram`` is not two-dimensional with two columns.
    """
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
    """Compute Vietoris–Rips persistence diagrams for a local point cloud.

    Wraps ripser on a single cloud (typically ``knn`` train neighbors around one
    row). Each list entry is one homology dimension's finite diagram.

    Parameters
    ----------
    points:
        Point cloud shaped ``(n_points, n_features)``.
    maxdim:
        Maximum homology dimension to compute (H0 through H``maxdim``).
    thresh:
        Optional ripser filtration cutoff. ``None`` uses ripser defaults.

    Returns
    -------
    list[numpy.ndarray]
        Diagrams indexed by dimension; each array is ``(n_pairs, 2)`` as
        ``(birth, death)``.

    Raises
    ------
    ValidationError
        When ``points`` is not two-dimensional or ripser returns no diagrams.
    MissingExtraError
        When ``buildml[tda]`` (ripser) is not installed.
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
    """Build a local point cloud from ``knn`` nearest train neighbors of ``query``.

    Each tabular row becomes a small Euclidean neighborhood in feature space;
    persistent homology runs on that cloud. Neighbors always come from the frozen
    train matrix — holdout rows never enter the index.

    Parameters
    ----------
    query:
        One row vector in the same feature space as ``train_x``.
    neighbor_index:
        Fitted ``sklearn.neighbors.NearestNeighbors`` on train rows.
    train_x:
        Standardized or raw train design matrix used to fit ``neighbor_index``.
    knn:
        Number of neighbors to include (capped by train size).

    Returns
    -------
    numpy.ndarray
        Local cloud shaped ``(knn, n_features)``.
    """
    k = int(min(max(knn, 2), len(train_x)))
    dists, idxs = neighbor_index.kneighbors(
        np.asarray(query, dtype=float).reshape(1, -1), n_neighbors=k
    )
    _ = dists
    return np.asarray(train_x[idxs[0]], dtype=float)
