"""Persistence diagram → fixed-length vectors (persim + in-tree silhouette)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.tda.extras import require_persim


def _collect_points(diagrams_by_dim: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Stack all finite (birth, death) pairs across samples and dimensions."""
    chunks: list[np.ndarray] = []
    for sample in diagrams_by_dim:
        for dgm in sample:
            arr = np.asarray(dgm, dtype=float)
            if arr.size:
                chunks.append(arr.reshape(-1, 2))
    if not chunks:
        return np.array([[0.0, 0.1]], dtype=float)
    return np.vstack(chunks)


def fit_vectorizer_state(
    train_diagrams: Sequence[Sequence[np.ndarray]],
    *,
    vectorization: str,
    homology_dims: Sequence[int],
    n_bins: int,
    n_layers: int,
    pixel_size: float | None = None,
) -> dict[str, Any]:
    """Fit vectorizer hyperparameters from **train** diagrams only."""
    key = str(vectorization).lower().replace("-", "_")
    dims = tuple(int(d) for d in homology_dims)
    if key == "persistence_image":
        require_persim(feature="persistence_image vectorization")
        from persim import PersistenceImager

        pts = _collect_points(train_diagrams)
        births = pts[:, 0]
        pers = pts[:, 1] - pts[:, 0]
        b_lo, b_hi = float(np.min(births)), float(np.max(births))
        p_lo, p_hi = float(np.min(pers)), float(np.max(pers))
        if abs(b_hi - b_lo) < 1e-9:
            b_hi = b_lo + 1.0
        if abs(p_hi - p_lo) < 1e-9:
            p_hi = p_lo + 1.0
        # Expand slightly so edge points land inside pixels
        b_pad = 0.05 * (b_hi - b_lo)
        p_pad = 0.05 * (p_hi - p_lo)
        birth_range = (b_lo - b_pad, b_hi + b_pad)
        pers_range = (max(0.0, p_lo - p_pad), p_hi + p_pad)
        if pixel_size is None:
            # Aim for roughly n_bins × n_bins per dimension
            px = max(
                (birth_range[1] - birth_range[0]) / max(int(n_bins), 2),
                (pers_range[1] - pers_range[0]) / max(int(n_bins), 2),
                1e-3,
            )
        else:
            px = float(pixel_size)
        pim = PersistenceImager(
            pixel_size=px,
            birth_range=birth_range,
            pers_range=pers_range,
        )
        # Probe output size on a tiny diagram
        probe = pim.transform(np.array([[birth_range[0], birth_range[0] + pers_range[0] + 1e-3]]))
        per_dim = int(np.asarray(probe).size)
        return {
            "kind": "persistence_image",
            "homology_dims": dims,
            "pixel_size": px,
            "birth_range": birth_range,
            "pers_range": pers_range,
            "per_dim": per_dim,
            "feature_dim": per_dim * len(dims),
        }

    if key == "landscape":
        # In-tree landscapes (same tent construction as silhouettes) — avoids
        # persim PersLandscapeApprox empty-grid failures on narrow H1 diagrams.
        return {
            "kind": "landscape",
            "homology_dims": dims,
            "n_bins": int(n_bins),
            "n_layers": int(n_layers),
            "per_dim": int(n_bins) * int(n_layers),
            "feature_dim": int(n_bins) * int(n_layers) * len(dims),
            "t_range": _infer_t_range(train_diagrams, dims),
        }

    if key == "silhouette":
        # In-tree weighted silhouette (Chazal-style); no extra import beyond numpy.
        return {
            "kind": "silhouette",
            "homology_dims": dims,
            "n_bins": int(n_bins),
            "per_dim": int(n_bins),
            "feature_dim": int(n_bins) * len(dims),
            "t_range": _infer_t_range(train_diagrams, dims),
            "weight_power": 1.0,
        }

    raise ValidationError(
        f"Unknown TDA vectorization {vectorization!r}; expected "
        "persistence_image, landscape, or silhouette."
    )


def _infer_t_range(
    train_diagrams: Sequence[Sequence[np.ndarray]], dims: Sequence[int]
) -> tuple[float, float]:
    pts: list[np.ndarray] = []
    for sample in train_diagrams:
        for d in dims:
            if d < len(sample):
                arr = np.asarray(sample[d], dtype=float)
                if arr.size:
                    pts.append(arr.reshape(-1, 2))
    if not pts:
        return (0.0, 1.0)
    stacked = np.vstack(pts)
    lo = float(np.min(stacked[:, 0]))
    hi = float(np.max(stacked[:, 1]))
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


def vectorize_diagrams(
    diagrams: Sequence[np.ndarray],
    state: dict[str, Any],
) -> np.ndarray:
    """Vectorize one sample's diagrams (indexed by homology dimension)."""
    kind = state["kind"]
    dims = tuple(int(d) for d in state["homology_dims"])
    parts: list[np.ndarray] = []
    if kind == "persistence_image":
        from persim import PersistenceImager

        pim = PersistenceImager(
            pixel_size=float(state["pixel_size"]),
            birth_range=tuple(state["birth_range"]),
            pers_range=tuple(state["pers_range"]),
        )
        per_dim = int(state["per_dim"])
        for d in dims:
            dgm = np.asarray(diagrams[d] if d < len(diagrams) else np.zeros((0, 2)), dtype=float)
            if dgm.size == 0:
                parts.append(np.zeros(per_dim, dtype=float))
            else:
                img = np.asarray(pim.transform(dgm), dtype=float).ravel()
                if img.size < per_dim:
                    img = np.pad(img, (0, per_dim - img.size))
                parts.append(img[:per_dim])
        return np.concatenate(parts)

    if kind == "landscape":
        t_lo, t_hi = state["t_range"]
        n_bins = int(state["n_bins"])
        n_layers = int(state["n_layers"])
        grid = np.linspace(float(t_lo), float(t_hi), n_bins)
        for d in dims:
            dgm = np.asarray(diagrams[d] if d < len(diagrams) else np.zeros((0, 2)), dtype=float)
            parts.append(_landscape_vector(dgm, grid, n_layers=n_layers))
        return np.concatenate(parts)

    if kind == "silhouette":
        t_lo, t_hi = state["t_range"]
        n_bins = int(state["n_bins"])
        power = float(state.get("weight_power", 1.0))
        grid = np.linspace(t_lo, t_hi, n_bins)
        for d in dims:
            dgm = np.asarray(diagrams[d] if d < len(diagrams) else np.zeros((0, 2)), dtype=float)
            parts.append(_silhouette_vector(dgm, grid, weight_power=power))
        return np.concatenate(parts)

    raise ValidationError(f"Unknown vectorizer kind {kind!r}.")


def _landscape_vector(
    diagram: np.ndarray, grid: np.ndarray, *, n_layers: int
) -> np.ndarray:
    """Persistence landscape λ_k(t) sampled on a fixed grid (in-tree)."""
    dgm = np.asarray(diagram, dtype=float)
    out = np.zeros(int(n_layers) * len(grid), dtype=float)
    if dgm.size == 0:
        return out
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    mids = 0.5 * (births + deaths)
    heights = 0.5 * (deaths - births)
    layers = np.zeros((int(n_layers), len(grid)), dtype=float)
    for t_idx, t in enumerate(grid):
        tents = np.maximum(0.0, heights - np.abs(t - mids))
        if tents.size == 0:
            continue
        ordered = np.sort(tents)[::-1]
        take = min(int(n_layers), ordered.size)
        layers[:take, t_idx] = ordered[:take]
    return layers.ravel()


def _silhouette_vector(
    diagram: np.ndarray, grid: np.ndarray, *, weight_power: float = 1.0
) -> np.ndarray:
    """Weighted persistence silhouette on a fixed grid (in-tree)."""
    dgm = np.asarray(diagram, dtype=float)
    out = np.zeros(len(grid), dtype=float)
    if dgm.size == 0:
        return out
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    pers = deaths - births
    weights = np.power(np.maximum(pers, 0.0), weight_power)
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return out
    for t_idx, t in enumerate(grid):
        # Tent function Λ(t) peaks at mid = (b+d)/2 with height pers/2
        mids = 0.5 * (births + deaths)
        heights = 0.5 * pers
        tents = np.maximum(0.0, heights - np.abs(t - mids))
        out[t_idx] = float(np.sum(weights * tents) / w_sum)
    return out


def feature_names_from_state(state: dict[str, Any]) -> tuple[str, ...]:
    """Stable feature names for the topological vector."""
    kind = state["kind"]
    dims = tuple(int(d) for d in state["homology_dims"])
    names: list[str] = []
    if kind == "persistence_image":
        per = int(state["per_dim"])
        for d in dims:
            for i in range(per):
                names.append(f"tda_pi_H{d}_{i}")
    elif kind == "landscape":
        n_bins = int(state["n_bins"])
        n_layers = int(state["n_layers"])
        for d in dims:
            for layer in range(n_layers):
                for i in range(n_bins):
                    names.append(f"tda_land_H{d}_L{layer}_{i}")
    elif kind == "silhouette":
        n_bins = int(state["n_bins"])
        for d in dims:
            for i in range(n_bins):
                names.append(f"tda_sil_H{d}_{i}")
    else:
        for i in range(int(state["feature_dim"])):
            names.append(f"tda_feat_{i}")
    return tuple(names)
