"""Approximate nearest-neighbor index for industry CBR retrieval."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.cbr.extras import require_ann_library
from buildml.core.errors import ValidationError


def _as_float32(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValidationError("ANN index expects a 2-d float matrix.")
    if arr.shape[0] == 0:
        raise ValidationError("Cannot build ANN index on empty case memory.")
    return arr


def build_ann_index(
    vectors: np.ndarray,
    *,
    metric: str = "euclidean",
    ef_construction: int = 200,
    m: int = 16,
) -> tuple[Any, str]:
    """Build hnswlib (preferred) or faiss index on case feature vectors."""
    data = _as_float32(vectors)
    dim = int(data.shape[1])
    if dim == 0:
        raise ValidationError("ANN index requires non-empty feature dimension.")
    lib = require_ann_library()
    metric_key = str(metric).lower().replace("-", "_")
    space = "l2" if metric_key == "euclidean" else "cosine"
    if lib == "hnswlib":
        import hnswlib

        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(
            max_elements=int(data.shape[0]),
            ef_construction=int(ef_construction),
            M=int(m),
        )
        index.add_items(data, np.arange(data.shape[0], dtype=np.int64))
        index.set_ef(max(int(ef_construction), 50))
        return index, "hnswlib"
    import faiss

    if metric_key == "cosine":
        faiss.normalize_L2(data)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(data)
    return index, "faiss"


def query_ann_index(
    index: Any,
    library: str,
    queries: np.ndarray,
    *,
    k: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Query ANN index; return (distances, indices) shaped (n_query, k)."""
    q = _as_float32(queries)
    kk = min(int(k), max(1, getattr(index, "ntotal", q.shape[0])))
    metric_key = str(metric).lower().replace("-", "_")
    if library == "hnswlib":
        labels, dists = index.knn_query(q, k=kk)
        dists_arr = np.asarray(dists, dtype=float)
        if metric_key == "cosine":
            # hnswlib cosine distance is 1 - similarity
            dists_arr = np.clip(dists_arr, 0.0, 2.0)
        return dists_arr, np.asarray(labels, dtype=int)
    import faiss

    qf = q.copy()
    if metric_key == "cosine":
        faiss.normalize_L2(qf)
        sims, labels = index.search(qf, kk)
        dists_arr = 1.0 - np.clip(sims, -1.0, 1.0)
    else:
        dists_arr, labels = index.search(qf, kk)
    return np.asarray(dists_arr, dtype=float), np.asarray(labels, dtype=int)


def add_vectors_to_ann_index(
    index: Any,
    library: str,
    vectors: np.ndarray,
    *,
    start_id: int,
) -> Any:
    """Append vectors to an existing index (retain path). Returns updated index."""
    data = _as_float32(vectors)
    if data.shape[0] == 0:
        return index
    if library == "hnswlib":
        ids = np.arange(start_id, start_id + data.shape[0], dtype=np.int64)
        index.add_items(data, ids)
        return index
    import faiss

    index.add(data)
    return index
