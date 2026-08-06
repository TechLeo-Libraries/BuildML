"""Approximate nearest-neighbour search, for case bases too large to scan.

Exact search checks every case, which is correct and linear. Past some scale
that becomes the dominant cost of every prediction, and the usual answer is to
stop checking everything: build a graph or a partition over the vectors and
follow it toward the query's neighbourhood, visiting a fraction of the data.

The trade is recall. Approximate search can miss a true nearest neighbour :
rarely, but it can, and no parameter setting eliminates it. Over a large memory
where the tenth-nearest and the twelfth-nearest are much of a muchness, that is
a good bargain. Over a few thousand cases it buys nothing that exact search was
not already giving away.

Only Euclidean and cosine distance are supported, because that is what these
index structures are built around. Manhattan and mixed distances need exact
search.

Both hnswlib and faiss are supported, with hnswlib preferred as the lighter
dependency. Their internal distance conventions differ, so this module
normalises them: whatever comes out is a distance where smaller means more
similar.

See Also
--------
buildml.cbr.adapters.sklearn_retrieval : Exact search.
buildml.cbr.extras.require_ann_library : The dependency gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.cbr.extras import require_ann_library
from buildml.core.errors import ValidationError


def _as_float32(matrix: np.ndarray) -> np.ndarray:
    """Coerce a matrix to the float32 layout both index libraries require.

    Neither library accepts float64, and passing it produces either a confusing
    type error or a silent conversion; doing it here makes the requirement
    explicit and the failure modes uniform.

    Parameters
    ----------
    matrix:
        The vectors to index or query.

    Returns
    -------
    numpy.ndarray
        A two-dimensional float32 array.

    Raises
    ------
    ValidationError
        If the input is not two-dimensional or has no rows.

    Notes
    -----
    **float32 halves the memory and loses precision that does not matter here.**
    Approximate search is approximate; the difference between float32 and
    float64 distances is far below the error the index itself introduces.
    """
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
    """Build a searchable index over the case vectors.

    Constructs an HNSW graph under hnswlib, or a flat index under faiss. The
    graph is where the speed comes from: search descends through progressively
    denser layers toward the query, visiting a small fraction of the data
    instead of all of it.

    Parameters
    ----------
    vectors:
        Case feature vectors, one row each.
    metric:
        ``'euclidean'`` or ``'cosine'``. Anything else is treated as cosine.
    ef_construction:
        How thoroughly the graph is explored while being built. Higher gives a
        better-connected graph and better recall later, at the cost of build
        time. It cannot be changed after construction.
    m:
        Connections per node. Higher improves recall and increases memory,
        roughly linearly.

    Returns
    -------
    tuple
        ``(index, library)``: the built index and which library built it.

    Raises
    ------
    ValidationError
        If the matrix is empty or has no feature dimension.
    MissingExtraError
        If neither library is installed.

    Notes
    -----
    **The defaults are conservative and generally sufficient.** ``M=16`` with
    ``ef_construction=200`` is the usual recommendation and gives high recall on
    typical data. Raise them only with a measurement showing recall is the
    problem.

    **The faiss path uses a flat index, which is exact.** It is fast because
    faiss's inner loops are heavily optimised, not because it approximates :
    so an installation with faiss but not hnswlib gets speed without any recall
    loss.

    **Cosine indexes normalise the vectors in place under faiss.** The input
    array is modified; pass a copy if you still need it.

    See Also
    --------
    query_ann_index : Searching what this builds.
    add_vectors_to_ann_index : Extending it during retention.
    """
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
    """Search the index, returning distances on a consistent convention.

    The two libraries disagree about what they return: faiss gives inner
    products for cosine indexes, where larger means *more* similar: so this
    normalises everything to a distance where smaller means more similar, and
    callers need not know which library is underneath.

    Parameters
    ----------
    index:
        The built index.
    library:
        ``'hnswlib'`` or ``'faiss'``.
    queries:
        Query vectors, one row each.
    k:
        Neighbours per query, clamped to the index size.
    metric:
        ``'euclidean'`` or ``'cosine'``. Must match how the index was built.

    Returns
    -------
    tuple
        ``(distances, indices)``, both shaped ``(n_queries, k)``, nearest first.

    Raises
    ------
    ValidationError
        If the query matrix is empty or not two-dimensional.

    Notes
    -----
    **The returned neighbours may not be the true nearest** under hnswlib. That
    is the nature of graph search and the reason the backend is fast.

    **A metric mismatch produces plausible nonsense.** Querying a Euclidean
    index with cosine normalisation returns confidently ranked, meaningless
    neighbours. The plan carries the metric so this cannot happen through the
    normal path.

    **Distances are not comparable across libraries.** hnswlib returns squared
    L2 where faiss returns L2; the ranking is the same, the numbers are not.
    """
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
    """Add retained cases to the index without rebuilding it.

    Retention would be unusable if every added case cost a full index rebuild.
    Both libraries support incremental insertion, so new vectors join the
    existing structure.

    Parameters
    ----------
    index:
        The existing index.
    library:
        ``'hnswlib'`` or ``'faiss'``.
    vectors:
        New case vectors, in the same space and dimension as the index.
    start_id:
        The label for the first new vector. Must continue the case base's
        indexing so labels keep matching case positions.

    Returns
    -------
    object
        The index, updated in place and returned for convenience.

    Notes
    -----
    **Labels must line up with case positions.** Retrieval uses returned labels
    as indices into the case list, so a wrong ``start_id`` silently returns the
    wrong cases as neighbours.

    **Incremental insertion degrades the graph over time.** HNSW quality depends
    on the order vectors arrive in, and a heavily-extended index has measurably
    worse recall than one rebuilt from the same data. Refit periodically after
    substantial retention.

    **hnswlib has a fixed capacity** set at construction, so insertion can fail
    once the case base has grown well past its original size.
    """
    data = _as_float32(vectors)
    if data.shape[0] == 0:
        return index
    if library == "hnswlib":
        ids = np.arange(start_id, start_id + data.shape[0], dtype=np.int64)
        index.add_items(data, ids)
        return index

    index.add(data)
    return index
