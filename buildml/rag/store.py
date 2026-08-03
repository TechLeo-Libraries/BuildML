"""Hold the vectors, and find the nearest ones to a query.

Similarity search over unit-length vectors is a matrix multiply. Normalise every
row when the store is built, normalise the query, and the dot product *is* the
cosine similarity: no per-query normalisation, no division, one BLAS call
across the whole corpus.

The default store does exactly that and nothing else: exact, brute-force, and
fast enough that tens of thousands of chunks search in milliseconds. Approximate
indexes exist for corpora large enough to need them, and trade recall for speed;
until that trade is measured rather than assumed, exact is the right default.

See Also
--------
buildml.rag.index : What builds a store.
buildml.rag.retrieve : What queries one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rag.hybrid import match_metadata_filters
from buildml.rag.results import Chunk, Hit


class VectorStore(Protocol):
    """What the RAG path needs from anything holding vectors.

    Parallel arrays and one search method. Row ``i`` of ``embeddings`` is the
    vector for ``chunks[i]``, and that correspondence is the store's whole
    contract: break it and every result points at the wrong passage.

    Attributes
    ----------
    chunks:
        The passages, in the same order as the vectors.
    embeddings:
        Shape ``(n_chunks, dim)``, L2-normalised.
    dim:
        Vector width. Queries must match.

    Notes
    -----
    **The alignment is positional and unchecked at query time.** Any operation
    that reorders or filters one array must do the same to the other.

    See Also
    --------
    NumpyCosineStore : The default implementation.
    """

    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray
    dim: int

    def query(
        self,
        vector: np.ndarray,
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Return the ``k`` chunks nearest to a query vector.

        The one search operation the retrieval path needs from a store.

        Parameters
        ----------
        vector:
            The embedded query, of width ``dim``.
        k:
            How many hits to return.
        filters:
            Metadata equality constraints applied before ranking.

        Returns
        -------
        list of Hit
            Best first, ranked from 1. May be shorter than ``k``.

        Notes
        -----
        **Filters apply before ranking**, so filtering never leaves a gap in
        the results: it changes which chunks were eligible in the first place.
        """
        ...


@dataclass
class NumpyCosineStore:
    """Exact cosine search over every vector, in one matrix multiply.

    No approximation and no index structure: the query is compared against all
    ``n_chunks`` vectors. That sounds expensive and generally is not: a single
    BLAS matrix-vector product over tens of thousands of rows takes
    milliseconds, and the result is exactly correct rather than probably close.

    Attributes
    ----------
    chunks:
        The passages, aligned with the vector rows.
    embeddings:
        Shape ``(n_chunks, dim)``, float32, L2-normalised at build time.
    dim:
        Vector width.
    backend:
        Identifier recorded on results.
    metadata:
        Store-level notes.

    Notes
    -----
    **Everything is resident in memory.** Roughly ``n_chunks × dim × 4`` bytes
   : a million chunks at 384 dimensions is about 1.5 GB, plus the chunk text.

    **Cost is linear in corpus size.** Search time grows in proportion, which is
    the point at which an approximate index starts to earn its recall loss.

    **Nothing is persisted.** Saving the index is
    :mod:`buildml.rag.checkpoint`'s job.

    See Also
    --------
    VectorStore : The contract.
    buildml.rag.checkpoint : Persisting a store.
    """

    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray
    dim: int
    backend: str = "numpy_cosine"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        chunks: tuple[Chunk, ...] | list[Chunk],
        embeddings: np.ndarray,
    ) -> NumpyCosineStore:
        """Pair chunks with their vectors, normalising once.

        Normalisation happens here rather than per query, so that every later
        search is a plain dot product. The row count is checked against the
        chunk count, because a mismatch would mean every hit reports the wrong
        passage: with no error and no obvious symptom.

        Parameters
        ----------
        chunks:
            The passages.
        embeddings:
            Shape ``(n_chunks, dim)``, in the same order as the chunks.

        Returns
        -------
        NumpyCosineStore
            The store, with unit-length rows.

        Raises
        ------
        ValidationError
            If the array is not two-dimensional, or its row count does not
            match the chunk count.

        Notes
        -----
        **Order is trusted.** The counts are checked; the correspondence is not,
        and cannot be.

        **Zero vectors are handled but meaningless.** They survive
        normalisation through a small epsilon and then sit at zero similarity to
        everything.
        """
        chunk_tuple = tuple(chunks)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValidationError(f"embeddings must be 2-D; got shape {matrix.shape}")
        if matrix.shape[0] != len(chunk_tuple):
            raise ValidationError(
                f"embeddings rows ({matrix.shape[0]}) != n_chunks ({len(chunk_tuple)})"
            )
        # Ensure unit rows for cosine via dot product.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        matrix = matrix / norms
        return cls(chunks=chunk_tuple, embeddings=matrix, dim=int(matrix.shape[1]))

    def query(
        self,
        vector: np.ndarray,
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Score every chunk against the query and return the best ``k``.

        One matrix-vector product produces all the scores; a partial sort then
        pulls out the top ``k`` without ordering the rest, which is what keeps
        this fast on a large corpus.

        Parameters
        ----------
        vector:
            The embedded query. Must be of width ``dim``.
        k:
            How many hits to return.
        filters:
            Metadata equality constraints. Only matching chunks are eligible.

        Returns
        -------
        list of Hit
            Best first, ranked from 1. Empty when the store is empty or the
            filters match nothing.

        Raises
        ------
        ValidationError
            If ``k`` is not positive, or the query width does not match the
            index. **A width mismatch usually means the query was embedded by a
            different model than the index**: the check is what turns that into
            an error rather than confident nonsense.

        Notes
        -----
        **Ties are broken by position**, using a stable sort, so repeated
        queries return the same order.

        **Filtering does not make search cheaper.** Every chunk is still scored;
        the filter only decides which scores are eligible.

        **Matching dimensions do not prove matching models.** Two different
        models with the same width pass this check and produce meaningless
        rankings; the embedder identity recorded on the index is the real
        guard.
        """
        if k <= 0:
            raise ValidationError(f"k must be positive; got {k}")
        if self.embeddings.shape[0] == 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValidationError(
                f"Query dim {q.shape[0]} does not match index dim {self.dim}"
            )
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        scores = self.embeddings @ q
        if filters:
            mask = np.array(
                [match_metadata_filters(c.metadata, filters) for c in self.chunks],
                dtype=bool,
            )
            if not bool(mask.any()):
                return []
            eligible = np.flatnonzero(mask)
            eligible_scores = scores[eligible]
            top_k = min(k, eligible_scores.shape[0])
            local = np.argpartition(-eligible_scores, top_k - 1)[:top_k]
            local = local[np.argsort(-eligible_scores[local], kind="stable")]
            idx = eligible[local]
        else:
            top_k = min(k, scores.shape[0])
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
            idx = idx[np.argsort(-scores[idx], kind="stable")]
        hits: list[Hit] = []
        for rank, i in enumerate(idx, start=1):
            chunk = self.chunks[int(i)]
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(scores[int(i)]),
                    text=chunk.text,
                    rank=rank,
                    metadata=dict(chunk.metadata),
                )
            )
        return hits

    def without_ids(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> NumpyCosineStore:
        """Return a store without the named chunks or documents.

        Deletion without re-embedding: surviving rows keep the vectors they
        already had. Removing a stale document from an index is therefore cheap,
        where a rebuild would mean re-encoding everything.

        Parameters
        ----------
        chunk_ids:
            Specific chunks to drop.
        doc_ids:
            Drop every chunk of these documents.

        Returns
        -------
        NumpyCosineStore
            A new store. The original is unchanged, and is returned as-is when
            nothing matched.

        Notes
        -----
        **Unknown identifiers are ignored.** Deleting something that is not
        there is not an error, so a typo removes nothing and says nothing :
        compare chunk counts to confirm.

        **The result can be empty**, and an empty store returns no hits for
        every query rather than failing.
        """
        drop_chunks = set(chunk_ids or ())
        drop_docs = set(doc_ids or ())
        keep_idx = [
            i
            for i, c in enumerate(self.chunks)
            if c.chunk_id not in drop_chunks and c.doc_id not in drop_docs
        ]
        if len(keep_idx) == len(self.chunks):
            return self
        if not keep_idx:
            empty = np.zeros((0, self.dim), dtype=np.float32)
            return NumpyCosineStore(chunks=(), embeddings=empty, dim=self.dim)
        kept_chunks = tuple(self.chunks[i] for i in keep_idx)
        kept_emb = self.embeddings[np.asarray(keep_idx, dtype=np.int64)]
        return NumpyCosineStore(chunks=kept_chunks, embeddings=kept_emb, dim=self.dim)
