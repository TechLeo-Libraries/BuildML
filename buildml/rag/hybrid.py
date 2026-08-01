"""Lexical BM25 retrieval and dense/sparse rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.results import Chunk, Hit
from buildml.rag.types import DEFAULT_BM25_B, DEFAULT_BM25_K1, DEFAULT_RRF_K

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 paths."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """In-process Okapi BM25 over chunk texts (no optional dependency)."""

    chunks: tuple[Chunk, ...]
    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B
    _doc_tokens: tuple[list[str], ...] = field(default_factory=tuple, repr=False)
    _doc_len: tuple[int, ...] = field(default_factory=tuple, repr=False)
    _avgdl: float = 0.0
    _df: dict[str, int] = field(default_factory=dict, repr=False)
    _n_docs: int = 0

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> BM25Index:
        chunk_tuple = tuple(chunks)
        doc_tokens = [tokenize(c.text) for c in chunk_tuple]
        doc_len = [len(toks) for toks in doc_tokens]
        n_docs = len(chunk_tuple)
        avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0
        df: dict[str, int] = {}
        for toks in doc_tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        return cls(
            chunks=chunk_tuple,
            k1=float(k1),
            b=float(b),
            _doc_tokens=tuple(doc_tokens),
            _doc_len=tuple(doc_len),
            _avgdl=float(avgdl),
            _df=df,
            _n_docs=n_docs,
        )

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # Robertson/Sparck Jones IDF with +1 smoothing
        return math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        """Return BM25 scores aligned with ``chunks`` order."""
        q_terms = tokenize(query)
        if not q_terms or self._n_docs == 0:
            return [0.0] * self._n_docs
        q_tf = Counter(q_terms)
        scores = [0.0] * self._n_docs
        avgdl = self._avgdl if self._avgdl > 0 else 1.0
        for i, toks in enumerate(self._doc_tokens):
            if not toks:
                continue
            tf_map = Counter(toks)
            dl = self._doc_len[i]
            score = 0.0
            for term, q_weight in q_tf.items():
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                score += q_weight * idf * (tf * (self.k1 + 1.0) / denom)
            scores[i] = score
        return scores

    def query(self, query: str, *, k: int) -> list[Hit]:
        if k <= 0:
            raise ValidationError(f"k must be positive; got {k}")
        if self._n_docs == 0:
            return []
        scores = self.score(query)
        top_k = min(k, self._n_docs)
        # Stable ranking: higher score first; ties keep chunk order.
        order = sorted(
            range(self._n_docs),
            key=lambda i: (-scores[i], i),
        )[:top_k]
        hits: list[Hit] = []
        for rank, i in enumerate(order, start=1):
            chunk = self.chunks[i]
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(scores[i]),
                    text=chunk.text,
                    rank=rank,
                    metadata=dict(chunk.metadata),
                )
            )
        return hits


def rrf_fuse(
    rankings: Sequence[Sequence[Hit]],
    *,
    k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[Hit]:
    """Reciprocal rank fusion across ranked hit lists.

    ``score(d) = Σ 1 / (rrf_k + rank_i(d))`` over input rankings.
    """
    if k <= 0:
        raise ValidationError(f"k must be positive; got {k}")
    if rrf_k <= 0:
        raise ValidationError(f"rrf_k must be positive; got {rrf_k}")
    fused: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for hit in ranking:
            entry = fused.get(hit.chunk_id)
            contrib = 1.0 / (rrf_k + hit.rank)
            if entry is None:
                fused[hit.chunk_id] = {
                    "hit": hit,
                    "score": contrib,
                }
            else:
                entry["score"] += contrib
    ordered = sorted(
        fused.values(),
        key=lambda item: (-float(item["score"]), item["hit"].chunk_id),
    )[:k]
    out: list[Hit] = []
    for rank, item in enumerate(ordered, start=1):
        base: Hit = item["hit"]
        out.append(
            Hit(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                score=float(item["score"]),
                text=base.text,
                rank=rank,
                metadata=dict(base.metadata),
            )
        )
    return out


def weighted_fuse(
    dense_hits: Sequence[Hit],
    sparse_hits: Sequence[Hit],
    *,
    k: int,
    dense_weight: float = 0.5,
) -> list[Hit]:
    """Min-max normalize each list, then blend scores with ``dense_weight``.

    ``final = dense_weight * dense_norm + (1 - dense_weight) * sparse_norm``.
    Missing list membership scores as 0 after normalization.
    """
    if k <= 0:
        raise ValidationError(f"k must be positive; got {k}")
    if not 0.0 <= dense_weight <= 1.0:
        raise ValidationError(
            f"dense_weight must be in [0, 1]; got {dense_weight}"
        )
    sparse_weight = 1.0 - dense_weight

    def _norm_map(hits: Sequence[Hit]) -> dict[str, float]:
        if not hits:
            return {}
        scores = [h.score for h in hits]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        out: dict[str, float] = {}
        for h in hits:
            out[h.chunk_id] = 0.0 if span <= 0 else (h.score - lo) / span
        return out

    dense_map = _norm_map(dense_hits)
    sparse_map = _norm_map(sparse_hits)
    by_id: dict[str, Hit] = {}
    for h in dense_hits:
        by_id[h.chunk_id] = h
    for h in sparse_hits:
        by_id.setdefault(h.chunk_id, h)

    scored: list[tuple[str, float]] = []
    for chunk_id in by_id:
        score = dense_weight * dense_map.get(chunk_id, 0.0) + sparse_weight * sparse_map.get(
            chunk_id, 0.0
        )
        scored.append((chunk_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    out: list[Hit] = []
    for rank, (chunk_id, score) in enumerate(scored[:k], start=1):
        base = by_id[chunk_id]
        out.append(
            Hit(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                score=float(score),
                text=base.text,
                rank=rank,
                metadata=dict(base.metadata),
            )
        )
    return out


def match_metadata_filters(
    metadata: dict[str, Any],
    filters: dict[str, Any] | None,
) -> bool:
    """Return True when ``metadata`` satisfies equality filters (AND)."""
    if not filters:
        return True
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True


def filter_chunks(
    chunks: Sequence[Chunk],
    filters: dict[str, Any] | None,
) -> list[Chunk]:
    """Keep chunks whose metadata matches ``filters``."""
    if not filters:
        return list(chunks)
    return [c for c in chunks if match_metadata_filters(c.metadata, filters)]
