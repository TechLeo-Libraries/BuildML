"""Keyword search, and the arithmetic for combining it with vector search.

BM25 is the keyword ranking that has been the strong baseline for decades, and
it is genuinely hard to beat on the queries it suits. It scores a passage by how
often the query's words appear in it, discounted three ways: rare words count
for more than common ones, repeated occurrences saturate rather than accumulate
without limit, and long passages are penalised for having more chances to
contain any given word.

What BM25 cannot do is match a word it has never seen. "Cancel" does not find
"terminate", and no parameter setting changes that.

Dense retrieval has the opposite profile: it handles paraphrase and misses
exact identifiers. Hybrid runs both, which leaves the problem of combining two
score scales that have nothing in common: BM25 is unbounded and corpus-relative,
cosine similarity sits in ``[-1, 1]``. Two answers here. **Reciprocal rank
fusion** ignores the scores and combines *positions*, which needs no calibration
and is the default. **Weighted fusion** min-max normalises each list and blends,
which gives explicit control at the cost of being sensitive to the score
distribution within each query.

See Also
--------
buildml.rag.retrieve : Where these are used.
buildml.rag.types.RetrieveConfig : The parameters.
"""

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
    """Split text into lowercase alphanumeric tokens.

    Deliberately simple, and shared by indexing and querying so that both sides
    tokenise identically: a mismatch there means query terms that cannot match
    anything.

    Parameters
    ----------
    text:
        The text to tokenise.

    Returns
    -------
    list of str
        Lowercase runs of letters and digits, in order.

    Notes
    -----
    **Punctuation is dropped, which splits hyphenated and dotted terms.**
    ``'covid-19'`` becomes two tokens and ``'v1.2'`` becomes two more, so an
    exact-identifier query may match more loosely than expected.

    **No stemming and no stopword removal.** ``'running'`` and ``'run'`` are
    unrelated terms here; BM25's inverse document frequency handles common words
    by weighting rather than by removing them.

    Examples
    --------
    >>> from buildml.rag.hybrid import tokenize
    >>> tokenize("Error-42: restart required.")
    ['error', '42', 'restart', 'required']
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """Okapi BM25 keyword ranking over chunk texts.

    The strong keyword baseline, implemented in pure Python with no optional
    dependencies. It excels at exactly what dense retrieval is worst at: error
    codes, part numbers, proper nouns, and any term whose value lies in matching
    literally.

    Attributes
    ----------
    chunks:
        The passages being ranked.
    k1:
        Term-frequency saturation. Controls how quickly repeated occurrences
        stop adding relevance: a word appearing ten times is not ten times as
        good a signal as appearing once.
    b:
        Length normalisation, 0 to 1. At 1, long passages are fully penalised
        for the extra chances they have to contain any word; at 0, not at all.

    Notes
    -----
    **Rebuilt per query when filters are used.** Unlike the vector store, which
    persists, this is constructed from the filtered chunk set each time: fine
    for moderate corpora, and a cost that grows with corpus size.

    **Exact matching only.** No stemming, no synonyms. This is the half of
    hybrid retrieval that cannot generalise, paired with the half that cannot
    match literally.

    See Also
    --------
    rrf_fuse : Combining these results with dense ones.
    tokenize : How text is split.
    """

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
        """Tokenise the chunks and precompute the corpus statistics.

        Document frequencies and the average length are computed once here, so
        that scoring a query afterwards is a pass over the term counts rather
        than over the corpus statistics.

        Parameters
        ----------
        chunks:
            The passages to index.
        k1:
            Term-frequency saturation.
        b:
            Length normalisation.

        Returns
        -------
        BM25Index
            The built index.

        Notes
        -----
        **Statistics are relative to this chunk set.** A term's rarity is
        measured against the passages given here, so building over a filtered
        subset produces different: and correct: weights for that subset.

        **Every token of every chunk is held in memory.**
        """
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
        """Score every chunk against the query, in chunk order.

        The raw scores, unsorted, for callers that want to combine them with
        something else rather than take a top-``k``.

        Parameters
        ----------
        query:
            The question, tokenised the same way the chunks were.

        Returns
        -------
        list of float
            One score per chunk, positionally aligned. Higher is more relevant.

        Notes
        -----
        **Scores are unbounded and corpus-relative.** A score of 12 means
        nothing on its own and cannot be compared against a score from a
        different corpus, a different query, or a cosine similarity. Only the
        ordering within one call is meaningful.

        **Zero means no query term appeared.** A query whose words are absent
        from the corpus scores everything at zero, and the ranking is then
        arbitrary.
        """
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
        """Return the ``k`` best-matching chunks.

        Scores everything, then sorts. Ties keep corpus order, so the same query
        always returns the same ranking.

        Parameters
        ----------
        query:
            The question.
        k:
            How many hits to return.

        Returns
        -------
        list of Hit
            Best first, ranked from 1. Shorter than ``k`` when the index has
            fewer chunks, and empty when it has none.

        Raises
        ------
        ValidationError
            If ``k`` is not positive.

        Notes
        -----
        **A query with no matching terms still returns ``k`` hits**, all scoring
        zero and ordered by position. There is no relevance threshold.

        **Full sort rather than partial selection**, which is fine at the corpus
        sizes this is built for and grows faster than the vector store's search.
        """
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
    """Combine rankings by position, ignoring the scores entirely.

    **The reason this works is that it never compares incomparable numbers.** A
    BM25 score of 12 and a cosine similarity of 0.7 cannot be added, averaged,
    or weighted against each other in any principled way. Their *positions* can:
    each chunk contributes ``1 / (rrf_k + rank)`` from every list it appears in,
    and the sums are compared.

    A chunk ranked highly by both methods beats one ranked first by a single
    method, which is exactly the behaviour hybrid retrieval is for: agreement
    between two different notions of relevance is stronger evidence than
    excellence under one.

    Parameters
    ----------
    rankings:
        Two or more ranked lists. Chunks may appear in any subset of them.
    k:
        How many fused hits to return.
    rrf_k:
        Damping constant. Larger values flatten the advantage of top positions,
        making the fusion less dominated by either list's first place. The
        conventional 60 is a robust default rather than a tuned one.

    Returns
    -------
    list of Hit
        Fused ranking, best first, renumbered from 1. Scores are RRF sums.

    Raises
    ------
    ValidationError
        If ``k`` or ``rrf_k`` is not positive.

    Notes
    -----
    **The returned scores are not similarities.** They are small numbers whose
    only meaning is relative order within this fusion, bounded above by the
    number of lists divided by ``rrf_k + 1``.

    **Absence costs nothing directly.** A chunk missing from one list simply
    contributes from the other, so a strong single-method hit still competes.

    **Ties break by chunk ID**, which makes the output deterministic.

    See Also
    --------
    weighted_fuse : The score-based alternative.
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
    """Blend the two rankings by normalised score, with an explicit weight.

    The alternative to :func:`rrf_fuse`, for when you want to say *how much*
    each method should count. Each list is min-max normalised to ``[0, 1]``
    within itself, then the two are combined by ``dense_weight``.

    The normalisation is what makes this possible and also what makes it
    fragile: it is computed per query, so the same absolute score maps to
    different normalised values depending on what else came back. A query where
    every dense result is equally mediocre still has a normalised 1.0 at the
    top.

    Parameters
    ----------
    dense_hits:
        Ranked results from vector search.
    sparse_hits:
        Ranked results from BM25.
    k:
        How many fused hits to return.
    dense_weight:
        Dense share, from 0 (pure BM25) to 1 (pure dense).

    Returns
    -------
    list of Hit
        Fused ranking, best first, renumbered from 1. Scores are the blend, in
        ``[0, 1]``.

    Raises
    ------
    ValidationError
        If ``k`` is not positive, or ``dense_weight`` is outside ``[0, 1]``.

    Notes
    -----
    **A chunk missing from one list scores zero for that half**, which is a real
    penalty: unlike RRF, where absence merely means no contribution. This makes
    weighted fusion favour chunks both methods found.

    **Prefer RRF unless you have a measured reason not to.** Per-query
    normalisation makes these scores unstable across queries, and the weight is
    another parameter to tune with a labelled set you may not have.

    **All scores collapse to zero when a list has no spread.** With identical
    scores throughout, min-max normalisation has nothing to normalise.

    See Also
    --------
    rrf_fuse : The calibration-free alternative.
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
    """Test one chunk's metadata against a set of equality constraints.

    All constraints must hold. Used to restrict retrieval to a document version,
    a language, a date range expressed as a label: anything recorded on the
    chunk at ingest.

    Parameters
    ----------
    metadata:
        The chunk's metadata.
    filters:
        Key to required value. ``None`` or empty matches everything.

    Returns
    -------
    bool
        True when every constraint is satisfied.

    Notes
    -----
    **Exact equality only.** No ranges, no membership, no substring matching. A
    filter of ``{"year": "2024"}`` does not match an integer ``2024``.

    **A missing key fails the filter**, rather than being treated as
    unconstrained. Chunks ingested before a metadata field existed will be
    excluded by any filter on it.

    See Also
    --------
    filter_chunks : Applying this across a collection.
    """
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
    """Narrow a chunk collection to those matching the constraints.

    Applied before scoring, so filtered-out chunks are never candidates rather
    than being ranked and discarded.

    Parameters
    ----------
    chunks:
        The passages to filter.
    filters:
        Key to required value. ``None`` or empty returns everything.

    Returns
    -------
    list of Chunk
        The matching chunks, in original order.

    Notes
    -----
    **An over-narrow filter returns nothing, and nothing is a valid result.**
    Retrieval then returns no hits rather than falling back to the unfiltered
    set, which is correct and can look like a broken index.

    See Also
    --------
    match_metadata_filters : The per-chunk test.
    """
    if not filters:
        return list(chunks)
    return [c for c in chunks if match_metadata_filters(c.metadata, filters)]
