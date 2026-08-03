"""What each stage of the RAG pipeline hands to the next, and what it admits to.

Documents become chunks, chunks become an index, an index answers queries with
hits, hits become citations, and citations accompany an answer. Each of those
transitions has a type here, and the chain of identifiers running through them
means any sentence in a generated answer can be traced back to the characters in
the file it came from.

Every result also carries ``disclosures`` and, where relevant, ``warnings``.
That is deliberate: RAG has more ways to be quietly wrong than most pipelines —
an index built with a placeholder embedder, a hybrid query that silently fell
back to dense, passages retrieved and then dropped for space — and none of them
raise. They are recorded instead, so the reason a system underperforms is
readable rather than deduced.

See Also
--------
buildml.rag.types : The configuration these results come from.
buildml.rag.explain_hooks : How these are summarised for history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A single document before it is cut into retrievable pieces.

    The unit of ingest. Documents are never retrieved directly — chunks are —
    but the document ID travels with every chunk, so a retrieved passage always
    identifies the document it came from.

    Attributes
    ----------
    doc_id:
        Stable identifier. Appears in every citation, so it should mean
        something to whoever reads the answer.
    text:
        The full content.
    metadata:
        Anything else worth keeping: source, date, version, author. Available
        as retrieval filters.
    role:
        ``'index'`` to be searchable, ``'eval_only'`` to be held out.

    Notes
    -----
    **``role`` is the leakage control.** An ``'eval_only'`` document is excluded
    from indexing, so evaluation questions are answered by retrieval rather than
    by finding the answer key.

    **Metadata is only useful if it is queried.** Recording a version field does
    nothing unless a retrieval filter uses it.

    See Also
    --------
    Chunk : What a document becomes.
    buildml.rag.corpus : How documents are created.
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = "index"  # "index" | "eval_only"

    def to_dict(self) -> dict[str, Any]:
        """Return the document as a JSON-safe mapping.

        Includes the full text, so this is a serialisation of the document
        rather than a summary of it.

        Returns
        -------
        dict
            Identifier, text, metadata copy, and role.

        Notes
        -----
        **The whole text is included.** For logging a corpus, prefer
        :meth:`CorpusHandle.to_dict`, which reports identifiers only.
        """
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Document:
        """Rebuild a document from a stored mapping.

        The inverse of :meth:`to_dict`, used when loading a saved corpus.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`. Must have ``doc_id`` and
            ``text``.

        Returns
        -------
        Document
            The reconstructed document.

        Raises
        ------
        KeyError
            If ``doc_id`` or ``text`` is missing.

        Notes
        -----
        **A missing role defaults to ``'index'``**, meaning a payload that lost
        its role becomes indexable. Check roles explicitly when loading a corpus
        that is meant to be held out.
        """
        return cls(
            doc_id=str(payload["doc_id"]),
            text=str(payload["text"]),
            metadata=dict(payload.get("metadata") or {}),
            role=str(payload.get("role") or "index"),
        )


@dataclass(slots=True)
class Chunk:
    """A passage: the unit that is actually embedded, retrieved, and cited.

    Chunks are what the system works with. A document is only ever a source of
    chunks, and the character offsets recorded here are what make a citation
    verifiable — they point at an exact span of the original file, not at the
    document in general.

    Attributes
    ----------
    chunk_id:
        Stable identifier, typically derived from the document ID and position.
    doc_id:
        Which document this came from.
    text:
        The passage. This is what gets embedded and what reaches the model.
    start_char:
        Where the passage begins in the source document.
    end_char:
        Where it ends, exclusive.
    metadata:
        Inherited from the document, plus anything chunking added.

    Notes
    -----
    **Offsets are character positions, not token positions**, and they refer to
    the document text as ingested. They will not line up if the document is
    later re-encoded or normalised differently.

    **Adjacent chunks overlap.** The same sentence can legitimately appear in
    two chunks, which is why two hits can look like duplicates.

    See Also
    --------
    Document : Where a chunk comes from.
    Hit : A chunk that was retrieved.
    """

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the chunk as a JSON-safe mapping.

        Includes the text and the offsets, so a stored chunk can be checked
        against its source document.

        Returns
        -------
        dict
            Identifiers, text, character offsets, and metadata copy.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        """Rebuild a chunk from a stored mapping.

        The inverse of :meth:`to_dict`, used when loading a saved index bundle.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`. Identifiers, text, and
            both offsets are required.

        Returns
        -------
        Chunk
            The reconstructed chunk.

        Raises
        ------
        KeyError
            If a required key is missing.

        Notes
        -----
        **Offsets are not checked against any document.** A chunk restored
        against changed source text will cite the wrong span.
        """
        return cls(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            text=str(payload["text"]),
            start_char=int(payload["start_char"]),
            end_char=int(payload["end_char"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class CorpusHandle:
    """A loaded corpus, held in memory.

    What the ingest functions produce and what indexing consumes. Documents keep
    the order they were loaded in, and may mix index and eval-only roles — the
    indexing path filters rather than trusting the caller to have separated
    them.

    Attributes
    ----------
    documents:
        The documents, in load order.
    source:
        Where they came from, for provenance.

    Notes
    -----
    **Entirely in memory.** The whole corpus text is resident, so a very large
    corpus needs chunking and indexing in batches rather than one handle.

    **A mixed-role handle is normal.** Nothing about holding both kinds is
    unsafe; the guards in :mod:`buildml.rag.corpus` are what keep the eval-only
    ones out of the index.

    See Also
    --------
    buildml.rag.corpus.indexable_documents : The role filter.
    """

    documents: tuple[Document, ...]
    source: str = "memory"

    @property
    def n_documents(self) -> int:
        """Return how many documents the corpus holds.

        Counts every document regardless of role, so this is not the number
        that will be indexed.

        Returns
        -------
        int
            The document count.

        See Also
        --------
        buildml.rag.corpus.indexable_documents : The indexable subset.
        """
        return len(self.documents)

    def to_dict(self) -> dict[str, Any]:
        """Describe the corpus without including any document text.

        Identifiers and roles only, so a corpus can be recorded in a run
        history without putting its contents there.

        Returns
        -------
        dict
            Document count, source, the list of identifiers, and the distinct
            roles present.

        Notes
        -----
        **Document IDs are included, and IDs can be revealing** when they are
        filenames or customer references.
        """
        return {
            "n_documents": self.n_documents,
            "source": self.source,
            "doc_ids": [d.doc_id for d in self.documents],
            "roles": sorted({d.role for d in self.documents}),
        }


@dataclass(slots=True)
class ChunkResult:
    """The chunks a corpus produced, with the settings that produced them.

    Worth inspecting before building an index. The ratio of chunks to documents
    is the quickest check that chunking did something sensible: a few hundred
    chunks from a few hundred documents means the chunk size is larger than the
    documents, and retrieval is effectively document-level.

    Attributes
    ----------
    chunks:
        Every chunk, in document then position order.
    config:
        The chunking settings, recorded so the result is reproducible.
    n_documents:
        How many documents were chunked.

    Notes
    -----
    **Chunks per document is the diagnostic.** Close to one means chunking is
    not dividing anything; very high means passages may be too small to be
    interpretable on their own.

    See Also
    --------
    buildml.rag.types.ChunkConfig : The settings.
    buildml.rag.chunk : What produces this.
    """

    chunks: tuple[Chunk, ...]
    config: dict[str, Any]
    n_documents: int

    @property
    def n_chunks(self) -> int:
        """Return how many chunks were produced.

        The size of the index that will be built from this.

        Returns
        -------
        int
            The chunk count.
        """
        return len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """Summarise the chunking without including any chunk text.

        Counts and settings only, so a run history can record how a corpus was
        divided without storing the corpus.

        Returns
        -------
        dict
            Chunk count, document count, and the configuration used.

        Notes
        -----
        **The chunks themselves are excluded.** This is a record of what
        happened, not the payload.
        """
        return {
            "n_chunks": self.n_chunks,
            "n_documents": self.n_documents,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class Hit:
    """A chunk that retrieval returned, with its position and score.

    Hits come back ordered, and **the order is the trustworthy part**. The score
    means different things in different modes — a BM25 score is unbounded, a
    cosine similarity sits in ``[-1, 1]``, and a reciprocal-rank-fusion score is
    a small number with no interpretation outside the fusion — so comparing
    scores across modes, or across queries, does not mean anything.

    Attributes
    ----------
    chunk_id:
        Which chunk.
    doc_id:
        Which document it came from.
    score:
        Relevance under whichever mode ran. Comparable within one result set,
        not across them.
    text:
        The passage.
    rank:
        Position in the ranking, starting at 1.
    metadata:
        Carried from the chunk.

    Notes
    -----
    **Rank is the comparable quantity; score is not.** A hit at rank 1 with a
    score of 0.31 is not worse than one at rank 1 with 0.95 in another mode.

    **A high score is not a guarantee of relevance.** Retrieval returns the
    closest chunks it has, so a query with no good answer in the corpus still
    produces ``k`` confident-looking hits.

    See Also
    --------
    RetrieveResult : The ranking these belong to.
    Citation : A hit that reached the answer.
    """

    chunk_id: str
    doc_id: str
    score: float
    text: str
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the hit as a JSON-safe mapping.

        Includes the passage text, so a stored hit can be reviewed without the
        index.

        Returns
        -------
        dict
            Identifiers, score, text, rank, and metadata copy.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "text": self.text,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class IndexResult:
    """A built index, described by what it contains and how it was made.

    Every field here constrains what queries against this index will do.
    ``embedder_id`` and ``dim`` in particular must match at query time —
    embeddings from different models are arithmetically comparable and
    semantically unrelated, so a mismatch returns confident nonsense rather
    than an error.

    Attributes
    ----------
    n_chunks:
        How many passages are searchable.
    n_documents:
        How many documents they came from.
    embedder_id:
        Which embedder produced the vectors. **Checked at query time.**
    dim:
        Vector width.
    store_backend:
        Which store holds them.
    chunk_config:
        The chunking used, recorded for reproducibility.
    embed_config:
        The embedding settings used.
    warnings:
        Problems found while building — empty documents, degenerate chunks.
    disclosures:
        Facts about the index that affect how results should be read, most
        importantly whether a placeholder embedder was used.

    Notes
    -----
    **Read the disclosures before judging retrieval quality.** An index built
    with the hashing embedder cannot match paraphrases at all, and that is
    recorded here rather than raised.

    **The index is immutable.** Adding documents means rebuilding.

    See Also
    --------
    buildml.rag.index : What produces this.
    """

    n_chunks: int
    n_documents: int
    embedder_id: str
    dim: int
    store_backend: str
    chunk_config: dict[str, Any]
    embed_config: dict[str, Any]
    warnings: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the index description as a JSON-safe mapping.

        Everything needed to know what this index is and how it was built. No
        vectors and no text.

        Returns
        -------
        dict
            Counts, embedder identity, dimension, backend, both configs,
            warnings, and disclosures.
        """
        return {
            "n_chunks": self.n_chunks,
            "n_documents": self.n_documents,
            "embedder_id": self.embedder_id,
            "dim": self.dim,
            "store_backend": self.store_backend,
            "chunk_config": dict(self.chunk_config),
            "embed_config": dict(self.embed_config),
            "warnings": list(self.warnings),
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class RetrieveResult:
    """The passages one query found, and how they were found.

    ``mode`` is the field to check first, because it may not be what was asked
    for. A hybrid request falls back to dense when BM25 dependencies are
    missing, and that substitution is recorded here rather than raised — so a
    system that quietly stopped doing keyword matching is visible in the result
    instead of only in the quality.

    Attributes
    ----------
    query:
        The question as asked.
    k:
        How many hits were requested. ``hits`` can be shorter.
    hits:
        The passages, best first.
    embedder_id:
        Which embedder ran, matched against the index.
    mode:
        What actually ran, which may differ from what was requested.
    fusion:
        How rankings were combined, for hybrid.
    filters:
        Metadata constraints applied before scoring.
    rerank:
        Whether a cross-encoder reordered the candidates.
    disclosures:
        Anything affecting how these hits should be read, including fallbacks.
    config:
        The full retrieval settings.

    Notes
    -----
    **Fewer hits than ``k`` is normal**, and means the filters or the corpus
    left less to rank. It is not an error.

    **Every query returns its closest matches, relevant or not.** There is no
    threshold below which retrieval declines to answer, so a question the corpus
    cannot answer still produces a full ranking.

    See Also
    --------
    Hit : One entry.
    buildml.rag.retrieve : What produces this.
    """

    query: str
    k: int
    hits: tuple[Hit, ...]
    embedder_id: str
    mode: str = "dense"
    fusion: str | None = None
    filters: dict[str, Any] | None = None
    rerank: bool = False
    disclosures: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the retrieval as a JSON-safe mapping.

        Includes every hit with its text, so a stored result is enough to
        review what the system found without re-running the query.

        Returns
        -------
        dict
            The query, requested ``k``, hit count, the hits, and all the
            provenance fields.

        Notes
        -----
        **The query and the passages are both included**, which makes this
        unsuitable for a log if either is sensitive.
        """
        return {
            "query": self.query,
            "k": self.k,
            "n_hits": len(self.hits),
            "hits": [h.to_dict() for h in self.hits],
            "embedder_id": self.embedder_id,
            "mode": self.mode,
            "fusion": self.fusion,
            "filters": None if self.filters is None else dict(self.filters),
            "rerank": self.rerank,
            "disclosures": list(self.disclosures),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class RagEvalResult:
    """How often retrieval found the right passage, over labelled questions.

    Four views of the same question. **Recall@k** is the one that bounds
    everything downstream: it is the share of relevant passages that made the
    top ``k``, and a passage that did not arrive cannot be answered from.
    **Hit rate** asks the easier question of whether *anything* relevant
    arrived. **MRR** rewards putting it first. **nDCG** accounts for graded
    relevance and position together.

    Attributes
    ----------
    n_queries:
        How many labelled questions were evaluated.
    k:
        The cutoff used.
    recall_at_k:
        Share of relevant passages retrieved. **The ceiling on answer quality.**
    mrr:
        Mean reciprocal rank of the first relevant hit.
    ndcg_at_k:
        Position-discounted gain.
    hit_rate_at_k:
        Share of queries with at least one relevant hit.
    per_query:
        Per-question detail, for finding which questions fail.
    relevance_mode:
        Whether a hit was counted at document or chunk level.
    retrieve_mode:
        Which retrieval mode was measured.
    disclosures:
        Anything affecting interpretation.
    warnings:
        Problems during evaluation, such as questions whose labelled documents
        are not in the index.

    Notes
    -----
    **Recall@5 of 0.6 caps answer accuracy at roughly 0.6**, whatever the model
    does. When answers are poor, this is the first number to look at.

    **Metrics are only comparable at the same ``k`` and relevance mode.** Both
    are recorded here for exactly that reason.

    **``per_query`` is where the diagnosis is.** A mean of 0.7 can be uniform
    mediocrity or most queries perfect and a category failing completely, and
    those need different fixes.

    See Also
    --------
    buildml.rag.types.EvalConfig : The settings.
    buildml.rag.evaluate : What produces this.
    """

    n_queries: int
    k: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float = 0.0
    hit_rate_at_k: float = 0.0
    per_query: tuple[dict[str, Any], ...] = ()
    relevance_mode: str = "document"
    retrieve_mode: str = "dense"
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a JSON-safe mapping.

        Includes the cutoff and relevance mode alongside the scores, so a
        recorded number can never be compared against one measured differently.

        Returns
        -------
        dict
            All four metrics, the settings they were measured under, per-query
            detail, disclosures, and warnings.
        """
        return {
            "n_queries": self.n_queries,
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "relevance_mode": self.relevance_mode,
            "retrieve_mode": self.retrieve_mode,
            "per_query": list(self.per_query),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RagGenerateEvalResult:
    """Rough, cheap signals about generated answers — not quality judgements.

    **These are heuristics, and the distinction matters.** Faithfulness here is
    token overlap between the answer and the context, which an answer can score
    highly on by copying text it has misunderstood, and score poorly on by
    correctly paraphrasing. They are useful as a regression signal — a sudden
    drop means something changed — and not as a measure of whether answers are
    good.

    Attributes
    ----------
    n_queries:
        How many answers were scored.
    mean_faithfulness:
        Average overlap between answers and their contexts.
    mean_answer_relevance:
        Average overlap between answers and their questions.
    citation_coverage:
        Share of answers that cite their sources. **The most reliable of the
        three**, because it measures something structural rather than semantic.
    per_query:
        Per-question detail.
    disclosures:
        Notes on interpretation.
    warnings:
        Problems encountered while scoring.

    Notes
    -----
    **High overlap is not correctness.** An answer that copies a passage
    verbatim scores well whether or not the passage answers the question.

    **Low overlap is not incorrectness.** A well-written paraphrase shares few
    exact tokens with its source.

    **Judging answer quality requires human review or a model-based grader.**
    Neither is what this provides.

    See Also
    --------
    FaithfulnessReport : The per-answer version.
    RagEvalResult : Retrieval quality, which is measurable.
    """

    n_queries: int
    mean_faithfulness: float
    mean_answer_relevance: float
    citation_coverage: float
    per_query: tuple[dict[str, Any], ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the heuristic scores as a JSON-safe mapping.

        Carries the disclosures alongside the numbers, since the numbers are
        easy to over-read on their own.

        Returns
        -------
        dict
            The three means, per-query detail, disclosures, and warnings.

        Notes
        -----
        **Record the disclosures with the numbers.** Stored on their own, these
        scores read as quality measurements, which they are not.
        """
        return {
            "n_queries": self.n_queries,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevance": self.mean_answer_relevance,
            "citation_coverage": self.citation_coverage,
            "per_query": list(self.per_query),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ConfigCompareResult:
    """Several retrieval configurations, measured the same way.

    The honest way to choose settings. Chunk size, retrieval mode, and reranking
    all interact — larger chunks change what BM25 matches, reranking recovers
    from a weak first stage — so reasoning about them separately is unreliable
    and measuring them together is not.

    Attributes
    ----------
    rows:
        One row per configuration, with its settings and metrics.
    k:
        The cutoff every configuration was measured at.
    relevance_mode:
        The relevance definition every configuration was measured under.
    disclosures:
        Notes on interpretation.

    Notes
    -----
    **The shared ``k`` and relevance mode are what make the rows comparable.**
    They are stored once, at this level, so a comparison cannot accidentally mix
    measurement settings.

    **A small labelled set will not separate close configurations.** Differences
    of a few points on a few dozen questions are noise; treat this as ranking
    candidates, not as proof.

    See Also
    --------
    RagEvalResult : A single configuration's metrics.
    """

    rows: tuple[dict[str, Any], ...]
    k: int
    relevance_mode: str
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as a JSON-safe mapping.

        The shared measurement settings travel with the rows, so a stored
        comparison stays interpretable.

        Returns
        -------
        dict
            The rows plus the shared measurement settings and disclosures.
        """
        return {
            "rows": list(self.rows),
            "k": self.k,
            "relevance_mode": self.relevance_mode,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class Citation:
    """A passage that reached the model, numbered so the answer can point at it.

    **The thing that makes a generated answer checkable.** Without citations, a
    grounded answer and a fabricated one look identical. With them, any claim
    can be followed back to a chunk, a document, and a character range in the
    original file.

    Attributes
    ----------
    source_id:
        The number the answer refers to, as in ``[1]``. Assigned by position in
        the prompt.
    chunk_id:
        Which chunk.
    doc_id:
        Which document.
    score:
        Its retrieval score.
    text:
        The passage as it appeared in the prompt.
    rank:
        Its position in the retrieval ranking.
    metadata:
        Carried through from the chunk.

    Notes
    -----
    **A citation records what was available, not what was used.** Every
    retrieved passage placed in the prompt becomes a citation whether or not the
    model drew on it. Whether the answer actually cites them is a separate
    question — see :class:`FaithfulnessReport`.

    **``source_id`` is prompt position, not chunk identity.** The same chunk
    gets a different number in a different query.

    See Also
    --------
    Hit : What a citation is made from.
    FaithfulnessReport : Whether the answer cited these.
    """

    source_id: int
    chunk_id: str
    doc_id: str
    score: float
    text: str
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the citation as a JSON-safe mapping.

        Includes the passage text, so a stored answer can be verified against
        its sources without the index.

        Returns
        -------
        dict
            Source number, identifiers, score, text, rank, and metadata copy.
        """
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "text": self.text,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class FaithfulnessReport:
    """Whether one answer looks like it came from its sources.

    Two cheap checks. **Citation coverage** counts how many of the supplied
    sources the answer actually refers to — structural, and reliable as far as
    it goes. **Token overlap** measures shared vocabulary between the answer and
    the context, which is a much weaker signal.

    Read this as a screen, not a verdict. It reliably catches an answer that
    cites nothing and shares no vocabulary with its sources. It cannot tell a
    correct answer from a plausible one.

    Attributes
    ----------
    citation_marker_coverage:
        Share of supplied sources the answer cites, in ``[0, 1]``.
    cited_source_ids:
        Which sources were referenced.
    missing_source_ids:
        Which were supplied and ignored. Often fine — not every retrieved
        passage is relevant.
    answer_context_token_overlap:
        Vocabulary shared with the context, in ``[0, 1]``.
    grounded:
        Whether both signals cleared their thresholds.
    disclosures:
        Notes on interpretation.
    limitations:
        What this cannot tell you.

    Notes
    -----
    **``grounded`` is a threshold on heuristics, not a verification.** An answer
    can be marked grounded and still be wrong, and a correct paraphrase can fail
    the overlap check.

    **Missing sources are usually expected.** Retrieval returns ``k`` passages;
    a focused answer uses one or two.

    See Also
    --------
    Citation : The sources this scores against.
    RagGenerateEvalResult : The same signals across many answers.
    """

    citation_marker_coverage: float
    cited_source_ids: tuple[int, ...]
    missing_source_ids: tuple[int, ...]
    answer_context_token_overlap: float
    grounded: bool
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """Combine the two signals into one number, weighted equally.

        Convenient for tracking a trend across runs, at the cost of hiding
        which half moved.

        Returns
        -------
        float
            The mean of citation coverage and token overlap, in ``[0, 1]``.

        Notes
        -----
        **The equal weighting is a convention, not a finding.** The two signals
        measure different things and are not equally reliable; citation
        coverage is the sturdier half.

        **A middling score is ambiguous.** Half could be perfect citations with
        no vocabulary overlap, or the reverse, and those mean different things.
        Read the components when it matters.
        """
        return float(
            0.5 * self.citation_marker_coverage + 0.5 * self.answer_context_token_overlap
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the grounding signals as a JSON-safe mapping.

        Includes the combined score and both components, plus the limitations —
        which belong with the numbers rather than beside them.

        Returns
        -------
        dict
            Combined score, both signals, cited and missing source lists, the
            grounded flag, disclosures, and limitations.
        """
        return {
            "score": self.score,
            "citation_marker_coverage": self.citation_marker_coverage,
            "cited_source_ids": list(self.cited_source_ids),
            "missing_source_ids": list(self.missing_source_ids),
            "answer_context_token_overlap": self.answer_context_token_overlap,
            "grounded": self.grounded,
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
        }


@dataclass(slots=True)
class GenerateResult:
    """An answer, its sources, and the whole chain that produced it.

    The end of the pipeline, and it carries everything needed to audit itself:
    the answer, the passages it was given, the retrieval that found them, the
    exact prompt context, and the grounding heuristics. A claim in the answer
    can be followed to a citation, to a chunk, to a character range in a
    document.

    Attributes
    ----------
    query:
        The question as asked.
    answer:
        What the model produced.
    citations:
        The passages supplied, numbered as they appear in the prompt.
    retrieve_result:
        The retrieval that found them, including mode and any fallback.
    provider_model:
        Which model answered.
    usage:
        Token counts, where the provider reports them.
    prompt_context:
        The assembled context, exactly as sent.
    disclosures:
        Anything affecting how the answer should be read — most importantly,
        whether passages were dropped for space.
    config:
        The generation settings.
    faithfulness:
        Grounding heuristics, when computed.

    Notes
    -----
    **Citations show what was available, not what was used.** Verifying that the
    answer follows from them is the reader's job; ``faithfulness`` is a screen,
    not a substitute.

    **Fewer citations than ``k`` means passages were dropped**, cut by the
    context budget after being retrieved. The disclosures say so.

    **``prompt_context`` is the ground truth for debugging.** When an answer is
    wrong, it settles whether the passage was missing from the prompt or
    present and ignored — which are different problems with different fixes.

    See Also
    --------
    Citation : One source.
    RetrieveResult : The retrieval behind it.
    buildml.rag.generate.generate_grounded : What produces this.
    """

    query: str
    answer: str
    citations: tuple[Citation, ...]
    retrieve_result: RetrieveResult | None = None
    provider_model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    prompt_context: str = ""
    disclosures: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    faithfulness: FaithfulnessReport | None = None

    @property
    def n_citations(self) -> int:
        """Return how many sources reached the model.

        Compare against the requested ``k``: a smaller number means passages
        were retrieved and then dropped by the context budget.

        Returns
        -------
        int
            The citation count.

        See Also
        --------
        buildml.rag.types.GenerateConfig : Where the budget is set.
        """
        return len(self.citations)

    def to_dict(self) -> dict[str, Any]:
        """Return the answer and its provenance as a JSON-safe mapping.

        The full audit trail, minus the assembled prompt itself — that is
        reported as a character count, since including it would duplicate every
        citation's text.

        Returns
        -------
        dict
            Query, answer, citation count and citations, nested retrieval,
            model, usage, prompt size, disclosures, config, and faithfulness.

        Notes
        -----
        **The query, answer, and all passage text are included.** Unsuitable for
        a log when any of those are sensitive.
        """
        return {
            "query": self.query,
            "answer": self.answer,
            "n_citations": self.n_citations,
            "citations": [c.to_dict() for c in self.citations],
            "retrieve": None
            if self.retrieve_result is None
            else self.retrieve_result.to_dict(),
            "provider_model": self.provider_model,
            "usage": dict(self.usage),
            "prompt_context_chars": len(self.prompt_context),
            "disclosures": list(self.disclosures),
            "config": dict(self.config),
            "faithfulness": None if self.faithfulness is None else self.faithfulness.to_dict(),
        }
