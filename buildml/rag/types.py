"""The knobs that decide what a RAG system can and cannot find.

Retrieval-augmented generation answers a question by first finding relevant
passages and then asking a language model to answer *from those passages*. The
model's fluency is rarely the bottleneck. What limits the answer is whether the
right passage was retrieved at all — and that is settled entirely by the
configuration in this module.

The four decisions that matter most, in the order they bite:

**Chunk size** (:class:`ChunkConfig`) sets what a retrievable unit is. Too large
and a passage matches everything without answering anything; too small and the
sentence that answers the question is separated from the context that makes it
interpretable.

**Embedding model** (:class:`EmbedConfig`) sets what "similar" means. It also
fixes the vector dimension, which cannot change without rebuilding the index.

**Retrieval mode** (:class:`RetrieveConfig`) chooses between keyword matching,
semantic similarity, or both. The two fail on opposite queries, which is why
hybrid is the default where the dependencies allow it.

**Context budget** (:class:`GenerateConfig`) sets how many retrieved passages
actually reach the model. Passages beyond the budget were retrieved and then
discarded.

Every config is JSON round-trippable, so the settings that produced an index
travel with it and a stored index can never be queried under settings it was not
built for.

See Also
--------
buildml.rag.defaults : How defaults adapt to what is installed.
buildml.rag.results : The objects these configs produce.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_CHUNK_STRATEGY: Literal["fixed", "recursive"] = "fixed"
DEFAULT_EMBED_DIM = 384
DEFAULT_TOP_K = 5
HASHING_EMBEDDER_ID = "buildml.hashing_embed.v1"
DEFAULT_STORE_BACKEND = "numpy_cosine"
# Static fallback; runtime prefers hybrid when buildml[rag] is importable.
DEFAULT_RETRIEVE_MODE: Literal["dense", "bm25", "hybrid"] = "dense"
DEFAULT_FUSION: Literal["rrf", "weighted"] = "rrf"
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_RERANK_CANDIDATES = 20

RetrieveMode = Literal["dense", "bm25", "hybrid"]
FusionMethod = Literal["rrf", "weighted"]
RelevanceMode = Literal["document", "chunk"]
ChunkStrategy = Literal["fixed", "recursive"]


@dataclass(slots=True)
class ChunkConfig:
    """How documents are cut into the passages that get retrieved.

    **The most consequential setting in RAG, and the least obvious.** Retrieval
    returns chunks, not documents, so a chunk is simultaneously the unit that
    gets matched and the unit that reaches the model. Those two jobs pull in
    opposite directions.

    A large chunk carries enough surrounding context to be understood on its
    own, but its embedding averages several topics together and ends up
    moderately similar to every query and strongly similar to none. A small
    chunk embeds sharply, and then arrives at the model as a sentence stripped
    of the paragraph that gave it meaning — the pronoun with no antecedent, the
    figure with no units.

    Attributes
    ----------
    size:
        Target chunk length in **characters, not tokens**. Roughly four
        characters per English token, so the default is on the order of 128
        tokens.
    overlap:
        Characters repeated between neighbours. This is the insurance against
        cutting through the middle of the sentence that answers the question.
    strategy:
        ``'fixed'`` cuts every ``size`` characters regardless of content.
        ``'recursive'`` tries the separators in order and cuts at the coarsest
        boundary that fits.
    separators:
        Boundaries the recursive strategy prefers, coarsest first: paragraphs,
        lines, sentences, words, then anywhere.

    Notes
    -----
    **Prefer ``'recursive'`` for prose.** Fixed chunking will cut mid-word and
    mid-sentence, and a chunk that starts halfway through a clause embeds
    poorly and reads worse.

    **Overlap costs storage and duplicates results.** At the default ratio
    roughly an eighth of the corpus is stored twice, and the same sentence can
    surface in two adjacent chunks, spending two of your ``k`` slots on one
    passage.

    **Changing any of this invalidates the index.** Chunk boundaries determine
    what was embedded, so a re-chunk is a rebuild.

    Examples
    --------
    Larger chunks with sentence-aware boundaries::

        config = ChunkConfig(size=1024, overlap=128, strategy="recursive")

    See Also
    --------
    buildml.rag.chunk : Where this is applied.
    """

    size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP
    strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-safe mapping.

        Stored alongside an index so the chunking that produced it is
        recoverable, and comparable across runs.

        Returns
        -------
        dict
            All fields, with ``separators`` as a list.
        """
        payload = asdict(self)
        payload["separators"] = list(self.separators)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChunkConfig:
        """Rebuild a config from stored settings.

        Missing keys fall back to defaults, so a config written by an older
        version still loads.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`.

        Returns
        -------
        ChunkConfig
            The reconstructed config.

        Notes
        -----
        **Values are coerced, not validated.** A nonsensical overlap survives
        here and is caught when chunking runs.
        """
        raw_seps = payload.get("separators")
        separators: tuple[str, ...]
        if raw_seps is None:
            separators = ("\n\n", "\n", ". ", " ", "")
        else:
            separators = tuple(str(s) for s in raw_seps)
        return cls(
            size=int(payload.get("size", DEFAULT_CHUNK_SIZE)),
            overlap=int(payload.get("overlap", DEFAULT_CHUNK_OVERLAP)),
            strategy=payload.get("strategy") or DEFAULT_CHUNK_STRATEGY,
            separators=separators,
        )


@dataclass(slots=True)
class EmbedConfig:
    """Which model turns text into vectors, and therefore what "similar" means.

    Dense retrieval finds passages whose vectors are close to the query's. What
    counts as close is entirely a property of the embedding model, so this
    choice decides whether "how do I cancel?" retrieves the passage titled
    "terminating your subscription".

    Attributes
    ----------
    embedder_id:
        Identifier recorded with the index. A query embedded by a different
        model than the index is a silent failure — the vectors are comparable
        arithmetically and meaningless semantically — so this is checked rather
        than trusted.
    dim:
        Vector width. Fixed at index build; changing it is a rebuild.
    backend:
        ``'hashing'`` is dependency-free and **not semantic** — it matches
        surface tokens, so synonyms do not match. ``'sentence-transformers'``
        is a real embedding model. ``'callable'`` is your own function.
    model_name:
        Model to load, for the sentence-transformers backend.
    device:
        Where to run it, such as ``'cuda'``. Defaults to automatic.

    Notes
    -----
    **The default hashing backend is a working placeholder, not a semantic
    model.** It exists so the RAG path runs with no optional dependencies, and
    it will miss any paraphrase. Install ``buildml[rag]`` and pick a real model
    before judging retrieval quality.

    **The index and the query must share an embedder.** This is the most common
    way a RAG system fails quietly: results come back, ranked, and unrelated.

    Examples
    --------
    Use a real embedding model::

        config = EmbedConfig(
            backend="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            dim=384,
        )

    See Also
    --------
    buildml.rag.embed : Where this is applied.
    """

    embedder_id: str = HASHING_EMBEDDER_ID
    dim: int = DEFAULT_EMBED_DIM
    backend: Literal["hashing", "sentence-transformers", "callable"] = "hashing"
    model_name: str | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-safe mapping.

        Stored with the index so the embedder that built it is recoverable and
        can be checked against the one used at query time.

        Returns
        -------
        dict
            All fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbedConfig:
        """Rebuild a config from stored settings.

        Missing keys fall back to the hashing backend, which always works even
        when optional dependencies are absent.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`.

        Returns
        -------
        EmbedConfig
            The reconstructed config.

        Notes
        -----
        **A missing backend silently becomes hashing.** Loading a config whose
        model failed to record leaves you with non-semantic retrieval; check
        ``backend`` after loading if it matters.
        """
        return cls(
            embedder_id=str(payload.get("embedder_id") or HASHING_EMBEDDER_ID),
            dim=int(payload.get("dim") or DEFAULT_EMBED_DIM),
            backend=payload.get("backend") or "hashing",
            model_name=payload.get("model_name"),
            device=payload.get("device"),
        )


@dataclass(slots=True)
class IndexConfig:
    """How vectors are stored and searched.

    Search over embeddings is either exact or approximate. Exact search compares
    the query against every vector — correct by construction, and linear in
    corpus size. Approximate search trades a small amount of recall for a large
    amount of speed, which starts to matter somewhere in the hundreds of
    thousands of chunks.

    Attributes
    ----------
    store_backend:
        Which store to use. ``'numpy_cosine'`` is exact brute-force cosine
        similarity, with no dependencies.

    Notes
    -----
    **The default is exact, and exact is usually right.** Brute force over tens
    of thousands of chunks is fast enough that approximation buys nothing but
    risk. Reach for an approximate index when measurements say to.

    See Also
    --------
    buildml.rag.store : The store implementations.
    buildml.rag.index : Where indexes are built.
    """

    store_backend: str = DEFAULT_STORE_BACKEND

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-safe mapping.

        Stored with the index so the backend that built it is recoverable.

        Returns
        -------
        dict
            All fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexConfig:
        """Rebuild a config from stored settings.

        Falls back to the exact backend, which is always available.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`.

        Returns
        -------
        IndexConfig
            The reconstructed config.
        """
        return cls(store_backend=str(payload.get("store_backend") or DEFAULT_STORE_BACKEND))


@dataclass(slots=True)
class RetrieveConfig:
    """How candidates are found, combined, and how many survive.

    Two retrieval methods exist and they fail on opposite queries. **BM25**
    matches words, so it nails an error code, a product SKU, or a surname, and
    it misses every paraphrase. **Dense** retrieval matches meaning, so it
    handles "cancel" against "terminate", and it can rank a passage about the
    wrong version highly because it is semantically adjacent.

    Hybrid runs both and fuses the rankings, which is why it is the default
    wherever the dependencies allow. Reciprocal rank fusion combines by
    *position* rather than score, which avoids having to make two incomparable
    scoring scales agree — a BM25 score of 12 and a cosine similarity of 0.7
    have no common unit.

    Attributes
    ----------
    k:
        How many chunks to return. The ceiling on what generation can use.
    mode:
        ``'bm25'``, ``'dense'``, or ``'hybrid'``.
    fusion:
        How hybrid combines rankings. ``'rrf'`` uses positions and needs no
        calibration; ``'weighted'`` blends normalised scores and gives you
        explicit control through ``dense_weight``.
    rrf_k:
        Damping for RRF. Larger values flatten the contribution of top ranks,
        making the fusion less sensitive to either method's first place.
    dense_weight:
        Dense share under weighted fusion, from 0 (pure BM25) to 1 (pure
        dense).
    bm25_k1:
        Term-frequency saturation. Controls how quickly repeated occurrences of
        a word stop adding relevance.
    bm25_b:
        Length normalisation, 0 to 1. At 1, long documents are fully penalised
        for their length; at 0, not at all.
    filters:
        Metadata equality constraints applied before scoring, such as
        restricting to one product version.
    rerank:
        Run a cross-encoder over the candidates. **The largest available
        quality gain, and the largest latency cost.**
    rerank_model:
        Which cross-encoder to use.
    rerank_candidates:
        How many to rerank before cutting to ``k``. Must exceed ``k`` for
        reranking to change anything.

    Notes
    -----
    **``k`` is a budget, not a quality dial.** Raising it adds lower-ranked
    passages, and the ones past a certain point are noise the model has to read
    past. Adding irrelevant context measurably degrades answers.

    **Filters apply before scoring**, so an over-narrow filter can leave nothing
    to rank. A filter that matches no chunks returns no results, not the
    unfiltered ones.

    Examples
    --------
    Hybrid with reranking::

        config = RetrieveConfig(
            k=5, mode="hybrid", rerank=True, rerank_candidates=50,
        )

    See Also
    --------
    buildml.rag.hybrid : The fusion implementation.
    buildml.rag.rerank : The cross-encoder pass.
    """

    k: int = DEFAULT_TOP_K
    mode: RetrieveMode = DEFAULT_RETRIEVE_MODE
    fusion: FusionMethod = DEFAULT_FUSION
    rrf_k: int = DEFAULT_RRF_K
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    filters: dict[str, Any] | None = None
    rerank: bool | str = False
    rerank_model: str | None = None
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-safe mapping.

        Recorded on every retrieval result, so a set of hits can always be
        traced back to the settings that produced it.

        Returns
        -------
        dict
            All fields, with ``filters`` copied rather than shared.
        """
        payload = asdict(self)
        payload["filters"] = None if self.filters is None else dict(self.filters)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrieveConfig:
        """Rebuild a config from stored settings.

        Missing keys fall back to defaults, so configs written by older versions
        still load.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`.

        Returns
        -------
        RetrieveConfig
            The reconstructed config.

        Notes
        -----
        **A stored ``mode='hybrid'`` may not be honoured** if the optional
        dependencies are absent in the loading environment. The retrieval path
        reports what it actually ran.
        """
        filters = payload.get("filters")
        return cls(
            k=int(payload.get("k") or DEFAULT_TOP_K),
            mode=payload.get("mode") or DEFAULT_RETRIEVE_MODE,
            fusion=payload.get("fusion") or DEFAULT_FUSION,
            rrf_k=int(payload.get("rrf_k") or DEFAULT_RRF_K),
            dense_weight=float(payload.get("dense_weight", DEFAULT_DENSE_WEIGHT)),
            bm25_k1=float(payload.get("bm25_k1", DEFAULT_BM25_K1)),
            bm25_b=float(payload.get("bm25_b", DEFAULT_BM25_B)),
            filters=None if filters is None else dict(filters),
            rerank=payload.get("rerank", False),
            rerank_model=payload.get("rerank_model"),
            rerank_candidates=int(
                payload.get("rerank_candidates") or DEFAULT_RERANK_CANDIDATES
            ),
        )


@dataclass(slots=True)
class EvalConfig:
    """How retrieval quality is measured against labelled questions.

    Evaluating retrieval means asking: for a question with a known answer, did
    the passage containing that answer come back in the top ``k``? Everything
    downstream depends on this, because a model cannot answer from a passage it
    never received.

    Attributes
    ----------
    k:
        Cutoff for the metrics. Should match what generation actually uses —
        measuring recall at 20 while generating from 5 reports a number the
        system never benefits from.
    relevance_mode:
        ``'document'`` counts a hit when any chunk of the right document is
        retrieved. ``'chunk'`` requires the specific labelled chunk.
    retrieve:
        The retrieval settings to evaluate. Must match production settings, or
        the measurement describes a system you are not running.

    Notes
    -----
    **Document mode is more forgiving and usually more honest.** Any chunk of
    the right document often contains the answer, and chunk-level labels are
    expensive and brittle to produce. Chunk mode is the stricter measure of
    whether chunking itself is working.

    **This measures retrieval, not answers.** Perfect retrieval and a wrong
    answer is entirely possible; that is a generation problem, and these
    metrics will not show it.

    See Also
    --------
    buildml.rag.evaluate : Where this is applied.
    """

    k: int = DEFAULT_TOP_K
    relevance_mode: RelevanceMode = "document"
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-safe mapping.

        Recorded with evaluation results so a score can be compared only
        against scores measured the same way.

        Returns
        -------
        dict
            The cutoff, relevance mode, and nested retrieval settings.
        """
        return {
            "k": self.k,
            "relevance_mode": self.relevance_mode,
            "retrieve": self.retrieve.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalConfig:
        """Rebuild a config from stored settings.

        Missing keys fall back to defaults, including a default retrieval
        config when none was recorded.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`.

        Returns
        -------
        EvalConfig
            The reconstructed config.
        """
        retrieve_payload = payload.get("retrieve") or {}
        return cls(
            k=int(payload.get("k") or DEFAULT_TOP_K),
            relevance_mode=payload.get("relevance_mode") or "document",
            retrieve=RetrieveConfig.from_dict(retrieve_payload),
        )


DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_GENERATE_TEMPERATURE = 0.0


@dataclass(slots=True)
class GenerateConfig:
    """How retrieved passages become a prompt, and how the model may answer.

    Grounded generation means the model answers from the retrieved passages
    rather than from what it remembers. That constraint is enforced by two
    things: the passages actually placed in the prompt, and the instructions
    that tell the model to stay within them.

    Attributes
    ----------
    k:
        How many passages to include.
    max_tokens:
        Cap on the answer length. ``None`` leaves it to the provider.
    temperature:
        Randomness. **Defaults to zero deliberately** — a grounded answer should
        be determined by the passages, and sampling introduces variation the
        evidence does not support.
    max_context_chars:
        Total character budget for the passages. Enforced *after* ``k``, so
        passages can be dropped even though they were retrieved.
    system_template:
        Instructions establishing that the model must answer from the context
        and say so when it cannot.
    user_template:
        Layout of question and passages in the prompt.

    Notes
    -----
    **``k`` and ``max_context_chars`` are both ceilings, and the tighter one
    wins.** Retrieving ten passages with a budget that fits four means six were
    ranked, returned, and then discarded — no error, and no way to tell from the
    answer.

    **Grounding is an instruction, not a guarantee.** The prompt asks the model
    to stay within the passages; nothing prevents it from drawing on training
    data anyway. Citations are what make that checkable, which is why the result
    carries them.

    **A non-zero temperature makes answers unreproducible.** The same question
    over the same passages will not give the same answer, which complicates
    evaluation and support.

    Examples
    --------
    Deterministic, tightly budgeted generation::

        config = GenerateConfig(k=5, temperature=0.0, max_context_chars=4000)

    See Also
    --------
    buildml.rag.generate.generate_grounded : Where this is applied.
    """

    k: int = DEFAULT_TOP_K
    max_tokens: int | None = None
    temperature: float = DEFAULT_GENERATE_TEMPERATURE
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    system_template: str | None = None
    user_template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-safe mapping.

        Recorded with generation results, so an answer can be traced to the
        budget and temperature that produced it.

        Returns
        -------
        dict
            All fields, including any custom templates.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GenerateConfig:
        """Rebuild a config from stored settings.

        Missing keys fall back to defaults, including the deterministic
        temperature.

        Parameters
        ----------
        payload:
            A mapping as produced by :meth:`to_dict`.

        Returns
        -------
        GenerateConfig
            The reconstructed config.
        """
        return cls(
            k=int(payload.get("k") or DEFAULT_TOP_K),
            max_tokens=payload.get("max_tokens"),
            temperature=float(payload.get("temperature", DEFAULT_GENERATE_TEMPERATURE)),
            max_context_chars=int(
                payload.get("max_context_chars") or DEFAULT_MAX_CONTEXT_CHARS
            ),
            system_template=payload.get("system_template"),
            user_template=payload.get("user_template"),
        )
