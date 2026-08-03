"""Choose RAG defaults from what is actually installed, at call time.

Two demands pull in opposite directions. Users who install the full stack should
get the strong configuration without asking: hybrid retrieval, semantic
embeddings: because a library that hides its best behaviour behind flags is a
library most people use badly. Users with only the core install should still get
something that works, not an import error.

Resolving defaults here, when they are needed rather than when the module is
imported, satisfies both. It also means installing an extra mid-session changes
behaviour immediately, and that tests can exercise both paths in one process.

The cost of that flexibility is that a default is not a constant. Two runs of
identical code on different machines can retrieve differently, which is why
every result records the mode it actually used.

See Also
--------
buildml.rag.types.RetrieveConfig : What these defaults populate.
buildml.rag.extras : The availability probes underneath.
buildml.rag.catalog.rag_capability_matrix : The full installed picture.
"""

from __future__ import annotations

from buildml.rag.types import (
    DEFAULT_FUSION,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    RetrieveConfig,
    RetrieveMode,
)


def rag_semantic_stack_available() -> bool:
    """Report whether the semantic stack is installed and importable.

    The single question every default here turns on. Kept as a named function so
    the intent reads clearly at each call site and so tests can patch one thing.

    Returns
    -------
    bool
        True when sentence-transformers imports cleanly.

    Notes
    -----
    **The import is deferred to the function body**, so this module stays cheap
    to import even in installs that have torch.

    See Also
    --------
    buildml.rag.extras.rag_available : The underlying probe.
    """
    from buildml.rag.extras import rag_available

    return rag_available()


def default_embedder_spec() -> str:
    """Name the embedder to use when the caller has no preference.

    Returns the string ``"auto"`` rather than resolving to a concrete backend
    here. Deferring the choice to :func:`~buildml.rag.embed.resolve_embedder`
    keeps it in one place, and means the decision is made: and disclosed :
    where the embedder is actually built.

    Returns
    -------
    str
        Always ``"auto"``.

    See Also
    --------
    buildml.rag.embed.resolve_embedder : Where ``"auto"`` becomes a backend.
    """
    return "auto"


def default_retrieve_mode() -> RetrieveMode:
    """Pick the retrieval mode that suits the current install.

    Hybrid where the semantic stack is present, dense otherwise. Hybrid is the
    better default when both signals are real: keyword search catches exact
    terms, identifiers, and rare words that embeddings blur together, while
    dense search catches paraphrase. With only hashing embeddings, the "dense"
    side is itself lexical, so fusing it with BM25 adds cost without adding a
    second point of view.

    Returns
    -------
    {'hybrid', 'dense'}
        The mode to use absent an explicit choice.

    Notes
    -----
    **This can differ between machines.** Check ``result.mode`` on a retrieval
    result rather than assuming.
    """
    return "hybrid" if rag_semantic_stack_available() else "dense"


def default_retrieve_config(**overrides: object) -> RetrieveConfig:
    """Build a fully-populated retrieval config, overriding only what you name.

    The convenient way to get a good configuration with one thing changed. Every
    field the caller does not mention is filled from the package defaults, with
    the mode resolved against the current install.

    Parameters
    ----------
    **overrides:
        Any field of :class:`~buildml.rag.types.RetrieveConfig`: ``mode``, ``k``,
        ``fusion``, ``rrf_k``, ``dense_weight``, ``bm25_k1``, ``bm25_b``,
        ``filters``, ``rerank``, ``rerank_model``, ``rerank_candidates``.

    Returns
    -------
    RetrieveConfig
        A complete, validated config.

    Raises
    ------
    TypeError
        If an override does not name a real field. Unknown keys are rejected
        rather than ignored, so a typo surfaces immediately instead of silently
        leaving the default in place.
    ValidationError
        If a value is out of range, raised by the config's own validation.

    Notes
    -----
    **Rerank defaults to off** even where the stack supports it, because it
    costs a model download and a forward pass per candidate. That is a choice
    the caller should make knowingly.

    Examples
    --------
    Ten results, reranked, everything else default::

        cfg = default_retrieve_config(k=10, rerank=True)

    See Also
    --------
    buildml.rag.types.RetrieveConfig : The fields and their meanings.
    """
    mode = overrides.pop("mode", default_retrieve_mode())  # type: ignore[misc]
    k = int(overrides.pop("k", DEFAULT_TOP_K))  # type: ignore[misc]
    fusion = overrides.pop("fusion", DEFAULT_FUSION)  # type: ignore[misc]
    rerank = overrides.pop("rerank", False)  # type: ignore[misc]
    cfg = RetrieveConfig(
        k=k,
        mode=mode,  # type: ignore[arg-type]
        fusion=fusion,  # type: ignore[arg-type]
        rrf_k=int(overrides.pop("rrf_k", DEFAULT_RRF_K)),  # type: ignore[misc]
        dense_weight=float(overrides.pop("dense_weight", 0.5)),  # type: ignore[misc]
        bm25_k1=float(overrides.pop("bm25_k1", 1.5)),  # type: ignore[misc]
        bm25_b=float(overrides.pop("bm25_b", 0.75)),  # type: ignore[misc]
        filters=overrides.pop("filters", None),  # type: ignore[misc]
        rerank=rerank,  # type: ignore[arg-type]
        rerank_model=overrides.pop("rerank_model", None),  # type: ignore[misc]
        rerank_candidates=int(overrides.pop("rerank_candidates", DEFAULT_RERANK_CANDIDATES)),  # type: ignore[misc]
    )
    if overrides:
        raise TypeError(f"Unexpected retrieve config overrides: {sorted(overrides)}")
    return cfg
