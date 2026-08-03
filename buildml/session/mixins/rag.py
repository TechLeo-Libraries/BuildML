"""Session mixin: rag domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import rag_ops
from buildml.session.mixins._shared import *  # noqa: F403


class RagSessionMixin:
    """Public Session methods for the rag domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _rag_eval_result: Any
        _rag_generate_result: Any
        _rag_index_result: Any
        _rag_retrieve_result: Any

    @staticmethod
    def rag_capability_matrix() -> dict[str, Any]:
        """
        Report which retrieval-augmented generation stacks are available here.

        Call before :meth:`build_rag_index` or generate helpers to confirm embed,
        store, rerank, and LLM extras on this machine. Read-only introspection.

        Returns
        -------
        dict[str, Any]
            RAG backends, embedders, and install hints from
            :func:`buildml.rag.catalog.rag_capability_matrix`.
        """
        from buildml.rag.catalog import rag_capability_matrix

        return cast("dict[str, Any]", rag_capability_matrix())

    def rag_ingest_corpus(
        self,
        source: str | Path | Sequence[Any] | None = None,
        *,
        text_column: str | None = None,
        id_column: str | None = None,
        glob: str = "*.txt",
        encoding: str = "utf-8",
        role: Literal["index", "eval_only"] = "index",
    ) -> Session:
        """Load a text corpus for the RAG path (requires ``buildml[rag]``).

        Session facade over :func:`buildml.session.rag_ops.rag_ingest_corpus`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with RAG corpus attached for chaining.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_ingest_corpus`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rag_ops.rag_ingest_corpus(
            self,
            source=source,
            text_column=text_column,
            id_column=id_column,
            glob=glob,
            encoding=encoding,
            role=role,
        ))

    def rag_chunk(
        self,
        *,
        size: int = 512,
        overlap: int = 64,
        strategy: str = "fixed",
    ) -> Session:
        """Chunk the active RAG corpus (fixed or recursive strategy).

        Session facade over :func:`buildml.session.rag_ops.rag_chunk`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with chunk result attached for chaining.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_chunk`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rag_ops.rag_chunk(self, size=size, overlap=overlap, strategy=strategy))

    def rag_embed_and_index(
        self,
        *,
        embedder: Any | None = "auto",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunk_strategy: str | None = None,
        device: str | None = None,
    ) -> Session:
        """Embed chunks and build the default NumPy cosine index (requires ``buildml[rag]``).

        Session facade over :func:`buildml.session.rag_ops.rag_embed_and_index`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with RAG index attached for chaining.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_embed_and_index`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rag_ops.rag_embed_and_index(
            self,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
            device=device,
        ))

    def rag_retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        mode: str | None = None,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool | str | None = None,
        config: Any | None = None,
    ) -> Any:
        """Retrieve ranked chunks (dense / BM25 / hybrid) against the active RAG index.

        Session facade over :func:`buildml.session.rag_ops.rag_retrieve`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        RetrieveResult
            Ranked chunks, scores, and retrieve provenance.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_retrieve`
            Canonical documentation for parameters, raises, and examples.
        """
        return rag_ops.rag_retrieve(
            self,
            query=query,
            k=k,
            mode=mode,
            fusion=fusion,
            filters=filters,
            rerank=rerank,
            config=config,
        )

    def rag_evaluate(
        self,
        qrels: Any,
        *,
        k: int = 5,
        relevance_mode: str = "document",
        mode: str | None = None,
        retrieve_config: Any | None = None,
    ) -> Any:
        """Score retrieval with gold qrels (recall@k, MRR, nDCG@k, hit-rate@k).

        Session facade over :func:`buildml.session.rag_ops.rag_evaluate`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        RagEvalResult
            Aggregate retrieval metrics and per-query summaries.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_evaluate`
            Canonical documentation for parameters, raises, and examples.
        """
        return rag_ops.rag_evaluate(
            self,
            qrels=qrels,
            k=k,
            relevance_mode=relevance_mode,
            mode=mode,
            retrieve_config=retrieve_config,
        )

    def rag_generate(
        self,
        query: str,
        *,
        k: int = 5,
        provider: RagChatProvider | None = None,
        mode: str | None = None,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool | str | None = None,
        retrieve_config: RetrieveConfig | None = None,
        config: GenerateConfig | None = None,
        use_last_retrieve: bool = False,
    ) -> GenerateResult:
        """Retrieve (unless reusing the last retrieve) and generate a grounded answer.

        Session facade over :func:`buildml.session.rag_ops.rag_generate`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        GenerateResult
            Answer text, citations (source ids / chunk / doc), and retrieve provenance.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_generate`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("GenerateResult", rag_ops.rag_generate(
            self,
            query=query,
            k=k,
            provider=provider,
            mode=mode,
            fusion=fusion,
            filters=filters,
            rerank=rerank,
            retrieve_config=retrieve_config,
            config=config,
            use_last_retrieve=use_last_retrieve,
        ))

    def rag_upsert(
        self,
        documents: Sequence[Any] | None = None,
        *,
        chunks: Sequence[Any] | None = None,
        chunk: bool = True,
    ) -> Session:
        """Upsert documents or chunks into the active RAG index without a full rebuild.

        Session facade over :func:`buildml.session.rag_ops.rag_upsert`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with updated index and chunk state attached.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_upsert`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rag_ops.rag_upsert(self, documents=documents, chunks=chunks, chunk=chunk))

    def rag_delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> Session:
        """Delete chunks by id and/or parent document id from the active RAG index.

        Session facade over :func:`buildml.session.rag_ops.rag_delete`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with updated index and chunk state attached.

        See Also
        --------
        :func:`buildml.session.rag_ops.rag_delete`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rag_ops.rag_delete(self, chunk_ids=chunk_ids, doc_ids=doc_ids))

    @property
    def rag_index_result(self) -> IndexResult | None:
        """Return the index metadata from the most recent embed-and-index call.

        Stored on Session after :meth:`rag_embed_and_index` or :meth:`load_rag_bundle`.

        Returns
        -------
        IndexResult or None
            ``None`` until :meth:`rag_embed_and_index` or :meth:`load_rag_bundle` has run."""
        return cast("IndexResult | None", self._rag_index_result)

    @property
    def rag_retrieve_result(self) -> RetrieveResult | None:
        """Return the ranked chunks from the most recent retrieval call.

        Stored on Session after :meth:`rag_retrieve` or a generate call that retrieved.

        Returns
        -------
        RetrieveResult or None
            ``None`` until :meth:`rag_retrieve` or :meth:`rag_generate` has run."""
        return cast("RetrieveResult | None", self._rag_retrieve_result)

    @property
    def rag_eval_result(self) -> RagEvalResult | None:
        """Return retrieval metrics from the most recent RAG evaluation.

        Stored on Session after :meth:`rag_evaluate` for offline retrieval QA.

        Returns
        -------
        RagEvalResult or None
            ``None`` until :meth:`rag_evaluate` has run."""
        return cast("RagEvalResult | None", self._rag_eval_result)

    @property
    def rag_generate_result(self) -> GenerateResult | None:
        """Return the grounded answer from the most recent RAG generate call.

        Stored on Session after :meth:`rag_generate` for audit and downstream reuse.

        Returns
        -------
        GenerateResult or None
            ``None`` until :meth:`rag_generate` has run."""
        return cast("GenerateResult | None", self._rag_generate_result)

    def save_rag_bundle(self, path: str | Path) -> Path:
        """Persist the active RAG index as ``buildml.rag_bundle.v1``.

        Session facade over :func:`buildml.session.rag_ops.save_rag_bundle`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            Resolved bundle directory path.

        See Also
        --------
        :func:`buildml.session.rag_ops.save_rag_bundle`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", rag_ops.save_rag_bundle(self, path=path))

    def load_rag_bundle(self, path: str | Path) -> Session:
        """Load a RAG bundle into this Session (requires ``buildml[rag]``).

        Session facade over :func:`buildml.session.rag_ops.load_rag_bundle`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self`` with RAG index attached for chaining.

        See Also
        --------
        :func:`buildml.session.rag_ops.load_rag_bundle`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", rag_ops.load_rag_bundle(self, path=path))
