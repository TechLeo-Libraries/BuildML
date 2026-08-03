"""Turn a corpus into something searchable, and keep it current.

Building an index is three steps: cut documents into chunks, embed the chunks,
store the vectors. What makes it worth its own module is everything around
those steps — the leakage refusal, the disclosures, and the incremental updates.

The leakage refusal comes first. Indexing an ``eval_only`` document means every
subsequent retrieval metric measures a system that was shown the answers, so
:func:`build_index` raises rather than filtering, and the same guard runs on
every upsert.

Updates avoid rebuilds. A corpus that changes daily should not be re-embedded
daily, so :meth:`RagIndex.upsert_chunks` re-encodes only what changed and
:meth:`RagIndex.delete` drops rows without touching the rest.

See Also
--------
buildml.rag.chunk : The first step.
buildml.rag.embed : The second.
buildml.rag.store : The third.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.rag.chunk import chunk_documents
from buildml.rag.corpus import corpus_from_documents, refuse_eval_only_index
from buildml.rag.embed import Embedder, EmbedFn, resolve_embedder
from buildml.rag.results import Chunk, ChunkResult, CorpusHandle, Document, IndexResult
from buildml.rag.store import NumpyCosineStore
from buildml.rag.types import HASHING_EMBEDDER_ID, ChunkConfig, EmbedConfig, IndexConfig


class RagIndex:
    """A searchable index, plus the embedder and settings that built it.

    Holding the embedder alongside the vectors is what makes the rest work.
    Queries have to be embedded the same way the passages were, and updates have
    to encode new chunks into the same space — keeping the model here means
    neither can be done with the wrong one by accident.

    Attributes
    ----------
    store:
        The vectors and their chunks.
    embedder:
        The model that produced them, reused for queries and updates.
    embed_config:
        Recorded embedding settings.
    chunk_config:
        Recorded chunking settings, reused when upserting documents so new
        content is cut the same way.
    index_config:
        Recorded store settings.
    n_documents:
        How many distinct documents are represented.
    warnings:
        Problems found while building.
    disclosures:
        Facts affecting how results should be read, including whether the
        placeholder embedder was used.

    Notes
    -----
    **Entirely in memory, and not persisted.** Saving is
    :mod:`buildml.rag.checkpoint`'s job.

    **Mutating methods replace the store in place**, so an index shared across
    threads is not safe to update while it is being queried.

    See Also
    --------
    build_index : What creates one.
    buildml.rag.retrieve : What queries one.
    """

    def __init__(
        self,
        *,
        store: NumpyCosineStore,
        embedder: Any,
        embed_config: EmbedConfig,
        chunk_config: ChunkConfig,
        index_config: IndexConfig,
        n_documents: int,
        warnings: tuple[str, ...] = (),
        disclosures: tuple[str, ...] = (),
    ) -> None:
        """Assemble an index from its parts.

        Normally called by :func:`build_index` rather than directly. Nothing is
        validated here — the components are assumed to be consistent, which they
        are when the builder produced them.

        Parameters
        ----------
        store:
            The vectors and chunks.
        embedder:
            The model that produced them.
        embed_config:
            Embedding settings to record.
        chunk_config:
            Chunking settings, reused for later document upserts.
        index_config:
            Store settings to record.
        n_documents:
            Distinct document count.
        warnings:
            Problems found while building.
        disclosures:
            Notes affecting interpretation.

        Notes
        -----
        **The embedder is not checked against the store.** Constructing this
        with a model that did not produce these vectors yields an index whose
        queries land in a different space.
        """
        self.store = store
        self.embedder = embedder
        self.embed_config = embed_config
        self.chunk_config = chunk_config
        self.index_config = index_config
        self.n_documents = n_documents
        self.warnings = warnings
        self.disclosures = disclosures

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Return the indexed passages, aligned with the vectors.

        Returns
        -------
        tuple of Chunk
            Every chunk in the index, in vector-row order.

        Notes
        -----
        **Position matters.** Element ``i`` corresponds to embedding row ``i``.
        """
        return self.store.chunks

    @property
    def embeddings(self) -> np.ndarray:
        """Return the vector matrix.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_chunks, dim)``, L2-normalised.

        Notes
        -----
        **This is the underlying array, not a copy.** Modifying it corrupts the
        index.
        """
        return self.store.embeddings

    def to_index_result(self) -> IndexResult:
        """Describe the index without exposing its contents.

        Counts, identities, and settings — enough to record what was built and
        to check compatibility, with no vectors and no text.

        Returns
        -------
        IndexResult
            The description, including warnings and disclosures.

        See Also
        --------
        buildml.rag.results.IndexResult : What the fields mean.
        """
        return IndexResult(
            n_chunks=len(self.store.chunks),
            n_documents=self.n_documents,
            embedder_id=self.embed_config.embedder_id,
            dim=self.embed_config.dim,
            store_backend=self.index_config.store_backend,
            chunk_config=self.chunk_config.to_dict(),
            embed_config=self.embed_config.to_dict(),
            warnings=self.warnings,
            disclosures=self.disclosures,
        )

    def _refresh_doc_count(self) -> None:
        self.n_documents = len({c.doc_id for c in self.store.chunks})

    def _refresh_disclosures(self, *, note: str | None = None) -> None:
        notes = [
            f"embedder_id={self.embed_config.embedder_id}",
            f"dim={self.embed_config.dim}",
            f"store_backend={self.index_config.store_backend}",
            f"n_chunks={len(self.store.chunks)}",
            f"n_documents={self.n_documents}",
        ]
        if self.embed_config.embedder_id == HASHING_EMBEDDER_ID:
            notes.append(
                "Default hashing embedder is lexical/hashed, not a semantic sentence model."
            )
        if note:
            notes.append(note)
        self.disclosures = tuple(notes)

    def delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> IndexResult:
        """Drop chunks or whole documents from the index.

        Surviving vectors are kept as they are, so removing outdated content
        costs nothing beyond the removal itself — no re-embedding, no rebuild.

        Parameters
        ----------
        chunk_ids:
            Specific chunks to remove.
        doc_ids:
            Remove every chunk of these documents. The usual case: a document
            was withdrawn or superseded.

        Returns
        -------
        IndexResult
            The index as it now stands, with the removal count in its
            disclosures.

        Raises
        ------
        ValidationError
            If neither argument is given. Deleting nothing is more likely a
            mistake than an intent.

        Notes
        -----
        **Unknown identifiers are ignored silently.** Check the reported chunk
        count to confirm a deletion did what you expected.

        **This mutates the index.** Anything holding it sees the change.

        Examples
        --------
        Remove a superseded document::

            index.delete(doc_ids=["policy-v1"])

        See Also
        --------
        upsert_documents : Replacing content instead of removing it.
        """
        if not chunk_ids and not doc_ids:
            raise ValidationError("rag delete requires chunk_ids and/or doc_ids.")
        before = len(self.store.chunks)
        self.store = self.store.without_ids(chunk_ids=chunk_ids, doc_ids=doc_ids)
        removed = before - len(self.store.chunks)
        self._refresh_doc_count()
        self._refresh_disclosures(note=f"deleted_chunks={removed}")
        return self.to_index_result()

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> IndexResult:
        """Add or replace chunks, re-embedding only what was supplied.

        Chunks whose IDs already exist are replaced; the rest are appended.
        Everything untouched keeps its existing vector, so the cost is
        proportional to the change rather than to the corpus.

        Parameters
        ----------
        chunks:
            The chunks to insert or replace, matched by ``chunk_id``.

        Returns
        -------
        IndexResult
            The index as it now stands, with insert and replace counts in its
            disclosures.

        Raises
        ------
        ValidationError
            If no chunks are given, or the embedder returns vectors of the
            wrong width for this index.

        Notes
        -----
        **Replaced chunks move to the end.** Positions shift, but IDs do not, so
        anything referring to chunks by ID is unaffected.

        **Chunk IDs are positional, and that matters here.** Re-chunking an
        edited document produces the same IDs with different content, so
        upserting a subset of them leaves the rest describing text that no
        longer exists. Upsert the whole document's chunks, or delete first.

        **The index embedder is used**, so new chunks land in the same space.

        See Also
        --------
        upsert_documents : The document-level form.
        delete : Removing instead.
        """
        incoming = list(chunks)
        if not incoming:
            raise ValidationError("upsert_chunks requires at least one chunk.")
        by_id = {c.chunk_id: (i, c) for i, c in enumerate(self.store.chunks)}
        emb = (
            np.asarray(self.store.embeddings, dtype=np.float32)
            if self.store.embeddings.size
            else np.zeros((0, self.embed_config.dim), dtype=np.float32)
        )
        kept_chunks: list[Chunk] = list(self.store.chunks)
        kept_emb_rows: list[np.ndarray] = (
            [emb[i] for i in range(emb.shape[0])] if emb.shape[0] else []
        )

        to_encode: list[Chunk] = []
        replace_ids: set[str] = set()
        for chunk in incoming:
            if chunk.chunk_id in by_id:
                replace_ids.add(chunk.chunk_id)
            to_encode.append(chunk)

        if replace_ids:
            kept_chunks = []
            next_emb_rows: list[np.ndarray] = []
            for c, row in zip(self.store.chunks, kept_emb_rows, strict=True):
                if c.chunk_id in replace_ids:
                    continue
                kept_chunks.append(c)
                next_emb_rows.append(row)
            kept_emb_rows = next_emb_rows

        matrix_new = self.embedder.encode([c.text for c in to_encode])
        if matrix_new.shape[1] != self.embed_config.dim:
            raise ValidationError(
                f"Upsert embed dim {matrix_new.shape[1]} != index dim {self.embed_config.dim}."
            )
        # L2-normalize new rows to match store convention.
        norms = np.linalg.norm(matrix_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        matrix_new = matrix_new / norms

        final_chunks = kept_chunks + to_encode
        if kept_emb_rows:
            final_emb = np.vstack([np.stack(kept_emb_rows, axis=0), matrix_new])
        else:
            final_emb = matrix_new
        self.store = NumpyCosineStore(
            chunks=tuple(final_chunks),
            embeddings=np.asarray(final_emb, dtype=np.float32),
            dim=self.embed_config.dim,
        )
        self._refresh_doc_count()
        self._refresh_disclosures(
            note=f"upserted_chunks={len(to_encode)}; replaced={len(replace_ids)}"
        )
        return self.to_index_result()

    def upsert_documents(
        self,
        documents: Sequence[Document | Mapping[str, Any] | str],
        *,
        chunk: bool = True,
    ) -> IndexResult:
        """Add or replace whole documents, chunked with the index's own settings.

        The convenient form of :meth:`upsert_chunks`. Reusing the recorded
        chunk config is the point: new content is cut exactly as the original
        corpus was, so it is comparable to what is already indexed.

        Parameters
        ----------
        documents:
            Strings, mappings, or ``Document`` objects.
        chunk:
            Cut the documents into chunks. Set false to index each document
            whole, which is appropriate for content already short enough to
            retrieve as a unit.

        Returns
        -------
        IndexResult
            The index as it now stands.

        Raises
        ------
        LeakageError
            If any document is marked ``'eval_only'``.
        ValidationError
            If the documents are malformed or the sequence is empty.

        Notes
        -----
        **The leakage guard runs on every upsert**, not only at build time, so
        held-out documents cannot enter an index later.

        **Adding a document does not replace an earlier version of it** unless
        the chunk IDs coincide. When a document has been edited, delete it by
        ``doc_id`` first — otherwise both versions are searchable and retrieval
        can cite the stale one.

        Examples
        --------
        Replace a document cleanly::

            index.delete(doc_ids=["faq"])
            index.upsert_documents([{"doc_id": "faq", "text": updated}])

        See Also
        --------
        upsert_chunks : The chunk-level form.
        delete : Removing the previous version.
        """
        corpus = corpus_from_documents(documents)
        refuse_eval_only_index(corpus)
        if chunk:
            chunked = chunk_documents(corpus, config=self.chunk_config)
            return self.upsert_chunks(chunked.chunks)
        # Treat each document as a single chunk when chunk=False.
        synthetic = [
            Chunk(
                chunk_id=f"{d.doc_id}::c0",
                doc_id=d.doc_id,
                text=d.text,
                start_char=0,
                end_char=len(d.text),
                metadata=dict(d.metadata),
            )
            for d in corpus.documents
        ]
        return self.upsert_chunks(synthetic)


def build_index(
    corpus: CorpusHandle,
    *,
    chunk_config: ChunkConfig | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedder: Embedder | EmbedFn | str | None = "auto",
    chunks: ChunkResult | Sequence[Chunk] | None = None,
    device: str | None = None,
) -> RagIndex:
    """Chunk, embed, and index a corpus in one call.

    The main entry point. Check the disclosures on the returned index before
    drawing conclusions about retrieval quality — in particular whether a real
    embedding model was used, since the fallback is not semantic and will miss
    every paraphrase.

    Parameters
    ----------
    corpus:
        The documents. Must contain no ``eval_only`` documents.
    chunk_config:
        Chunking settings. Ignored when ``chunks`` is supplied.
    chunk_size:
        Chunk length in characters, overriding ``chunk_config``.
    chunk_overlap:
        Overlap in characters, overriding ``chunk_config``.
    embedder:
        ``'auto'`` picks a real model when the extra is installed and falls back
        to hashing otherwise. Also accepts a model identifier, a callable, or an
        embedder object.
    chunks:
        Pre-computed chunks, to skip chunking entirely.
    device:
        Where to run the embedder.

    Returns
    -------
    RagIndex
        The searchable index, its embedder, and the settings used.

    Raises
    ------
    LeakageError
        If the corpus contains any ``eval_only`` document. Indexing one would
        make every later retrieval metric meaningless, so this refuses rather
        than filtering.
    ValidationError
        If chunking produced nothing to index.
    MissingExtraError
        If a sentence-transformers model was requested without the extra.

    Notes
    -----
    **``'auto'`` is not a guarantee of quality.** Without ``buildml[rag]``
    installed it resolves to the hashing embedder, which matches words rather
    than meaning. The index records this in its disclosures; poor retrieval
    from such an index says nothing about your corpus.

    **Every chunk is embedded now.** On a large corpus with a real model this is
    the slow step, and it is proportional to chunk count — which chunk size
    controls.

    **The recorded dimension comes from the actual matrix**, not from what was
    requested, so it always matches what queries will be checked against.

    Examples
    --------
    Build with a real embedding model::

        index = build_index(corpus, embedder="minilm", chunk_size=1024)
        print(index.to_index_result().disclosures)

    See Also
    --------
    RagIndex : What comes back.
    buildml.rag.retrieve : Querying it.
    buildml.rag.chunk.chunk_documents : Chunking separately first.
    """
    refuse_eval_only_index(corpus)
    cfg = chunk_config or ChunkConfig()
    if chunk_size is not None or chunk_overlap is not None:
        cfg = ChunkConfig(
            size=cfg.size if chunk_size is None else chunk_size,
            overlap=cfg.overlap if chunk_overlap is None else chunk_overlap,
        )
    if chunks is None:
        chunk_result = chunk_documents(corpus, config=cfg)
        chunk_list = list(chunk_result.chunks)
        cfg = ChunkConfig.from_dict(chunk_result.config)
    elif isinstance(chunks, ChunkResult):
        chunk_list = list(chunks.chunks)
        cfg = ChunkConfig.from_dict(chunks.config)
    else:
        chunk_list = list(chunks)
    if not chunk_list:
        raise ValidationError("Cannot build an index from zero chunks.")

    resolved, embed_cfg = resolve_embedder(embedder, device=device)
    texts = [c.text for c in chunk_list]
    matrix = resolved.encode(texts)
    store = NumpyCosineStore.build(chunk_list, matrix)
    # Align recorded dim with actual matrix.
    embed_cfg = EmbedConfig(
        embedder_id=embed_cfg.embedder_id,
        dim=int(matrix.shape[1]),
        backend=embed_cfg.backend,
        model_name=embed_cfg.model_name,
        device=embed_cfg.device or device,
    )
    index_cfg = IndexConfig()
    warnings: list[str] = []
    disclosures = [
        f"embedder_id={embed_cfg.embedder_id}",
        f"dim={embed_cfg.dim}",
        f"store_backend={index_cfg.store_backend}",
        f"n_chunks={len(chunk_list)}",
        f"n_documents={corpus.n_documents}",
    ]
    if embed_cfg.device:
        disclosures.append(f"embed_device={embed_cfg.device}")
    if embed_cfg.embedder_id == HASHING_EMBEDDER_ID:
        disclosures.append(
            "Default hashing embedder is lexical/hashed, not a semantic sentence model."
        )
    return RagIndex(
        store=store,
        embedder=resolved,
        embed_config=embed_cfg,
        chunk_config=cfg,
        index_config=index_cfg,
        n_documents=corpus.n_documents,
        warnings=tuple(warnings),
        disclosures=tuple(disclosures),
    )
