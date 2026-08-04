"""Save an index to disk and load it back, without re-embedding the corpus.

Building an index is the expensive part of RAG. Embedding a large corpus with a
semantic model takes minutes and, on a paid API, money. Doing that again on
every process start is the difference between a demo and something usable, so
the embeddings are persisted.

A RAG bundle is a directory, deliberately not a single opaque file: ``meta.json``
holding the configuration, ``chunks.jsonl`` holding the text and metadata, and
``embeddings.npy`` holding the vectors. Every part is inspectable with ordinary
tools, and nothing is pickled: a bundle can be loaded without executing
anything it contains.

**This is not a Session checkpoint, and the distinction costs people their
indexes.** A Session checkpoint stores data, roles, splits, history, and
classical plans; it does not store the vector index. Restore a checkpoint and
retrieval will not work until an index is rebuilt or a bundle is loaded. The
boundary is stated in every bundle's metadata for the same reason it is stated
here.

See Also
--------
buildml.rag.index.build_index : Producing what gets saved.
buildml.rag.explain_hooks.rag_status : Seeing whether an index is attached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.rag.embed import resolve_embedder
from buildml.rag.index import RagIndex
from buildml.rag.results import Chunk, IndexResult, RagEvalResult
from buildml.rag.store import NumpyCosineStore
from buildml.rag.types import ChunkConfig, EmbedConfig, IndexConfig

BUNDLE_FORMAT = "buildml.rag_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "RAG bundles, Torch trainer bundles, and Session checkpoints are complementary, "
    "not interchangeable. A RAG bundle (buildml.rag_bundle.v1) stores chunk config, "
    "embedder id/dim, chunk metadata, and embeddings/index files. "
    "A Session checkpoint stores data, roles, splits, history, and optional classical "
    "plans; it does not embed the vector index. A Torch trainer bundle "
    "(buildml.torch_bundle.v1) stores module/optimizer state. "
    "Reload tabular workflow via checkpoint_load; reload retrieval via load_rag_bundle."
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON object per line, UTF-8, without escaping non-ASCII.

    Line-delimited JSON so the file can be read incrementally and inspected with
    ordinary text tools, and so a corrupted line does not destroy the rest.

    Parameters
    ----------
    path:
        Destination, overwritten if it exists.
    rows:
        JSON-serialisable mappings.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read line-delimited JSON, skipping blank lines.

    Parameters
    ----------
    path:
        Source file.

    Returns
    -------
    list of dict
        The parsed objects, in file order.

    Raises
    ------
    json.JSONDecodeError
        If a non-blank line is not valid JSON.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_rag_bundle(
    path: str | Path,
    index: RagIndex,
    *,
    eval_result: RagEvalResult | None = None,
) -> Path:
    """Persist an index so it can be reloaded without re-embedding.

    Writes three files into ``path``: ``meta.json`` with the configuration,
    counts, warnings, disclosures, and an optional evaluation result;
    ``chunks.jsonl`` with the text and metadata of every chunk; and
    ``embeddings.npy`` with the vectors as float32.

    Parameters
    ----------
    path:
        Destination directory. Created if needed; existing bundle files are
        overwritten.
    index:
        The index to save.
    eval_result:
        Evaluation metrics to record alongside, so a bundle can carry evidence
        of how well it retrieved.

    Returns
    -------
    Path
        The bundle directory.

    Raises
    ------
    ValidationError
        If ``index`` is ``None``.
    OSError
        If the directory cannot be created or written.

    Notes
    -----
    **Embeddings dominate the size.** Roughly ``n_chunks × dim × 4`` bytes: a
    hundred thousand chunks at 384 dimensions is about 150 MB, plus the chunk
    text.

    **Writes are not atomic.** An interrupted save can leave a partial bundle;
    :func:`load_rag_bundle` checks for all three files, so a partial bundle
    fails cleanly rather than loading wrong.

    **Nothing is pickled**, so bundles are safe to load and portable across
    Python versions.

    **A custom callable embedder cannot be saved.** Its identity is recorded but
    the function is not, and loading falls back to hashing. See
    :func:`load_rag_bundle`.

    Examples
    --------
    Save with its evaluation attached::

        save_rag_bundle("artifacts/faq_index", index, eval_result=metrics)

    See Also
    --------
    load_rag_bundle : The other half.
    """
    if index is None:
        raise ValidationError("No RagIndex to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    chunks = list(index.chunks)
    _write_jsonl(destination / "chunks.jsonl", [c.to_dict() for c in chunks])
    np.save(destination / "embeddings.npy", np.asarray(index.embeddings, dtype=np.float32))
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "n_chunks": len(chunks),
        "n_documents": index.n_documents,
        "chunk_config": index.chunk_config.to_dict(),
        "embed_config": index.embed_config.to_dict(),
        "index_config": index.index_config.to_dict(),
        "warnings": list(index.warnings),
        "disclosures": list(index.disclosures),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_rag_bundle(path: str | Path) -> RagIndex:
    """Rebuild a queryable index from a saved bundle.

    Reads the chunks and embeddings back, rebuilds the vector store, and
    re-attaches an embedder: that last part being the subtle one. Stored
    vectors are enough to *hold* an index but not to *query* it, because the
    query must be embedded into the same space. So the embedder is reconstructed
    from what the bundle recorded.

    How well that works depends on the backend. A sentence-transformer bundle
    re-instantiates the named model, and the dimension is checked against the
    stored vectors to catch a model that has changed underneath you. A hashing
    bundle reproduces exactly, being deterministic. A custom callable cannot be
    reconstructed at all.

    Parameters
    ----------
    path:
        The bundle directory.

    Returns
    -------
    RagIndex
        A ready-to-query index carrying the original configuration, warnings,
        and disclosures.

    Raises
    ------
    ValidationError
        If any of the three files is missing, the format marker does not match
        ``buildml.rag_bundle.v1``, or the re-instantiated model's dimension
        differs from the stored vectors'.
    OSError
        If the files cannot be read.

    Notes
    -----
    **A bundle saved with a custom callable embedder loads with hashing
    substituted.** Stored vectors came from your function; queries would be
    embedded by a different one, and the two spaces are unrelated, so retrieval
    returns confident nonsense. Rebuild the index with the callable re-supplied.

    **Sentence-transformer bundles need the model available.** First load may
    download it, and needs ``buildml[rag]`` installed.

    **Everything loads into memory**, so a 150 MB bundle needs at least that
    much resident.

    Examples
    --------
    Reload and query::

        index = load_rag_bundle("artifacts/faq_index")
        hits = retrieve(index, "how do I cancel?", k=5)

    See Also
    --------
    save_rag_bundle : Producing the bundle.
    buildml.rag.embed.resolve_embedder : How the embedder is rebound.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    chunks_path = root / "chunks.jsonl"
    emb_path = root / "embeddings.npy"
    if not meta_path.is_file() or not chunks_path.is_file() or not emb_path.is_file():
        raise ValidationError(
            f"Incomplete RAG bundle at {root}. "
            f"Expected meta.json, chunks.jsonl, and embeddings.npy ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported RAG bundle format {fmt!r}; expected {BUNDLE_FORMAT}. "
            "This is not a Session checkpoint or Torch trainer bundle."
        )
    chunk_rows = _read_jsonl(chunks_path)
    chunks = tuple(Chunk.from_dict(row) for row in chunk_rows)
    embeddings = np.load(emb_path).astype(np.float32)
    store = NumpyCosineStore.build(chunks, embeddings)
    embed_cfg = EmbedConfig.from_dict(meta.get("embed_config") or {})
    chunk_cfg = ChunkConfig.from_dict(meta.get("chunk_config") or {})
    index_cfg = IndexConfig.from_dict(meta.get("index_config") or {})

    backend = embed_cfg.backend
    load_warnings = list(meta.get("warnings") or ())
    load_disclosures = list(meta.get("disclosures") or ())
    if backend == "sentence-transformers" and embed_cfg.model_name:
        embedder, resolved_cfg = resolve_embedder(embed_cfg.model_name)
        # Prefer recorded id/dim from disk when dims match.
        if resolved_cfg.dim != embed_cfg.dim:
            raise ValidationError(
                f"Loaded embedder dim {resolved_cfg.dim} != bundle dim {embed_cfg.dim}."
            )
        embed_cfg = EmbedConfig(
            embedder_id=embed_cfg.embedder_id or resolved_cfg.embedder_id,
            dim=embed_cfg.dim,
            backend="sentence-transformers",
            model_name=embed_cfg.model_name,
        )
    else:
        embedder, _ = resolve_embedder("hashing", dim=embed_cfg.dim)
        # Keep recorded embedder_id for disclosure honesty even if we rebound hashing.
        if embed_cfg.backend == "callable":
            note = (
                "Callable embedder cannot be reconstituted from the bundle; "
                "queries are rebound to hashing and will not match stored vectors. "
                "Rebuild the index with the callable re-supplied."
            )
            load_warnings.append(note)
            load_disclosures.append(note)
        elif embed_cfg.backend not in {"hashing", "sentence-transformers", None, ""}:
            note = (
                f"Embedder backend {embed_cfg.backend!r} rebounded to hashing on load; "
                "confirm query embedding space matches stored vectors."
            )
            load_warnings.append(note)
            load_disclosures.append(note)

    return RagIndex(
        store=store,
        embedder=embedder,
        embed_config=embed_cfg,
        chunk_config=chunk_cfg,
        index_config=index_cfg,
        n_documents=int(meta.get("n_documents") or len({c.doc_id for c in chunks})),
        warnings=tuple(load_warnings),
        disclosures=tuple(load_disclosures),
    )


def index_result_from_bundle_meta(meta: dict[str, Any]) -> IndexResult:
    """Describe a bundle from its metadata alone, without loading it.

    Lets a caller report what a bundle contains: chunk counts, embedder,
    dimension, store backend: after reading only ``meta.json``, which is
    cheap where loading the embeddings is not.

    Parameters
    ----------
    meta:
        A parsed ``meta.json`` mapping.

    Returns
    -------
    IndexResult
        The same summary shape :func:`~buildml.rag.index.build_index` produces.

    Notes
    -----
    **Missing keys become zeros and empty strings** rather than raising, so an
    older or hand-edited bundle still yields a usable summary.

    **This describes what was saved, not a live index.** Nothing here can be
    queried.
    """
    embed_cfg = meta.get("embed_config") or {}
    return IndexResult(
        n_chunks=int(meta.get("n_chunks") or 0),
        n_documents=int(meta.get("n_documents") or 0),
        embedder_id=str(embed_cfg.get("embedder_id") or ""),
        dim=int(embed_cfg.get("dim") or 0),
        store_backend=str((meta.get("index_config") or {}).get("store_backend") or ""),
        chunk_config=dict(meta.get("chunk_config") or {}),
        embed_config=dict(embed_cfg),
        warnings=tuple(meta.get("warnings") or ()),
        disclosures=tuple(meta.get("disclosures") or ()),
    )
