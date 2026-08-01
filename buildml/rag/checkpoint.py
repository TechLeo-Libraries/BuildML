"""RAG bundle persistence (distinct from Session checkpoints and Torch bundles)."""

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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    """Write a RAG bundle directory (``buildml.rag_bundle.v1``).

    Layout
    ------
    ``meta.json``, ``chunks.jsonl``, ``embeddings.npy``.
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
    """Load a RAG bundle into a :class:`RagIndex`.

    Rebinds the default hashing embedder (or recorded callable backend via hashing
    fallback) so ``rag_retrieve`` works after load. Sentence-transformer bundles
    re-instantiate the named model when the ``rag`` extra is installed.
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
            # Callable embedders cannot be reconstituted; rebound hashing for query
            # only when dims match. Callers needing custom encode must re-supply it.
            pass

    return RagIndex(
        store=store,
        embedder=embedder,
        embed_config=embed_cfg,
        chunk_config=chunk_cfg,
        index_config=index_cfg,
        n_documents=int(meta.get("n_documents") or len({c.doc_id for c in chunks})),
        warnings=tuple(meta.get("warnings") or ()),
        disclosures=tuple(meta.get("disclosures") or ()),
    )


def index_result_from_bundle_meta(meta: dict[str, Any]) -> IndexResult:
    """Build a compact :class:`IndexResult` from bundle metadata."""
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
