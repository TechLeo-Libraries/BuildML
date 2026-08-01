"""Dense top-k retrieval."""

from __future__ import annotations

from buildml.core.errors import ValidationError
from buildml.rag.index import RagIndex
from buildml.rag.results import RetrieveResult
from buildml.rag.types import RetrieveConfig


def retrieve(
    index: RagIndex,
    query: str,
    *,
    k: int | None = None,
    config: RetrieveConfig | None = None,
) -> RetrieveResult:
    """Embed ``query`` and return ranked dense hits from ``index``."""
    if index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("query must be a non-empty string.")
    cfg = config or RetrieveConfig()
    top_k = cfg.k if k is None else int(k)
    if top_k <= 0:
        raise ValidationError(f"k must be positive; got {top_k}")
    vector = index.embedder.encode([query])[0]
    hits = index.store.query(vector, k=top_k)
    return RetrieveResult(
        query=query,
        k=top_k,
        hits=tuple(hits),
        embedder_id=index.embed_config.embedder_id,
        disclosures=tuple(index.disclosures),
    )
