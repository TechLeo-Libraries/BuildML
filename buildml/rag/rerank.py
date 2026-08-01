"""Optional local cross-encoder rerank behind the ``rag`` extra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.results import Hit

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def require_cross_encoder(
    *,
    feature: str = "Cross-encoder rerank",
) -> Any:
    """Import sentence-transformers CrossEncoder or raise MissingExtraError."""
    from buildml.rag.extras import require_sentence_transformers

    st = require_sentence_transformers(feature=feature)
    if not hasattr(st, "CrossEncoder"):
        from buildml.core.errors import MissingExtraError

        raise MissingExtraError("rag", feature)
    return st.CrossEncoder


class CrossEncoderReranker:
    """Local cross-encoder reranker (requires ``buildml[rag]``)."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        *,
        device: str | None = None,
    ) -> None:
        CrossEncoder = require_cross_encoder(feature="Cross-encoder rerank")
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        self._model = CrossEncoder(model_name, **kwargs)
        self.model_name = model_name
        self.device = device

    def rerank(self, query: str, hits: Sequence[Hit], *, k: int) -> list[Hit]:
        """Score ``(query, hit.text)`` pairs and return top-``k`` by score."""
        if k <= 0:
            raise ValidationError(f"k must be positive; got {k}")
        if not hits:
            return []
        pairs = [[query, h.text] for h in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk_id),
        )[:k]
        out: list[Hit] = []
        for rank, (hit, score) in enumerate(ranked, start=1):
            out.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    score=float(score),
                    text=hit.text,
                    rank=rank,
                    metadata=dict(hit.metadata),
                )
            )
        return out


def resolve_reranker(
    rerank: bool | str,
    *,
    model_name: str | None = None,
    device: str | None = None,
) -> CrossEncoderReranker | None:
    """Resolve a rerank flag into a cross-encoder or ``None``.

    Accepted truthy values: ``True``, ``"cross-encoder"``, or a model id string.
    ``False`` / ``None`` / ``""`` disable rerank without importing extras.
    """
    if rerank is False or rerank is None or rerank == "":
        return None
    if rerank is True or rerank == "cross-encoder":
        name = model_name or DEFAULT_CROSS_ENCODER
    elif isinstance(rerank, str):
        name = model_name or rerank
    else:
        raise ValidationError(
            "rerank must be False, True, 'cross-encoder', or a model id string."
        )
    return CrossEncoderReranker(model_name=name, device=device)
