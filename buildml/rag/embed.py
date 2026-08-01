"""Embedding backends for the RAG path."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from buildml.core.errors import ValidationError
from buildml.rag.types import DEFAULT_EMBED_DIM, HASHING_EMBEDDER_ID, EmbedConfig

EmbedFn = Callable[[Sequence[str]], np.ndarray]


class Embedder(Protocol):
    """Minimal embedder contract: texts → float matrix ``[n, dim]``."""

    embedder_id: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic hashed bag-of-features embedder (M1 default).

    Uses sklearn ``HashingVectorizer`` with L2 normalization. Lexical/hashed —
    not a semantic sentence embedding. Disclosures must say so.
    """

    def __init__(self, *, n_features: int = DEFAULT_EMBED_DIM) -> None:
        if n_features <= 0:
            raise ValidationError(f"n_features must be positive; got {n_features}")
        self.dim = int(n_features)
        self.embedder_id = HASHING_EMBEDDER_ID
        self._vectorizer = HashingVectorizer(
            n_features=self.dim,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            analyzer="word",
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vectorizer.transform(list(texts)).astype(np.float32)
        dense = matrix.toarray()
        return normalize(dense, norm="l2", axis=1).astype(np.float32)


class CallableEmbedder:
    """Wrap a ``list[str] → ndarray`` callable as an :class:`Embedder`."""

    def __init__(
        self,
        fn: EmbedFn,
        *,
        dim: int,
        embedder_id: str = "callable",
    ) -> None:
        self._fn = fn
        self.dim = int(dim)
        self.embedder_id = embedder_id

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        arr = np.asarray(self._fn(list(texts)), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise ValidationError(
                f"Embed callable must return shape (n_texts, dim); got {arr.shape} "
                f"for n_texts={len(texts)}"
            )
        if arr.shape[1] != self.dim:
            raise ValidationError(
                f"Embed callable dim mismatch: expected {self.dim}, got {arr.shape[1]}"
            )
        return arr


class SentenceTransformerEmbedder:
    """Optional local sentence-transformer backend (requires ``buildml[rag]``)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        from buildml.rag.extras import require_sentence_transformers

        st = require_sentence_transformers(feature="Sentence-transformer embeddings")
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        self._model = st.SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name
        self.device = device
        self.embedder_id = f"sentence-transformers:{model_name}"
        # Probe dim with an empty-safe encode of a single token.
        probe = np.asarray(self._model.encode(["probe"], convert_to_numpy=True))
        self.dim = int(probe.shape[-1])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = np.asarray(
            self._model.encode(list(texts), convert_to_numpy=True),
            dtype=np.float32,
        )
        return normalize(arr, norm="l2", axis=1).astype(np.float32)


def resolve_embedder(
    embedder: Embedder | EmbedFn | str | None = None,
    *,
    dim: int = DEFAULT_EMBED_DIM,
    device: str | None = None,
) -> tuple[Any, EmbedConfig]:
    """Resolve the public embedder argument into an object + config."""
    if embedder is None or embedder == "hashing":
        if device is not None:
            # Hashing is CPU-only; record the request without claiming GPU use.
            pass
        resolved = HashingEmbedder(n_features=dim)
        cfg = EmbedConfig(
            embedder_id=resolved.embedder_id,
            dim=resolved.dim,
            backend="hashing",
            device=None,
        )
        return resolved, cfg
    if isinstance(embedder, str):
        if embedder in {"sentence-transformers", "minilm"}:
            model = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            model = embedder
        resolved = SentenceTransformerEmbedder(model_name=model, device=device)
        cfg = EmbedConfig(
            embedder_id=resolved.embedder_id,
            dim=resolved.dim,
            backend="sentence-transformers",
            model_name=resolved.model_name,
            device=device,
        )
        return resolved, cfg
    if callable(embedder) and not hasattr(embedder, "encode"):
        resolved = CallableEmbedder(embedder, dim=dim, embedder_id="callable")
        cfg = EmbedConfig(
            embedder_id=resolved.embedder_id,
            dim=resolved.dim,
            backend="callable",
            device=device,
        )
        return resolved, cfg
    # Protocol / object with encode
    if not hasattr(embedder, "encode"):
        raise ValidationError(
            "embedder must be 'hashing', a sentence-transformers model id, "
            "a callable list[str]->ndarray, or an object with .encode(texts)."
        )
    resolved = embedder
    embedder_id = str(getattr(resolved, "embedder_id", type(resolved).__name__))
    resolved_dim = int(getattr(resolved, "dim", dim))
    cfg = EmbedConfig(
        embedder_id=embedder_id,
        dim=resolved_dim,
        backend="callable",
        device=device or getattr(resolved, "device", None),
    )
    return resolved, cfg
