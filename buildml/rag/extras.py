"""Optional RAG dependency gate."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_rag_stack(*, feature: str = "RAG") -> None:
    """Gate Session RAG entrypoints.

    M1's default hashing embedder + NumPy cosine store use core numpy/sklearn, so
    this check always succeeds when BuildML itself is importable. Semantic
    sentence-transformer backends still call :func:`require_sentence_transformers`.
    The install contract remains ``pip install 'buildml[rag]'`` for the declared
    optional pins and future store backends.
    """
    _ = feature
    return None


def require_sentence_transformers(
    *,
    feature: str = "Sentence-transformer embeddings",
) -> Any:
    """Import and return ``sentence_transformers``, or raise :class:`MissingExtraError`."""
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("rag", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rag", feature) from exc
    return sentence_transformers


def rag_available() -> bool:
    """Return True when optional semantic RAG deps can be imported.

    The hashing default path does not require this; Session retrieve still works
    without sentence-transformers.
    """
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, OSError):
        return False
    return True
