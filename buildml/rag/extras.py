"""Optional RAG dependency gates."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_rag_stack(*, feature: str = "RAG") -> None:
    """Gate Session RAG entrypoints.

    Hashing embedder + NumPy cosine store use core numpy/sklearn and always work.
    Semantic sentence-transformer backends call :func:`require_sentence_transformers`.
    Install contract: ``pip install 'buildml[rag]'`` for HF embeddings and rerank.
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
    """Return True when optional semantic RAG deps can be imported."""
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def require_langchain_community(*, feature: str = "LangChain RAG adapter") -> Any:
    """Import langchain_community or raise :class:`MissingExtraError` for rag-advanced."""
    try:
        import langchain_community
    except ImportError as exc:
        raise MissingExtraError("rag-advanced", feature) from exc
    return langchain_community


def rag_advanced_available() -> bool:
    """True when ``buildml[rag-advanced]`` LangChain pins import cleanly."""
    if importlib.util.find_spec("langchain_community") is None:
        return False
    try:
        import langchain_community  # noqa: F401
    except Exception:
        return False
    return True
