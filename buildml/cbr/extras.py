"""Optional dependency gates for CBR industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def hnswlib_spec_present() -> bool:
    return importlib.util.find_spec("hnswlib") is not None


def faiss_spec_present() -> bool:
    return importlib.util.find_spec("faiss") is not None


def ann_library_available() -> bool:
    """True when hnswlib or faiss is importable (buildml[cbr-industry])."""
    return hnswlib_spec_present() or faiss_spec_present()


def hnswlib_available() -> bool:
    if not hnswlib_spec_present():
        return False
    try:
        import hnswlib  # noqa: F401
    except Exception:
        return False
    return True


def faiss_available() -> bool:
    if not faiss_spec_present():
        return False
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


def cbr_industry_available() -> bool:
    """Industry ANN retrieval (hnswlib or faiss) is importable."""
    return hnswlib_available() or faiss_available()


def sentence_transformers_spec_present() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def text_embedding_available() -> bool:
    """Text/hybrid case embedding via sentence-transformers (rag or ssl extra)."""
    if not sentence_transformers_spec_present():
        return False
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def require_ann_library(*, feature: str = "CBR approximate nearest-neighbor retrieval") -> str:
    """Return preferred ANN library name ('hnswlib' or 'faiss') or raise."""
    if hnswlib_available():
        return "hnswlib"
    if faiss_available():
        return "faiss"
    raise MissingExtraError("cbr-industry", feature)


def require_sentence_transformers(
    *, feature: str = "CBR text case embedding"
) -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("rag", feature) from exc
    except OSError as exc:
        raise MissingExtraError("ssl", feature) from exc
    return sentence_transformers


def require_torch_cbr(*, feature: str = "CBR learned-metric encoder"):
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "ann_library_available",
    "cbr_industry_available",
    "faiss_available",
    "faiss_spec_present",
    "hnswlib_available",
    "hnswlib_spec_present",
    "require_ann_library",
    "require_sentence_transformers",
    "require_torch_cbr",
    "sentence_transformers_spec_present",
    "text_embedding_available",
    "torch_available",
    "torch_spec_available",
]
