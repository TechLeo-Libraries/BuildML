"""Check for optional RAG dependencies, and fail helpfully when they are absent.

BuildML's RAG stack is designed so the core path works with nothing beyond numpy
and scikit-learn: hashing embeddings, a NumPy cosine store, BM25. The heavier
capabilities — semantic embeddings, cross-encoder reranking, LangChain
interoperability — pull in torch and a model download, and are therefore opt-in.

Two shapes appear here, and the distinction matters. ``require_*`` functions
raise :class:`~buildml.core.errors.MissingExtraError` with the exact install
command; use them at the point of use, when the feature genuinely cannot
proceed. ``*_available`` predicates return a bool and never raise; use them to
choose a default that degrades gracefully.

Imports happen inside the functions, never at module scope, so importing
BuildML stays fast and never triggers a torch import as a side effect.

See Also
--------
buildml.rag.defaults : Picking defaults from what is installed.
buildml.core.errors.MissingExtraError : The error, with its install hint.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_rag_stack(*, feature: str = "RAG") -> None:
    """Confirm the baseline RAG stack is usable, which it always is.

    A deliberate no-op, kept as the gate every Session RAG entrypoint calls. The
    baseline path — hashing embedder, NumPy cosine store, BM25 — needs only core
    dependencies, so there is nothing to check. Keeping the call site means the
    gate is already in place should that ever change.

    Parameters
    ----------
    feature:
        The capability being gated. Unused today; present so the signature
        matches the other gates and so error messages would be specific if this
        ever started raising.

    Notes
    -----
    **This never raises.** For the parts that can be missing, see
    :func:`require_sentence_transformers` and
    :func:`require_langchain_community`.
    """
    _ = feature
    return None


def require_sentence_transformers(
    *,
    feature: str = "Sentence-transformer embeddings",
) -> Any:
    """Import sentence-transformers, or explain how to install it.

    The gate in front of every semantic capability: real embeddings and
    cross-encoder reranking both live in this package.

    Parameters
    ----------
    feature:
        What the caller wanted, quoted back in the error so the user learns
        which capability needs the extra.

    Returns
    -------
    module
        The imported ``sentence_transformers`` module.

    Raises
    ------
    MissingExtraError
        If the package is missing or fails to load.

    Notes
    -----
    **``OSError`` is caught alongside ``ImportError``**, because a broken torch
    installation — a missing CUDA library, an unloadable shared object —
    typically surfaces as an OS-level error rather than an import failure. Both
    mean the same thing to the user: the extra is not usable here.

    **The import is slow the first time**, since torch loads with it.
    """
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("rag", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rag", feature) from exc
    return sentence_transformers


def rag_available() -> bool:
    """Report whether semantic embeddings are usable, without raising.

    The non-fatal counterpart to :func:`require_sentence_transformers`, used to
    pick a default rather than to enforce one.

    Returns
    -------
    bool
        True when sentence-transformers imports cleanly.

    Notes
    -----
    **Checks the spec first, then actually imports.** The spec check is cheap
    and rules out the common case; the import is what catches an installed but
    broken torch, which a spec check would happily call present.

    **The first call is slow** for the same reason the import is. Later calls
    hit Python's module cache.
    """
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    # sentence-transformers imports torch; on Windows a broken torch install can
    # hard-crash the process. Defer to the torch probe first, then try import.
    import sys

    from buildml.dl.extras import _subprocess_import_ok, torch_available

    if not torch_available():
        return False
    if sys.platform == "win32":
        return _subprocess_import_ok("sentence_transformers")
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def require_langchain_community(*, feature: str = "LangChain RAG adapter") -> Any:
    """Import langchain-community, or explain how to install it.

    Gates the LangChain interoperability adapter, which lives behind a separate
    extra from the semantic stack because it carries its own pinned dependency
    tree.

    Parameters
    ----------
    feature:
        What the caller wanted, quoted back in the error.

    Returns
    -------
    module
        The imported ``langchain_community`` module.

    Raises
    ------
    MissingExtraError
        If the package is missing, naming the ``rag-advanced`` extra.

    See Also
    --------
    buildml.rag.adapters.langchain : What this gate protects.
    """
    try:
        import langchain_community
    except ImportError as exc:
        raise MissingExtraError("rag-advanced", feature) from exc
    return langchain_community


def rag_advanced_available() -> bool:
    """Report whether the LangChain adapter is usable, without raising.

    The non-fatal counterpart to :func:`require_langchain_community`, for
    deciding whether to offer the adapter rather than demanding it.

    Returns
    -------
    bool
        True when langchain-community imports cleanly.

    Notes
    -----
    **Every exception is swallowed here**, not just import errors. LangChain's
    import chain is large and version-sensitive, and can fail with deprecation
    machinery or pydantic validation rather than an ``ImportError``. For a
    predicate whose only job is to answer "can I use this?", any failure is a
    no.
    """
    if importlib.util.find_spec("langchain_community") is None:
        return False
    try:
        import langchain_community  # noqa: F401
    except Exception:
        return False
    return True
