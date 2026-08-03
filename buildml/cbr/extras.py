"""Check which optional CBR backends are installed, and fail helpfully if not.

The exact-search path needs nothing beyond numpy and scikit-learn, and it is
correct: it examines every case and returns the genuinely nearest. What the
optional backends buy is speed at scale — approximate indexes that skip most of
the memory, sentence-transformer embeddings for text features, and torch for
learned metrics.

Two shapes appear here. ``*_available`` predicates return a bool and never
raise; use them to pick a default that degrades rather than breaks.
``require_*`` gates raise :class:`~buildml.core.errors.MissingExtraError` with
the install command; use them where a named backend genuinely cannot proceed.

Each predicate checks the module spec first and then actually imports. The spec
check is cheap and rules out the common absence; the import is what catches a
package that is installed but unloadable — a mismatched binary wheel, a broken
CUDA library — which a spec check would happily report as present.

See Also
--------
buildml.cbr.catalog.cbr_capability_matrix : The full installed picture.
buildml.core.errors.MissingExtraError : The error, with its install hint.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def hnswlib_spec_present() -> bool:
    """Report whether hnswlib is installed, without importing it.

    Checks the import path only, so a capability listing can be assembled
    without paying for a compiled extension load.

    Returns
    -------
    bool
        True when the module can be found on the import path.

    Notes
    -----
    **Present is not the same as usable.** hnswlib ships compiled extensions
    that can fail to load; see :func:`hnswlib_available` for the stronger check.
    """
    return importlib.util.find_spec("hnswlib") is not None


def faiss_spec_present() -> bool:
    """Report whether faiss is installed, without importing it.

    Checks the import path only. faiss is heavy enough that importing it just
    to answer a yes-or-no question is not worth the cost.

    Returns
    -------
    bool
        True when the module can be found on the import path.

    Notes
    -----
    **Present is not the same as usable.** faiss is a compiled library and its
    GPU build in particular can fail at import; see :func:`faiss_available`.
    """
    return importlib.util.find_spec("faiss") is not None


def ann_library_available() -> bool:
    """Report whether either approximate-search library appears installed.

    The cheap check, based on module specs alone. Suitable for a capability
    listing where an import is too expensive to justify.

    Returns
    -------
    bool
        True when hnswlib or faiss is on the import path.

    See Also
    --------
    cbr_industry_available : The stronger check that actually imports.
    """
    return hnswlib_spec_present() or faiss_spec_present()


def hnswlib_available() -> bool:
    """Report whether hnswlib actually imports.

    The stronger check: the module is not only findable but loads. This is what
    to call before committing to the approximate backend.

    Returns
    -------
    bool
        True when the module loads cleanly.

    Notes
    -----
    **Every exception is swallowed.** A compiled extension can fail with an
    ``OSError``, a version conflict, or something stranger; for a predicate
    whose only job is to answer "can I use this?", any failure is a no.
    """
    if not hnswlib_spec_present():
        return False
    try:
        import hnswlib  # noqa: F401
    except Exception:
        return False
    return True


def faiss_available() -> bool:
    """Report whether faiss actually imports.

    The stronger check. A faiss install can be present and broken — a mismatched
    build or a missing GPU runtime — and only an import reveals it.

    Returns
    -------
    bool
        True when the module loads cleanly.

    Notes
    -----
    **Every exception is swallowed**, for the same reason as
    :func:`hnswlib_available`: a compiled library has many ways to be present
    and unusable, and they all mean the same thing here.
    """
    if not faiss_spec_present():
        return False
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


def cbr_industry_available() -> bool:
    """Report whether approximate nearest-neighbour search is usable.

    The check that matters before offering the ``'industry'`` backend: not just
    installed, but importable.

    Returns
    -------
    bool
        True when hnswlib or faiss loads cleanly.

    Notes
    -----
    **Approximate search trades recall for speed.** It may miss a true nearest
    neighbour, which is a good bargain over a large memory and no bargain at
    all over a small one, where exact search is already fast.
    """
    return hnswlib_available() or faiss_available()


def sentence_transformers_spec_present() -> bool:
    """Report whether sentence-transformers is installed, without importing it.

    Checks the import path only, which keeps the text-embedding backend out of
    startup cost until something actually asks for it.

    Returns
    -------
    bool
        True when the module can be found on the import path.

    Notes
    -----
    **The cheap check exists because the import is not.** Loading
    sentence-transformers pulls in torch, which is slow enough to matter in a
    capability listing.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


def text_embedding_available() -> bool:
    """Report whether text case features are usable.

    Gates the embedding backend, which turns text columns into vectors so that
    two cases described in different words can still be recognised as similar —
    something no categorical encoding can do.

    Returns
    -------
    bool
        True when sentence-transformers loads cleanly.

    Notes
    -----
    **``OSError`` is caught alongside ``ImportError``**, because a broken torch
    installation typically surfaces as an OS-level failure rather than an import
    one. Both mean the backend is unavailable here.
    """
    if not sentence_transformers_spec_present():
        return False
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def require_ann_library(*, feature: str = "CBR approximate nearest-neighbor retrieval") -> str:
    """Choose an approximate-search library, or explain how to install one.

    Prefers hnswlib over faiss. Both implement approximate nearest-neighbour
    search well; hnswlib is the lighter dependency and installs cleanly in more
    environments, which is the whole basis for the preference.

    Parameters
    ----------
    feature:
        What the caller wanted, quoted back in the error.

    Returns
    -------
    str
        ``'hnswlib'`` or ``'faiss'``.

    Raises
    ------
    MissingExtraError
        If neither library is usable, naming the ``cbr-industry`` extra.

    Notes
    -----
    **The two libraries build different index structures**, so neighbours may
    differ slightly between them. Both approximate; neither guarantees the exact
    nearest set.
    """
    if hnswlib_available():
        return "hnswlib"
    if faiss_available():
        return "faiss"
    raise MissingExtraError("cbr-industry", feature)


def require_sentence_transformers(
    *, feature: str = "CBR text case embedding"
) -> Any:
    """Import sentence-transformers, or explain how to install it.

    The gate in front of text case embedding.

    Parameters
    ----------
    feature:
        What the caller wanted, quoted back in the error.

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
    **The two failure paths name different extras deliberately.** A missing
    package points at ``rag``, which is where these models normally arrive; a
    package that fails at the OS level points at ``ssl``, whose pins are the
    usual fix for a broken native stack.
    """
    try:
        import sentence_transformers
    except ImportError as exc:
        raise MissingExtraError("rag", feature) from exc
    except OSError as exc:
        raise MissingExtraError("ssl", feature) from exc
    return sentence_transformers


def require_torch_cbr(*, feature: str = "CBR learned-metric encoder"):
    """Import torch, or explain how to install it.

    Gates the learned-metric backend, which trains a small encoder so that
    distance reflects what actually predicts the target rather than raw feature
    geometry. Delegates to the shared deep-learning gate so the install guidance
    is identical wherever torch is needed.

    Parameters
    ----------
    feature:
        What the caller wanted, quoted back in the error.

    Returns
    -------
    module
        The imported ``torch`` module.

    Raises
    ------
    MissingExtraError
        If torch is not installed or fails to load.

    See Also
    --------
    buildml.dl.extras.require_torch : The shared gate.
    """
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
