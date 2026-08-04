"""Optional TDA dependency gates for ``buildml[tda]`` and ``buildml[tda-industry]``.

Native TDA uses ripser for Vietoris–Rips persistence and persim for diagram
vectorization. Industry TDA adds giotto-tda (``gtda``) for Betti curves and
sklearn-style transformers. ``require_*`` functions raise
:class:`~buildml.core.errors.MissingExtraError`; ``*_available`` predicates never
raise and use real import probes where DLL load failures matter.

See Also
--------
buildml.tda.catalog.tda_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_giotto(*, feature: str = "Topological Data Analysis (giotto-tda)") -> Any:
    """Import and return ``gtda``, or raise a helpful :class:`MissingExtraError`.

    Call when giotto-tda vectorizers or Mapper summaries are requested.

    Parameters
    ----------
    feature:
        Capability name embedded in the error message.

    Returns
    -------
    module
        The imported ``gtda`` module.

    Raises
    ------
    MissingExtraError
        When giotto-tda is not installed. Install with
        ``pip install 'buildml[tda-industry]'``.
    """
    try:
        import gtda
    except ImportError as exc:
        raise MissingExtraError("tda-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("tda-industry", feature) from exc
    return gtda


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def giotto_spec_present() -> bool:
    """Cheap find_spec discovery for giotto-tda (``gtda``)."""
    return importlib.util.find_spec("gtda") is not None


def giotto_available() -> bool:
    """Return whether ``giotto-tda`` (``gtda``) imports cleanly (subprocess)."""
    if not giotto_spec_present():
        return False
    return _runtime_ok("gtda")


def tda_industry_available() -> bool:
    """Return whether the full industry TDA stack (giotto plus native) is importable.

    Industry paths require both giotto-tda and the native ripser/persim stack.

    Returns
    -------
    bool
        ``True`` when both :func:`giotto_available` and :func:`tda_available`.
    """
    return giotto_available() and tda_available()


def require_ripser(*, feature: str = "Persistent homology (ripser)") -> Any:
    """Import and return ``ripser``, or raise a helpful :class:`MissingExtraError`.

    Native Vietoris–Rips persistence depends on ripser. Call at fit time when
    the capability matrix reports native backend available.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ``ripser`` module.

    Raises
    ------
    MissingExtraError
        When ripser is not installed. Install with ``pip install 'buildml[tda]'``.
    """
    try:
        import ripser
    except ImportError as exc:
        raise MissingExtraError("tda", feature) from exc
    except OSError as exc:
        raise MissingExtraError("tda", feature) from exc
    return ripser


def require_persim(*, feature: str = "Persistence vectorization (persim)") -> Any:
    """Import and return ``persim``, or raise a helpful :class:`MissingExtraError`.

    Native persistence-image and landscape vectorizers use persim helpers. Call
    when vectorizing diagrams on the native backend.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ``persim`` module.

    Raises
    ------
    MissingExtraError
        When persim is not installed. Install with ``pip install 'buildml[tda]'``.
    """
    try:
        import persim
    except ImportError as exc:
        raise MissingExtraError("tda", feature) from exc
    except OSError as exc:
        raise MissingExtraError("tda", feature) from exc
    return persim


def require_tda_stack(*, feature: str = "Topological Data Analysis") -> tuple[Any, Any]:
    """Import both ``ripser`` and ``persim``, or raise :class:`MissingExtraError`.

    Convenience gate for native TDA fit paths that need homology and vectorization.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    tuple[Any, Any]
        ``(ripser_module, persim_module)``.

    Raises
    ------
    MissingExtraError
        When either package is missing from ``buildml[tda]``.
    """
    return require_ripser(feature=feature), require_persim(feature=feature)


def ripser_spec_present() -> bool:
    """Cheap find_spec discovery for ripser."""
    return importlib.util.find_spec("ripser") is not None


def persim_spec_present() -> bool:
    """Cheap find_spec discovery for persim."""
    return importlib.util.find_spec("persim") is not None


def tda_available() -> bool:
    """Return whether both ``ripser`` and ``persim`` import cleanly (subprocess)."""
    if not ripser_spec_present() or not persim_spec_present():
        return False
    return _runtime_ok("ripser") and _runtime_ok("persim")
