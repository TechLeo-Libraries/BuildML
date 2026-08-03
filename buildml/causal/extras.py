"""Optional dependency gates for causal industry backends (DoWhy / EconML).

Native sklearn nuisance ATE paths are always available. DoWhy refutation and
EconML DML/CATE require ``buildml[causal-industry]``.

See Also
--------
buildml.causal.catalog.causal_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_dowhy(*, feature: str = "DoWhy causal backend") -> Any:
    """Import and return ``dowhy``, or raise :class:`MissingExtraError`.

    Called by the DoWhy adapter at fit and refute time so missing extras
    surface as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported dowhy module.

    Raises
    ------
    MissingExtraError
        When DoWhy is not installed. Install with
        ``pip install 'buildml[causal-industry]'``.
    """
    try:
        import dowhy
    except ImportError as exc:
        raise MissingExtraError("causal-industry", feature) from exc
    return dowhy


def require_econml(*, feature: str = "EconML causal backend") -> Any:
    """Import and return ``econml``, or raise :class:`MissingExtraError`.

    Called by the EconML adapter when DML, causal forest, or policy-tree paths
    are requested on the industry backend.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported econml module.

    Raises
    ------
    MissingExtraError
        When EconML is not installed.
    """
    try:
        import econml
    except ImportError as exc:
        raise MissingExtraError("causal-industry", feature) from exc
    return econml


def dowhy_available() -> bool:
    """Return whether DoWhy is discoverable for refutation paths.

    Used by the capability matrix without importing dowhy at module load time.

    Returns
    -------
    bool
        ``True`` when ``dowhy`` is importable.
    """
    return importlib.util.find_spec("dowhy") is not None


def econml_available() -> bool:
    """Return whether EconML is discoverable for DML/CATE paths.

    Uses ``importlib.util.find_spec`` so the capability matrix and catalog
    can report install status without importing ``econml`` at module load.

    Returns
    -------
    bool
        ``True`` when ``econml`` is importable.
    """
    return importlib.util.find_spec("econml") is not None


def causal_industry_available() -> bool:
    """Return whether any industry causal backend (DoWhy or EconML) is usable.

    Used when choosing default backends in
    :func:`buildml.causal.catalog.causal_capability_matrix`.

    Returns
    -------
    bool
        ``True`` when at least one industry extra imports cleanly.
    """
    return dowhy_available() or econml_available()
