"""Optional dependency gates for causal industry backends (DoWhy / EconML).

Native sklearn nuisance ATE paths are always available. DoWhy refutation and
EconML DML/CATE require ``buildml[causal-industry]``.

Industry ``*_available`` predicates use subprocess import probes so broken
installs are never reported as ready. Use ``*_spec_present`` for cheap
discovery disclosure in capability matrices.

See Also
--------
buildml.causal.catalog.causal_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def require_dowhy(*, feature: str = "DoWhy causal backend") -> Any:
    """Import and return ``dowhy``, or raise :class:`MissingExtraError`."""
    try:
        import dowhy
    except ImportError as exc:
        raise MissingExtraError("causal-industry", feature) from exc
    return dowhy


def require_econml(*, feature: str = "EconML causal backend") -> Any:
    """Import and return ``econml``, or raise :class:`MissingExtraError`."""
    try:
        import econml
    except ImportError as exc:
        raise MissingExtraError("causal-industry", feature) from exc
    return econml


def dowhy_spec_present() -> bool:
    """Cheap find_spec discovery for DoWhy."""
    return importlib.util.find_spec("dowhy") is not None


def econml_spec_present() -> bool:
    """Cheap find_spec discovery for EconML."""
    return importlib.util.find_spec("econml") is not None


def dowhy_available() -> bool:
    """Return whether DoWhy imports cleanly for refutation paths."""
    if not dowhy_spec_present():
        return False
    return _runtime_ok("dowhy")


def econml_available() -> bool:
    """Return whether EconML imports cleanly for DML/CATE paths."""
    if not econml_spec_present():
        return False
    return _runtime_ok("econml")


def causal_industry_available() -> bool:
    """Return whether any industry causal backend imports cleanly at runtime."""
    return dowhy_available() or econml_available()


__all__ = [
    "causal_industry_available",
    "dowhy_available",
    "dowhy_spec_present",
    "econml_available",
    "econml_spec_present",
    "require_dowhy",
    "require_econml",
]
