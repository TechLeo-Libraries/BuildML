"""Optional dependency gates for KG industry backends (PyKEEN).

Native numpy TransE/DistMult is always available. PyKEEN RotatE/ComplEx and
torch-backed pipelines require ``buildml[kg-industry]``.

See Also
--------
buildml.kg.catalog.kg_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_pykeen(*, feature: str = "PyKEEN KG backend") -> Any:
    """Import and return ``pykeen``, or raise :class:`MissingExtraError`.

    Ensures torch is available first because PyKEEN training depends on it.
    Called by the PyKEEN adapter at fit time so missing extras surface as
    actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported pykeen module.

    Raises
    ------
    MissingExtraError
        When pykeen or torch is not installed. Install with
        ``pip install 'buildml[kg-industry]'``.
    """
    from buildml.dl.extras import require_torch

    require_torch(feature=f"{feature} (PyKEEN requires torch)")
    try:
        import pykeen
    except ImportError as exc:
        raise MissingExtraError("kg-industry", feature) from exc
    return pykeen


def pykeen_spec_present() -> bool:
    """Cheap find_spec discovery for PyKEEN (does not prove import works)."""
    return importlib.util.find_spec("pykeen") is not None


def pykeen_available() -> bool:
    """Return whether a PyKEEN distribution is discoverable (find_spec).

    Prefer :func:`pykeen_runtime_available` when deciding if the pykeen backend
    can actually train.
    """
    return pykeen_spec_present()


def pykeen_runtime_available() -> bool:
    """Return whether PyKEEN and torch both import cleanly (subprocess).

    Used when deciding if the pykeen backend can actually train, not merely
    appear in the capability matrix install probe.
    """
    if not pykeen_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok, torch_available

    if not torch_available():
        return False
    return _subprocess_import_ok("pykeen")


def kg_industry_available() -> bool:
    """Return whether the KG industry extra (PyKEEN) imports cleanly at runtime.

    Gates capability-matrix ``available`` / backend readiness. Use
    :func:`pykeen_spec_present` for install-discovery disclosure.
    """
    return pykeen_runtime_available()
