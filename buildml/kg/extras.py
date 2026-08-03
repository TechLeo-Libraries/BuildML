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


def pykeen_available() -> bool:
    """Return whether a PyKEEN distribution is installed on this machine.

    Uses ``find_spec`` for a cheap catalog probe without importing torch.

    Returns
    -------
    bool
        ``True`` when the ``pykeen`` package is discoverable.
    """
    return importlib.util.find_spec("pykeen") is not None


def pykeen_runtime_available() -> bool:
    """Return whether PyKEEN and torch both import cleanly.

    Used when deciding if the pykeen backend can actually train, not merely
    appear in the capability matrix install probe.

    Returns
    -------
    bool
        ``True`` when both pykeen and torch import successfully.
    """
    if not pykeen_available():
        return False
    from buildml.dl.extras import torch_available

    return torch_available()


def kg_industry_available() -> bool:
    """Return whether the KG industry extra (PyKEEN) is importable.

    Mirrors :func:`pykeen_available` for capability-matrix ``industry_extra_present``.

    Returns
    -------
    bool
        ``True`` when PyKEEN is discoverable on this machine.
    """
    return pykeen_available()
