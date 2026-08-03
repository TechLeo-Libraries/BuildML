"""Optional dependency gates for federated industry backends (Flower / flwr)."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def flwr_available() -> bool:
    """Return whether ``flwr`` appears on the import path without importing it.

    Used for capability-matrix disclosure before attempting a real import probe.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('flwr')`` succeeds.
    """
    return importlib.util.find_spec("flwr") is not None


def federated_industry_available() -> bool:
    """Return whether the Flower federated backend extra is usable.

    Gates ``backend='flower'`` without importing ``flwr`` at module load time.

    Returns
    -------
    bool
        ``True`` when :func:`flwr_available` succeeds.
    """
    return flwr_available()


def require_flwr(*, feature: str = "Flower federated backend") -> Any:
    """Import and return ``flwr``, or raise :class:`MissingExtraError`.

    Called by the Flower adapter at fit time so missing extras surface as
    actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ``flwr`` module.

    Raises
    ------
    MissingExtraError
        When ``flwr`` is not installed. Install with
        ``pip install 'buildml[federated-industry]'``.
    """
    try:
        import flwr
    except ImportError as exc:
        raise MissingExtraError("federated-industry", feature) from exc
    return flwr
