"""Optional dependency gates for federated industry backends (Flower / flwr)."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def flwr_spec_present() -> bool:
    """Return whether ``flwr`` appears on the import path without importing it.

    Cheap discovery only — a find_spec hit can still fail at import time.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('flwr')`` succeeds.
    """
    return importlib.util.find_spec("flwr") is not None


def flwr_available() -> bool:
    """Return whether ``flwr`` is discoverable (find_spec).

    Prefer :func:`flwr_runtime_available` when deciding if Flower can run.
    Kept as the cheap discovery alias for install probes / extras flags.
    """
    return flwr_spec_present()


def flwr_runtime_available() -> bool:
    """Return whether ``flwr`` imports cleanly in a child process.

    Flower stacks can hard-crash on broken native deps; subprocess isolation
    keeps capability matrices from taking down the host process.
    """
    if not flwr_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("flwr")


def federated_industry_available() -> bool:
    """Return whether the Flower federated backend extra is usable at runtime.

    Gates ``backend='flower'`` on a successful import probe, not find_spec alone.

    Returns
    -------
    bool
        ``True`` when :func:`flwr_runtime_available` succeeds.
    """
    return flwr_runtime_available()


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
