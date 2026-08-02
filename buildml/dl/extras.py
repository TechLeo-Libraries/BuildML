"""Optional Torch dependency gate for the DL domain."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_torch(*, feature: str = "Deep learning (Torch)") -> Any:
    """Import and return ``torch``, or raise :class:`MissingExtraError`.

    Parameters
    ----------
    feature:
        Human-readable feature name embedded in the install hint.
    """
    try:
        import torch
    except ImportError as exc:
        raise MissingExtraError("torch", feature) from exc
    except OSError as exc:
        # Broken local wheels (e.g. Windows DLL init) should surface as a missing extra.
        raise MissingExtraError("torch", feature) from exc
    return torch


def torch_available() -> bool:
    """Return True when ``torch`` can be imported and initialized.

    Uses ``find_spec`` first so callers can cheaply detect a missing install.
    Catchable import failures (including OSError from broken wheels) count as
    unavailable. Fatal process-killing faults (some Windows AV DLL scans) cannot
    be caught here; tests should skip on :class:`MissingExtraError` from
    :func:`require_torch` instead of assuming ``find_spec`` alone means usable.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def torch_spec_available() -> bool:
    """Cheap check: is a torch distribution installed (may not import cleanly)?"""
    return importlib.util.find_spec("torch") is not None
