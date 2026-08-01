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
    Import failures (including OSError from broken wheels) count as unavailable.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        return False
    return True
