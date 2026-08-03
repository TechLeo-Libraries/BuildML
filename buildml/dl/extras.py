"""Check for PyTorch without making it a hard requirement.

BuildML's deep learning path needs PyTorch; the rest of BuildML does not. This
module is the boundary, so that installing BuildML does not drag in a
multi-gigabyte dependency that most users of the classical path will never touch.

Two shapes of check, used in different places. ``torch_available`` answers a
question and returns a boolean — right for a capability matrix or a test skip.
``require_torch`` returns the module or raises with an install hint — right at
the point where the work genuinely cannot proceed.

The checks are deliberately more careful than a plain import. Torch is
unusually prone to being installed but broken: a CUDA wheel on a machine with
mismatched drivers, or a Windows install whose DLL load fails. Both raise
``OSError`` rather than ``ImportError``, and both are treated as unavailable —
because from the caller's point of view an unusable install and a missing one
are the same situation.

See Also
--------
buildml.core.errors.MissingExtraError : The error, carrying the install hint.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_torch(*, feature: str = "Deep learning (Torch)") -> Any:
    """Import PyTorch, or explain how to install it.

    Call this at the point where the work actually needs Torch, not at module
    import time — keeping the import lazy is what lets the rest of BuildML load
    on a machine without it.

    Parameters
    ----------
    feature:
        What the caller was trying to do. Appears in the error message, so
        ``"Torch DataLoaders"`` produces a more useful failure than a bare
        import error would.

    Returns
    -------
    module
        The ``torch`` module.

    Raises
    ------
    MissingExtraError
        If Torch is absent or cannot initialise. Install with
        ``pip install buildml[dl]``.

    Notes
    -----
    **``OSError`` is treated as a missing extra, not an unexpected crash.** A
    Torch install with mismatched CUDA libraries or a failed Windows DLL load
    raises ``OSError``, and reporting that as a dependency problem with an
    install hint is far more actionable than surfacing the raw loader error.

    See Also
    --------
    torch_available : The boolean form, for capability checks.
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
    """Report whether PyTorch is present and actually usable.

    Checks for the distribution first, then attempts a real import — being
    installed and being importable are different things for Torch, and only the
    second one matters.

    Returns
    -------
    bool
        True when Torch imports cleanly.

    Notes
    -----
    **A broken install reports False.** Any exception during import counts as
    unavailable, since a Torch that cannot import is not a Torch you can train
    with.

    **One failure mode escapes this check.** A few environments — notably
    Windows machines where antivirus scans the CUDA DLLs — kill the process
    during import rather than raising. Nothing in Python can catch that. Tests
    that need to be robust should skip on ``MissingExtraError`` from
    :func:`require_torch` at the point of use rather than gating on this.

    See Also
    --------
    require_torch : The raising form.
    torch_spec_available : Installation check without importing.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def torch_spec_available() -> bool:
    """Report whether a Torch distribution exists, without importing it.

    Consults package metadata only. Cheap and safe — importing Torch takes
    seconds and initialises CUDA, which is too much for a capability listing
    that may never use the answer.

    Returns
    -------
    bool
        True when a Torch distribution is installed. Says nothing about whether
        it imports cleanly.

    See Also
    --------
    torch_available : The stricter check that actually imports.
    """
    return importlib.util.find_spec("torch") is not None
