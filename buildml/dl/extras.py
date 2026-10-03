"""Check optional PyTorch availability and import it when required.

Core classical workflows do not require PyTorch. Capability checks return a
boolean, while ``require_torch`` returns the imported module or raises an error
with installation guidance.

An installed package may still fail to import because of incompatible native
libraries. Windows probes run in a subprocess to contain native import crashes.
Probe failures and timeouts are reported as unavailable.

See Also
--------
buildml.core.errors.MissingExtraError : The error, carrying the install hint.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Any

from buildml.core.errors import MissingExtraError

_SUBPROCESS_IMPORT_CACHE: dict[str, bool] = {}

# These scientific stacks can exceed the default 12-second probe limit on
# Windows even when they import successfully. Allow time for native libraries,
# transitive imports, and JIT initialization while retaining a bounded probe.
_SLOW_IMPORT_TIMEOUTS: dict[str, float] = {
    "torch": 90.0,
    "torch_geometric": 90.0,
    "dowhy": 90.0,
    "sdv": 90.0,
    "sdmetrics": 90.0,
    "umap": 90.0,
    "hdbscan": 45.0,
    "sentence_transformers": 90.0,
    "numba": 90.0,
}


def clear_subprocess_import_cache() -> None:
    """Clear cached subprocess import probes (tests / rare reinstalls)."""
    _SUBPROCESS_IMPORT_CACHE.clear()


def _subprocess_import_ok(module: str, *, timeout: float | None = None) -> bool:
    """Import ``module`` in a child process so a hard crash cannot kill us.

    Used on Windows where broken Torch DLL loads can raise a fatal access
    violation instead of a catchable Python exception. Results are cached
    process-wide so walkthrough / capability matrices do not re-spawn probes.
    """
    cached = _SUBPROCESS_IMPORT_CACHE.get(module)
    if cached is not None:
        return cached
    limit = _SLOW_IMPORT_TIMEOUTS.get(module, 12.0) if timeout is None else timeout
    try:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            check=False,
            capture_output=True,
            timeout=limit,
        )
        ok = completed.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    except (OSError, subprocess.SubprocessError):
        ok = False
    _SUBPROCESS_IMPORT_CACHE[module] = ok
    return ok


def require_torch(*, feature: str = "Deep learning (Torch)") -> Any:
    """Import PyTorch, or explain how to install it.

    Call this at the point where the work actually needs Torch, not at module
    import time: keeping the import lazy is what lets the rest of BuildML load
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

    Checks for the distribution first, then attempts a real import: being
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

    **One failure mode escapes this check.** A few environments: notably
    Windows machines where antivirus scans the CUDA DLLs: kill the process
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
    # Windows + broken CUDA/DLL installs can hard-crash the process on import.
    # Probe in a subprocess so capability matrices / EDA never kill the parent.
    if sys.platform == "win32":
        return _subprocess_import_ok("torch")
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def torch_spec_available() -> bool:
    """Report whether a Torch distribution exists, without importing it.

    Consults package metadata only. Cheap and safe: importing Torch takes
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
