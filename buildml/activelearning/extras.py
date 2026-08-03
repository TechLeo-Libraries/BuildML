"""Optional dependency gates for active-learning industry backends.

Native sklearn query strategies are always available. Industry CoreSet/QBC and
torch BALD/MC-dropout paths require optional extras.

See Also
--------
buildml.activelearning.catalog.activelearning_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Any

from buildml.dl.extras import torch_available, torch_spec_available

# Process-wide cache for subprocess import probes. In-process import of
# skactiveml can hard-crash (Windows access violation via torch/skorch) which
# is not catchable with try/except — never probe in-process for availability.
_SKACTIVEML_IMPORTABLE_CACHE: bool | None = None


def scikit_activeml_spec_present() -> bool:
    """Return whether ``skactiveml`` appears on the import path without importing it.

    Used for capability-matrix disclosure before attempting a real import probe.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('skactiveml')`` succeeds.
    """
    return importlib.util.find_spec("skactiveml") is not None


def scikit_activeml_available() -> bool:
    """Return whether scikit-activeml appears installed (``find_spec`` only).

    Does **not** guarantee the package imports cleanly. Broken skorch/skactiveml
    stacks still report ``True`` here; runtime scoring discloses native fallback
    when the real import fails. Prefer :func:`scikit_activeml_importable` for
    honest host-path readiness.

    Returns
    -------
    bool
        ``True`` when :func:`scikit_activeml_spec_present` succeeds.
    """
    return scikit_activeml_spec_present()


def scikit_activeml_importable() -> bool:
    """Return whether scikit-activeml query classes import in a subprocess.

    Uses a subprocess probe so broken torch/skorch stacks that hard-crash on
    import cannot take down the parent BuildML process. Result is cached
    process-wide. Capability matrices should **not** call this (latency); query
    scoring may.

    Returns
    -------
    bool
        ``True`` when ``GreedySamplingX`` imports successfully in a child process.
    """
    global _SKACTIVEML_IMPORTABLE_CACHE
    if _SKACTIVEML_IMPORTABLE_CACHE is not None:
        return _SKACTIVEML_IMPORTABLE_CACHE
    if not scikit_activeml_spec_present():
        _SKACTIVEML_IMPORTABLE_CACHE = False
        return False
    code = (
        "from skactiveml.pool._greedy_sampling import GreedySamplingX; "
        "print('ok')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        ok = proc.returncode == 0 and "ok" in (proc.stdout or "")
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    _SKACTIVEML_IMPORTABLE_CACHE = ok
    return ok


def activelearning_industry_available() -> bool:
    """Return whether industry CoreSet/QBC query strategies can run.

    Native numpy/sklearn scoring is always available in-tree; this gate marks
    the industry backend as usable for catalog defaults and resolve logic.
    Optional scikit-activeml enhancement is disclosed separately via
    ``scikit_activeml_present`` / ``scikit_activeml_importable``.

    Returns
    -------
    bool
        ``True``: industry strategies use the native scorer by default.
    """
    return True


def require_scikit_activeml(*, feature: str = "scikit-activeml industry scoring") -> None:
    """Verify scikit-activeml is installed, or raise :class:`MissingExtraError`.

    Called when an adapter explicitly requires scikit-activeml rather than the
    native industry fallback.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Raises
    ------
    MissingExtraError
        When scikit-activeml is not installed. Install with
        ``pip install 'buildml[activelearning-industry]'``.
    """
    from buildml.core.errors import MissingExtraError

    if not scikit_activeml_spec_present():
        raise MissingExtraError("activelearning-industry", feature)


def require_torch_activelearning(
    *, feature: str = "Torch BALD / MC-dropout active learning"
) -> Any:
    """Import and return ``torch`` for BALD / MC-dropout active-learning backends.

    Delegates to :func:`buildml.dl.extras.require_torch` with active-learning
    wording in the error message.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported torch module.

    Raises
    ------
    MissingExtraError
        When torch is not installed. Install with ``pip install 'buildml[torch]'``.
    """
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


__all__ = [
    "activelearning_industry_available",
    "require_scikit_activeml",
    "require_torch_activelearning",
    "scikit_activeml_available",
    "scikit_activeml_importable",
    "scikit_activeml_spec_present",
    "torch_available",
    "torch_spec_available",
]
