"""Optional dependency gates for online / continual industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def river_spec_present() -> bool:
    """Return whether ``river`` appears on the import path without importing it.

    Used for capability-matrix disclosure before attempting a real import probe.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('river')`` succeeds.
    """
    return importlib.util.find_spec("river") is not None


def river_available() -> bool:
    """Return whether River can be imported for industry streaming online paths.

    Performs a real import probe so broken installs are not reported as available.

    Returns
    -------
    bool
        ``True`` when ``river`` imports cleanly.
    """
    if not river_spec_present():
        return False
    try:
        import river  # noqa: F401
    except Exception:
        return False
    return True


def online_industry_available() -> bool:
    """Return whether industry River streaming adapters can run.

    Gates ``backend='industry'`` without importing River at module load time.

    Returns
    -------
    bool
        ``True`` when :func:`river_available` succeeds.
    """
    return river_available()


def require_river(*, feature: str = "River streaming online learning") -> Any:
    """Import and return ``river``, or raise :class:`MissingExtraError`.

    Called by industry online adapters when River is required at fit time so
    missing extras surface as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported river module.

    Raises
    ------
    MissingExtraError
        When river is not installed. Install with
        ``pip install 'buildml[online-industry]'``.
    """
    try:
        import river  # noqa: F401
    except ImportError as exc:
        raise MissingExtraError("online-industry", feature) from exc
    return river


def require_torch_continual(*, feature: str = "Torch replay / EWC continual learning") -> Any:
    """Import and return ``torch`` for continual tabular MLP backends.

    Delegates to :func:`buildml.dl.extras.require_torch` with online-learning
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
    "online_industry_available",
    "require_river",
    "require_torch_continual",
    "river_available",
    "river_spec_present",
    "torch_available",
    "torch_spec_available",
]
