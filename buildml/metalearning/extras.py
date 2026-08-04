"""Optional dependency gates for meta-learning industry / torch backends.

Native sklearn prototypical and warm-start paths are always available. Torch
ProtoNet and industry MAML/Reptile require ``buildml[torch]`` and optionally
``buildml[metalearning-industry]``.

See Also
--------
buildml.metalearning.catalog.metalearning_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.dl.extras import torch_available, torch_spec_available


def learn2learn_spec_present() -> bool:
    """Return whether ``learn2learn`` appears on the import path without importing it.

    Used for capability-matrix disclosure before attempting a real import probe.

    Returns
    -------
    bool
        ``True`` when ``importlib.util.find_spec('learn2learn')`` succeeds.
    """
    return importlib.util.find_spec("learn2learn") is not None


def learn2learn_available() -> bool:
    """Return whether learn2learn imports cleanly (subprocess probe)."""
    if not learn2learn_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("learn2learn")


def metalearning_industry_available() -> bool:
    """Return whether industry tabular MAML/Reptile adapters can run.

    Prefer ``learn2learn`` when installed; otherwise the industry adapter uses an
    honest native first-order SGD meta-loop (disclosed in the capability matrix).
    Do **not** claim industry availability from ``find_spec('torch')`` alone.

    Returns
    -------
    bool
        ``True`` when :func:`buildml.dl.extras.torch_available` succeeds.
    """
    return torch_available()


def metalearning_torch_available() -> bool:
    """Return whether the deep tabular ProtoNet encoder path is usable.

    Gates ``prototypical_torch`` without importing torch at module load time.

    Returns
    -------
    bool
        ``True`` when :func:`buildml.dl.extras.torch_available` succeeds.
    """
    return torch_available()


def require_learn2learn(*, feature: str = "MAML/Reptile meta-learning") -> Any:
    """Import and return ``learn2learn``, or raise :class:`MissingExtraError`.

    Called by industry MAML/Reptile adapters when learn2learn is required at fit
    time so missing extras surface as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported learn2learn module.

    Raises
    ------
    MissingExtraError
        When learn2learn is not installed. Install with
        ``pip install 'buildml[metalearning-industry,torch]'``.
    """
    from buildml.core.errors import MissingExtraError

    try:
        import learn2learn
    except ImportError as exc:
        raise MissingExtraError("metalearning-industry", feature) from exc
    return learn2learn


def require_torch_metalearning(*, feature: str = "Torch meta-learning") -> Any:
    """Import and return ``torch`` for meta-learning torch/industry backends.

    Delegates to :func:`buildml.dl.extras.require_torch` with meta-learning
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
    "learn2learn_available",
    "learn2learn_spec_present",
    "metalearning_industry_available",
    "metalearning_torch_available",
    "require_learn2learn",
    "require_torch_metalearning",
    "torch_available",
    "torch_spec_available",
]
