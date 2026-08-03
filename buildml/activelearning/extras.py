"""Optional dependency gates for active-learning industry backends.

Native sklearn query strategies are always available. Industry CoreSet/QBC and
torch BALD/MC-dropout paths require optional extras.

See Also
--------
buildml.activelearning.catalog.activelearning_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.dl.extras import torch_available, torch_spec_available


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
    """Return whether scikit-activeml is installed for industry host enhancements.

    Uses ``find_spec`` only — no import probe — so broken installs are not
    reported as available until a real import is attempted.

    Returns
    -------
    bool
        ``True`` when :func:`scikit_activeml_spec_present` succeeds.
    """
    return scikit_activeml_spec_present()


def activelearning_industry_available() -> bool:
    """Return whether industry CoreSet/QBC query strategies can run.

    Native numpy/sklearn scoring is always available in-tree; this gate marks
    the industry backend as usable for catalog defaults and resolve logic.

    Returns
    -------
    bool
        ``True`` — industry strategies use the native scorer by default.
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
    "scikit_activeml_spec_present",
    "torch_available",
    "torch_spec_available",
]
