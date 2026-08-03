"""Optional dependency gates for recommender industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def implicit_available() -> bool:
    """Return whether the implicit library imports cleanly.

    Uses ``find_spec`` first, then attempts a real import because some
    environments expose a spec but fail at import time.

    Returns
    -------
    bool
        ``True`` when ``implicit`` is importable for ALS/BPR backends.
    """
    if importlib.util.find_spec("implicit") is None:
        return False
    try:
        import implicit  # noqa: F401
    except Exception:
        return False
    return True


def lightfm_available() -> bool:
    """Return whether LightFM imports cleanly for hybrid recommenders.

    Uses ``find_spec`` first, then attempts a real import because wheel
    availability varies by platform and Python version.

    Returns
    -------
    bool
        ``True`` when ``lightfm`` is importable.
    """
    if importlib.util.find_spec("lightfm") is None:
        return False
    try:
        import lightfm  # noqa: F401
    except Exception:
        return False
    return True


def recommenders_industry_available() -> bool:
    """Return whether any industry recommender backend is importable.

    True when implicit or LightFM is available for optional CF/hybrid paths.

    Returns
    -------
    bool
        ``True`` when at least one industry extra backend imports cleanly.
    """
    return implicit_available() or lightfm_available()


def require_implicit(*, feature: str = "implicit ALS/BPR recommenders") -> Any:
    """Import and return ``implicit``, or raise :class:`MissingExtraError`.

    Called by the implicit adapter at fit/score time so missing extras surface
    as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ``implicit`` module.

    Raises
    ------
    MissingExtraError
        When implicit is not installed. Install with
        ``pip install 'buildml[recommenders-industry]'``.
    """
    try:
        import implicit
    except ImportError as exc:
        raise MissingExtraError("recommenders-industry", feature) from exc
    return implicit


def require_lightfm(*, feature: str = "LightFM hybrid recommender") -> Any:
    """Import and return ``lightfm``, or raise :class:`MissingExtraError`.

    Called by the LightFM adapter at fit/score time so missing extras surface
    as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ``lightfm`` module.

    Raises
    ------
    MissingExtraError
        When LightFM is not installed. Install with
        ``pip install 'buildml[recommenders-lightfm]'``.
    """
    try:
        import lightfm
    except ImportError as exc:
        raise MissingExtraError("recommenders-lightfm", feature) from exc
    return lightfm
