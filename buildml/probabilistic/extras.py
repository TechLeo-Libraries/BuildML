"""Optional dependency gates for probabilistic industry backends.

Native sklearn Bayesian/GP estimators are always available. MAPIE conformal
and NGBoost distribution boosting require ``buildml[probabilistic-industry]``.

See Also
--------
buildml.probabilistic.catalog.probabilistic_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def mapie_available() -> bool:
    """Return whether MAPIE imports cleanly for conformal prediction paths.

    Used by the capability matrix and fit routing so split/CV+/jackknife+
    methods are offered only when MAPIE is installed.

    Returns
    -------
    bool
        ``True`` when ``mapie`` is importable.
    """
    return importlib.util.find_spec("mapie") is not None


def ngboost_available() -> bool:
    """Return whether NGBoost imports cleanly for distribution boosting paths.

    Gates NGBoost regressor/classifier backends without importing ngboost at
    module load time.

    Returns
    -------
    bool
        ``True`` when ``ngboost`` is importable.
    """
    return importlib.util.find_spec("ngboost") is not None


def probabilistic_industry_available() -> bool:
    """Return whether any industry probabilistic backend (MAPIE or NGBoost) is usable.

    Used when choosing default backends in
    :func:`buildml.probabilistic.catalog.probabilistic_capability_matrix`.

    Returns
    -------
    bool
        ``True`` when at least one industry extra imports cleanly.
    """
    return mapie_available() or ngboost_available()


def require_mapie(*, feature: str = "MAPIE conformal prediction") -> Any:
    """Import and return ``mapie``, or raise :class:`MissingExtraError`.

    Called by the MAPIE adapter at fit time so missing extras surface as
    actionable install guidance instead of opaque import errors.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported mapie module.

    Raises
    ------
    MissingExtraError
        When MAPIE is not installed. Install with
        ``pip install 'buildml[probabilistic-industry]'``.
    """
    try:
        import mapie
    except ImportError as exc:
        raise MissingExtraError("probabilistic-industry", feature) from exc
    return mapie


def require_ngboost(*, feature: str = "NGBoost probabilistic boosting") -> Any:
    """Import and return ``ngboost``, or raise :class:`MissingExtraError`.

    Called by the NGBoost adapter when distribution boosting is requested on
    the industry backend path.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported ngboost module.

    Raises
    ------
    MissingExtraError
        When NGBoost is not installed.
    """
    try:
        import ngboost
    except ImportError as exc:
        raise MissingExtraError("probabilistic-industry", feature) from exc
    return ngboost
