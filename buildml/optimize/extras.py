"""Optional dependency gates for optimisation / decision industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def pulp_available() -> bool:
    """Return whether PuLP is importable for MIP knapsack backends.

    Used by the capability matrix and backend routing without importing PuLP
    at module load time.

    Returns
    -------
    bool
        ``True`` when ``pulp`` is installed.
    """
    return importlib.util.find_spec("pulp") is not None


def ortools_available() -> bool:
    """Return whether OR-Tools is importable for MIP knapsack backends.

    Gates OR-Tools knapsack routing separately from PuLP so either industry
    MIP path can be available independently.

    Returns
    -------
    bool
        ``True`` when ``ortools`` is installed.
    """
    return importlib.util.find_spec("ortools") is not None


def cvxpy_available() -> bool:
    """Return whether CVXPY appears installed for convex LP allocation.

    Reflects ``importlib.util.find_spec`` only: broken wheels may still fail
    at :func:`require_cvxpy` time. Prefer native linprog unless convex hooks
    are explicitly needed.

    Returns
    -------
    bool
        ``True`` when the ``cvxpy`` package is discoverable on the path.
    """
    return importlib.util.find_spec("cvxpy") is not None


def xgboost_available() -> bool:
    """Return whether XGBoost is importable for cost-sensitive thresholds.

    Gates the ``backend='xgb'`` threshold path without importing xgboost at
    module load time.

    Returns
    -------
    bool
        ``True`` when ``xgboost`` is installed.
    """
    return importlib.util.find_spec("xgboost") is not None


def mip_available() -> bool:
    """Return whether at least one integer MIP knapsack backend is importable.

    Convenience flag combining PuLP and OR-Tools availability checks.

    Returns
    -------
    bool
        ``True`` when :func:`pulp_available` or :func:`ortools_available`
        returns ``True``.
    """
    return pulp_available() or ortools_available()


def optimize_industry_available() -> bool:
    """Return whether any optimize-industry optional backend is importable.

    True when at least one of MIP (PuLP/OR-Tools), CVXPY, or XGBoost is
    available for industry decision helpers.

    Returns
    -------
    bool
        ``True`` when any industry extra from ``buildml[optimize-industry]``
        is discoverable.
    """
    return mip_available() or cvxpy_available() or xgboost_available()


def require_pulp(*, feature: str = "PuLP 0-1 knapsack MIP allocation") -> Any:
    """Import and return ``pulp``, or raise :class:`MissingExtraError`.

    Called by the PuLP knapsack adapter at solve time so missing extras
    surface as actionable install guidance.

    Parameters
    ----------
    feature:
        Capability name included in the error message.

    Returns
    -------
    module
        The imported ``pulp`` module.

    Raises
    ------
    MissingExtraError
        When PuLP is not installed. Install with
        ``pip install 'buildml[optimize-industry]'``.
    """
    try:
        import pulp
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return pulp


def require_ortools(*, feature: str = "OR-Tools 0-1 knapsack MIP allocation") -> Any:
    """Import and return ``ortools``, or raise :class:`MissingExtraError`.

    Called by the OR-Tools knapsack adapter when ``backend='ortools'`` is
    resolved.

    Parameters
    ----------
    feature:
        Capability name included in the error message.

    Returns
    -------
    module
        The imported ``ortools`` module.

    Raises
    ------
    MissingExtraError
        When OR-Tools is not installed.
    """
    try:
        import ortools
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return ortools


def require_cvxpy(*, feature: str = "CVXPY convex LP allocation") -> Any:
    """Import and return ``cvxpy``, or raise :class:`MissingExtraError`.

    Called by the CVXPY LP adapter when ``backend='cvxpy'`` is resolved.

    Parameters
    ----------
    feature:
        Capability name included in the error message.

    Returns
    -------
    module
        The imported ``cvxpy`` module.

    Raises
    ------
    MissingExtraError
        When CVXPY is not installed or fails to import.
    """
    try:
        import cvxpy
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return cvxpy


def require_xgboost(*, feature: str = "XGBoost cost-sensitive decision threshold") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`.

    Called by the XGB threshold adapter when ``backend='xgb'`` is resolved.

    Parameters
    ----------
    feature:
        Capability name included in the error message.

    Returns
    -------
    module
        The imported ``xgboost`` module.

    Raises
    ------
    MissingExtraError
        When XGBoost is not installed.
    """
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return xgboost


__all__ = [
    "cvxpy_available",
    "mip_available",
    "optimize_industry_available",
    "ortools_available",
    "pulp_available",
    "require_cvxpy",
    "require_ortools",
    "require_pulp",
    "require_xgboost",
    "xgboost_available",
]
