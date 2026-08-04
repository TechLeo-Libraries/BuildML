"""Optional dependency gates for optimisation / decision industry backends.

Industry ``*_available`` predicates use subprocess import probes so broken
wheels are never reported as ready. Use ``*_spec_present`` for cheap discovery
disclosure in capability matrices.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def _runtime_ok(module: str) -> bool:
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok(module)


def pulp_spec_present() -> bool:
    """Cheap find_spec discovery for PuLP."""
    return importlib.util.find_spec("pulp") is not None


def ortools_spec_present() -> bool:
    """Cheap find_spec discovery for OR-Tools."""
    return importlib.util.find_spec("ortools") is not None


def cvxpy_spec_present() -> bool:
    """Cheap find_spec discovery for CVXPY."""
    return importlib.util.find_spec("cvxpy") is not None


def xgboost_spec_present() -> bool:
    """Cheap find_spec discovery for XGBoost."""
    return importlib.util.find_spec("xgboost") is not None


def pulp_available() -> bool:
    """Return whether PuLP imports cleanly for MIP knapsack backends."""
    if not pulp_spec_present():
        return False
    return _runtime_ok("pulp")


def ortools_available() -> bool:
    """Return whether OR-Tools imports cleanly for MIP knapsack backends."""
    if not ortools_spec_present():
        return False
    return _runtime_ok("ortools")


def cvxpy_available() -> bool:
    """Return whether CVXPY imports cleanly for convex LP allocation."""
    if not cvxpy_spec_present():
        return False
    return _runtime_ok("cvxpy")


def xgboost_available() -> bool:
    """Return whether XGBoost imports cleanly for cost-sensitive thresholds."""
    if not xgboost_spec_present():
        return False
    return _runtime_ok("xgboost")


def mip_available() -> bool:
    """Return whether at least one integer MIP knapsack backend imports cleanly."""
    return pulp_available() or ortools_available()


def optimize_industry_available() -> bool:
    """Return whether any optimize-industry optional backend imports cleanly."""
    return mip_available() or cvxpy_available() or xgboost_available()


def require_pulp(*, feature: str = "PuLP 0-1 knapsack MIP allocation") -> Any:
    """Import and return ``pulp``, or raise :class:`MissingExtraError`."""
    try:
        import pulp
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return pulp


def require_ortools(*, feature: str = "OR-Tools 0-1 knapsack MIP allocation") -> Any:
    """Import and return ``ortools``, or raise :class:`MissingExtraError`."""
    try:
        import ortools
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return ortools


def require_cvxpy(*, feature: str = "CVXPY convex LP allocation") -> Any:
    """Import and return ``cvxpy``, or raise :class:`MissingExtraError`."""
    try:
        import cvxpy
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return cvxpy


def require_xgboost(*, feature: str = "XGBoost cost-sensitive decision threshold") -> Any:
    """Import and return ``xgboost``, or raise :class:`MissingExtraError`."""
    try:
        import xgboost
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return xgboost


__all__ = [
    "cvxpy_available",
    "cvxpy_spec_present",
    "mip_available",
    "optimize_industry_available",
    "ortools_available",
    "ortools_spec_present",
    "pulp_available",
    "pulp_spec_present",
    "require_cvxpy",
    "require_ortools",
    "require_pulp",
    "require_xgboost",
    "xgboost_available",
    "xgboost_spec_present",
]
