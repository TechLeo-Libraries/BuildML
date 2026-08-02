"""Optional dependency gates for optimisation / decision industry backends."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def pulp_available() -> bool:
    return importlib.util.find_spec("pulp") is not None


def ortools_available() -> bool:
    return importlib.util.find_spec("ortools") is not None


def cvxpy_available() -> bool:
    """Reflects package install (find_spec). Broken wheels may fail at require_cvxpy."""
    return importlib.util.find_spec("cvxpy") is not None


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def mip_available() -> bool:
    """True when at least one integer MIP backend is importable."""
    return pulp_available() or ortools_available()


def optimize_industry_available() -> bool:
    """True when any optimize-industry library is importable."""
    return mip_available() or cvxpy_available() or xgboost_available()


def require_pulp(*, feature: str = "PuLP 0-1 knapsack MIP allocation") -> Any:
    try:
        import pulp
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return pulp


def require_ortools(*, feature: str = "OR-Tools 0-1 knapsack MIP allocation") -> Any:
    try:
        import ortools
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return ortools


def require_cvxpy(*, feature: str = "CVXPY convex LP allocation") -> Any:
    try:
        import cvxpy
    except ImportError as exc:
        raise MissingExtraError("optimize-industry", feature) from exc
    return cvxpy


def require_xgboost(*, feature: str = "XGBoost cost-sensitive decision threshold") -> Any:
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
