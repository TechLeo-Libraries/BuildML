"""Decision / optimisation catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.optimize.extras import (
    cvxpy_available,
    cvxpy_spec_present,
    mip_available,
    optimize_industry_available,
    ortools_available,
    ortools_spec_present,
    pulp_available,
    pulp_spec_present,
    xgboost_available,
    xgboost_spec_present,
)

DecisionBackendName = Literal["native", "pulp", "ortools", "cvxpy", "calibrated", "xgb"]
DecisionMethodName = Literal[
    "threshold",
    "cost_matrix",
    "topk",
    "knapsack",
    "lp_allocate",
]


def decision_capability_matrix() -> dict[str, Any]:
    """Return an honest capability matrix for decision backends and solvers.

    Summarizes which threshold, cost-matrix, and allocation methods each
    backend supports, install extras, default routing, and non-goals. Consult
    before choosing ``backend=`` in
    :func:`~buildml.optimize.fit.fit_decision_policy`.

    Returns
    -------
    dict[str, Any]
        Nested backend entries, default routing hints, leakage discipline
        notes, and install guidance for ``buildml[optimize-industry]``.
    """
    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "methods": ["threshold", "cost_matrix", "topk", "knapsack", "lp_allocate"],
                "solvers": {
                    "threshold": "threshold_report / sklearn sweep",
                    "knapsack": "numpy DP or density-greedy",
                    "lp_allocate": "scipy.optimize.linprog (HiGHS)",
                },
                "notes": (
                    "Always-available fallback: classical threshold sweep, "
                    "multiclass Bayes cost matrix, top-K, knapsack-lite, scipy LP."
                ),
            },
            "pulp": {
                "available": pulp_available(),
                "extra": "optimize-industry",
                "methods": ["knapsack"],
                "solvers": {"knapsack": "0-1 integer MIP (PuLP + CBC)"},
                "notes": (
                    "Exact 0-1 knapsack via binary integer program. "
                    "Scoped to single-constraint selection: not a general MIP suite."
                ),
            },
            "ortools": {
                "available": ortools_available(),
                "extra": "optimize-industry",
                "methods": ["knapsack"],
                "solvers": {"knapsack": "0-1 integer MIP (OR-Tools SCIP/CBC)"},
                "notes": (
                    "Exact 0-1 knapsack via OR-Tools MIP. Alternative to PuLP when "
                    "ortools is installed without pulp."
                ),
            },
            "cvxpy": {
                "available": cvxpy_available(),
                "extra": "optimize-industry",
                "methods": ["lp_allocate"],
                "solvers": {"lp_allocate": "convex LP (CVXPY + default solver)"},
                "notes": (
                    "Continuous fractional budget allocation. Same problem class as "
                    "native linprog: CVXPY when convex modelling hooks are needed."
                ),
            },
            "calibrated": {
                "available": True,
                "extra": None,
                "methods": ["threshold"],
                "solvers": {"threshold": "CalibratedClassifierCV + cost sweep"},
                "notes": (
                    "Platt/isotonic calibration of Session.fit estimator on train, "
                    "then validation cost-sensitive threshold selection."
                ),
            },
            "xgb": {
                "available": xgboost_available(),
                "extra": "optimize-industry",
                "methods": ["threshold"],
                "solvers": {
                    "threshold": "XGBClassifier scale_pos_weight + validation sweep"
                },
                "notes": (
                    "Cost-sensitive gradient-boosted classifier trained on train; "
                    "threshold tuned on validation expected cost."
                ),
            },
        },
        "allocation_vs_threshold": {
            "threshold": (
                "Binary operating point on scores: wraps tune_threshold engine "
                "or industry cost-sensitive classifiers."
            ),
            "allocation": (
                "topk / knapsack / lp_allocate over candidate scores and costs."
            ),
        },
        "leakage_discipline": {
            "default_partition": "validation",
            "test_tuning": "requires allow_test_tuning=True + disclosure",
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_knapsack_backend_when_installed": _default_knapsack_backend(),
        "default_lp_backend_when_installed": _default_lp_backend(),
        "default_threshold_backend_when_installed": _default_threshold_backend(),
        "install_hints": {
            "optimize-industry": (
                "pip install 'buildml[optimize-industry]'  "
                "# PuLP / OR-Tools MIP knapsack, CVXPY LP, XGB cost-sensitive thresholds"
            ),
        },
        "non_goals": [
            "General mixed-integer programming platform",
            "Multi-period stochastic OR / digital twin",
            "Causal decision analysis",
            "Production fleet scheduling",
        ],
        "industry_extra_present": (
            pulp_spec_present()
            or ortools_spec_present()
            or cvxpy_spec_present()
            or xgboost_spec_present()
        ),
        "industry_runtime_present": optimize_industry_available(),
        "pulp_present": pulp_available(),
        "ortools_present": ortools_available(),
        "cvxpy_present": cvxpy_available(),
        "cvxpy_import_honesty": (
            "cvxpy backend 'available' and industry_runtime_present use "
            "subprocess import probes. industry_extra_present / *_spec_present "
            "are find_spec only. Prefer native linprog unless convex hooks "
            "are explicitly needed."
        ),
        "xgboost_present": xgboost_available(),
        "mip_present": mip_available(),
    }


def optimize_capability_matrix() -> dict[str, Any]:
    """Return the same capability matrix as :func:`decision_capability_matrix`.

    Alias kept for symmetry with other BuildML domain catalogs. Prefer
    :func:`decision_capability_matrix` in new code.

    Returns
    -------
    dict[str, Any]
        Identical backend/solver capability payload from
        :func:`decision_capability_matrix`.
    """
    return decision_capability_matrix()


def _default_backend_when_installed() -> str:
    if xgboost_available():
        return "xgb"
    if pulp_available():
        return "pulp"
    if ortools_available():
        return "ortools"
    if cvxpy_available():
        return "cvxpy"
    return "native"


def _default_knapsack_backend() -> str:
    if pulp_available():
        return "pulp"
    if ortools_available():
        return "ortools"
    return "native"


def _default_lp_backend() -> str:
    # scipy linprog (HiGHS) is always available transitively via sklearn.
    # CVXPY is opt-in when convex modelling hooks are explicitly requested.
    return "native"


def _default_threshold_backend() -> str:
    if xgboost_available():
        return "xgb"
    return "native"


def backend_available(name: DecisionBackendName) -> bool:
    """Return whether a named decision backend is currently usable.

    Looks up the backend entry in :func:`decision_capability_matrix` and
    returns its ``available`` flag (reflecting optional installs).

    Parameters
    ----------
    name:
        Backend identifier such as ``'native'``, ``'pulp'``, or ``'xgb'``.

    Returns
    -------
    bool
        ``True`` when the backend is known and marked available.
    """
    entry = decision_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend(
    *,
    method: str,
    backend: DecisionBackendName | None,
) -> DecisionBackendName:
    """Validate backend/method pairing and apply honest defaults.

    When ``backend`` is ``None``, picks an installed default for knapsack,
    LP, or threshold methods. Raises when the requested backend does not
    support the method or required extras are missing.

    Parameters
    ----------
    method:
        Decision method name (``'knapsack'``, ``'lp_allocate'``,
        ``'threshold'``, etc.).
    backend:
        Explicit backend override; ``None`` selects a default when installed.

    Returns
    -------
    DecisionBackendName
        Resolved backend name safe to pass into fit/apply routing.

    Raises
    ------
    ValidationError
        When the backend is unknown, incompatible with ``method``, or marked
        unavailable without a missing-extra hint.
    MissingExtraError
        When an industry backend is requested but its optional extra is not
        installed.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    matrix = decision_capability_matrix()
    if backend is None:
        if method == "knapsack":
            resolved: DecisionBackendName = _default_knapsack_backend()  # type: ignore[assignment]
        elif method == "lp_allocate":
            resolved = _default_lp_backend()  # type: ignore[assignment]
        elif method == "threshold":
            resolved = _default_threshold_backend()  # type: ignore[assignment]
        else:
            resolved = "native"
    else:
        resolved = backend

    entry = matrix["backends"].get(resolved)
    if entry is None:
        raise ValidationError(f"Unknown decision backend: {resolved!r}")
    if method not in entry.get("methods", []):
        raise ValidationError(
            f"Backend {resolved!r} does not support method={method!r}. "
            f"Supported: {entry.get('methods')}."
        )
    if not entry.get("available"):
        extra = entry.get("extra")
        if extra:
            raise MissingExtraError(str(extra), f"decision backend {resolved!r}")
        raise ValidationError(f"Decision backend {resolved!r} is not available.")
    return resolved


__all__ = [
    "DecisionBackendName",
    "DecisionMethodName",
    "backend_available",
    "decision_capability_matrix",
    "optimize_capability_matrix",
    "resolve_backend",
]
