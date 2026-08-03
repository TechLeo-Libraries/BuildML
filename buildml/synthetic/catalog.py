"""Synthetic-data catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.synthetic.extras import (
    great_expectations_available,
    sdmetrics_available,
    sdv_available,
    synthetic_industry_available,
)

SyntheticBackendName = Literal["native", "sdv"]

NATIVE_METHODS = ("bootstrap", "gaussian_copula", "smote")
SDV_METHODS = ("ctgan", "tvae", "copulagan")


def synthetic_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for synthetic backends and evaluation paths.

Reports installed backends, supported methods, evaluation rules, install hints, and explicit non-goals for teaching overlays.

Returns
-------
dict[str, Any]
    JSON-friendly mapping for history, bundles, or walkthrough overlays.
    """
    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "methods": list(NATIVE_METHODS),
                "notes": (
                    "Bootstrap / Gaussian copula (mixed types) / SMOTE wrap. "
                    "Core numpy/scipy/sklearn; SMOTE needs buildml[imbalanced]."
                ),
            },
            "sdv": {
                "available": sdv_available(),
                "extra": "synthetic-industry",
                "methods": list(SDV_METHODS),
                "notes": (
                    "SDV CTGAN / TVAE / CopulaGAN deep tabular synthesizers "
                    "(buildml[synthetic-industry]). Not differential privacy."
                ),
            },
        },
        "evaluation": {
            "builtin": {
                "available": True,
                "extra": None,
                "modes": ["fidelity", "tstr"],
                "metrics": [
                    "mean_ks",
                    "mean_tv",
                    "corr_l1",
                    "score",
                    "tstr_gap_vs_trtr",
                ],
            },
            "sdmetrics": {
                "available": sdmetrics_available(),
                "extra": "synthetic-industry",
                "modes": ["fidelity"],
                "metrics": [
                    "sdmetrics_overall",
                    "sdmetrics_column_shapes",
                    "sdmetrics_column_pair_trends",
                ],
                "notes": (
                    "SDMetrics QualityReport when sdmetrics is installed; "
                    "falls back to built-in KS/TV/corr fidelity otherwise."
                ),
            },
        },
        "validation": {
            "builtin": {
                "available": True,
                "checks": [
                    "columns_present",
                    "row_count",
                    "null_rate_tolerance",
                    "categorical_vocabulary",
                    "numeric_range_tolerance",
                ],
            },
            "great_expectations": {
                "available": great_expectations_available(),
                "extra": None,
                "notes": (
                    "Optional GE PandasDataset expectations when great_expectations "
                    "is separately installed; built-in validation always runs."
                ),
            },
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_eval_backend_when_installed": _default_eval_backend_when_installed(),
        "install_hints": {
            "synthetic-industry": (
                "pip install 'buildml[synthetic-industry]'  "
                "# SDV CTGAN/TVAE/CopulaGAN + SDMetrics quality reports"
            ),
            "imbalanced": (
                "pip install 'buildml[imbalanced]'  "
                "# native method='smote' via imbalanced-learn"
            ),
        },
        "cross_links": {
            "resample": (
                "Session.resample is class-balance preprocessing (mutates train). "
                "fit_synthesizer fits a reusable generator; merge is explicit."
            ),
        },
        "non_goals": [
            "Differential privacy guarantees",
            "Time-series / sequential SDV models in this tabular path",
            "Full SDV multi-table / relational synthesis",
            "Membership-inference or anonymization audits",
        ],
        "industry_extra_present": synthetic_industry_available(),
        "synthetic_vs_resample": (
            "Synthetic path: reusable generator + optional extend_train merge. "
            "Resample path: in-place class rebalance preprocess."
        ),
    }


def _default_backend_when_installed() -> str:
    if sdv_available():
        return "sdv"
    return "native"


def _default_eval_backend_when_installed() -> str:
    if sdmetrics_available():
        return "sdmetrics"
    return "builtin"


def list_synthetic_methods(
    *,
    backend: SyntheticBackendName | None = None,
) -> list[str]:
    """List synthesizer methods for a backend (or all when backend is None).

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
backend:
    Optional backend override (see capability matrix for identifiers).

Returns
-------
list[str]
    List of string identifiers from the catalog.
    """
    matrix = synthetic_capability_matrix()
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None:
            return []
        return list(entry.get("methods") or [])
    methods: list[str] = []
    for entry in matrix["backends"].values():
        for method in entry.get("methods") or []:
            if method not in methods:
                methods.append(method)
    return methods


def backend_available(name: SyntheticBackendName) -> bool:
    """Return whether backend optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
name:
    Backend or catalog identifier to look up.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    entry = synthetic_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: SyntheticBackendName | None,
    method: str,
) -> tuple[SyntheticBackendName, str]:
    """Validate backend/method pairing and apply honest defaults.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
backend:
    Optional backend override (see capability matrix for identifiers).
method:
    Method or strategy identifier for the resolved backend.

Returns
-------
tuple[SyntheticBackendName, str]
    Tuple of results (tuple[SyntheticBackendName, str]) for downstream Session steps.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_backend: SyntheticBackendName
    if backend is None:
        if method in NATIVE_METHODS:
            resolved_backend = "native"
        elif method in SDV_METHODS:
            resolved_backend = "sdv"
        else:
            raise ValidationError(
                f"Unknown synthesizer method '{method}'. "
                f"Choose from {list_synthetic_methods()}."
            )
    else:
        resolved_backend = backend

    allowed = list_synthetic_methods(backend=resolved_backend)
    if method not in allowed:
        raise ValidationError(
            f"method='{method}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = synthetic_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "synthetic-industry"), f"backend='{resolved_backend}'")
    return resolved_backend, method
