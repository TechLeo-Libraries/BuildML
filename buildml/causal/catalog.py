"""Causal backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.causal.extras import (
    causal_industry_available,
    dowhy_available,
    econml_available,
)

CausalBackendName = Literal["native", "dowhy", "econml"]
NativeMethodName = Literal["t_learner", "ipw", "aipw"]
DoWhyMethodName = Literal[
    "backdoor_linear",
    "backdoor_propensity_score",
    "backdoor_propensity_weighting",
]
EconMLMethodName = Literal["dml", "causal_forest", "policy_tree"]


def causal_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for causal backends and optional extras."""
    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "methods": ["t_learner", "ipw", "aipw"],
                "refute_kinds": ["placebo_treatment", "random_confounder"],
                "bootstrap_ci": True,
                "causal_graph": False,
                "identification_api": "caller-declared backdoor set only",
                "cate_heterogeneity": False,
                "policy_learning": False,
            },
            "dowhy": {
                "available": dowhy_available(),
                "extra": "causal-industry",
                "methods": [
                    "backdoor_linear",
                    "backdoor_propensity_score",
                    "backdoor_propensity_weighting",
                ],
                "refute_kinds": [
                    "placebo_treatment",
                    "random_confounder",
                    "random_common_cause",
                    "add_unobserved_common_cause",
                    "data_subset",
                    "placebo_outcome",
                ],
                "bootstrap_ci": False,
                "causal_graph": True,
                "identification_api": "DoWhy identify_effect on declared DAG",
                "cate_heterogeneity": False,
                "policy_learning": False,
                "notes": (
                    "DoWhy refutation suite runs when backend='dowhy' and "
                    "buildml[causal-industry] is installed. Graph is built from "
                    "declared confounders → treatment/outcome; not causal discovery."
                ),
            },
            "econml": {
                "available": econml_available(),
                "extra": "causal-industry",
                "methods": ["dml", "causal_forest", "policy_tree"],
                "refute_kinds": ["placebo_treatment", "random_confounder"],
                "bootstrap_ci": True,
                "causal_graph": False,
                "identification_api": "caller-declared backdoor adjustment via DML",
                "cate_heterogeneity": True,
                "policy_learning": True,
                "notes": (
                    "DML / CausalForestDML estimate ATE with optional CATE std; "
                    "policy_tree learns a treatment assignment rule on train — "
                    "not a deployment-ready policy product."
                ),
            },
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "causal-industry": (
                "pip install 'buildml[causal-industry]'  "
                "# DoWhy refutation suite + EconML DML/CATE/policy paths"
            ),
        },
        "non_goals": [
            "Causal discovery / automatic graph learning",
            "IV / front-door identification (instruments refused)",
            "Multi-valued or continuous treatment",
            "Substituting EDA associations for CausalAssumptions",
            "Proof of unconfoundedness from holdout metrics",
        ],
        "industry_extra_present": causal_industry_available(),
        "assumption_gate": (
            "All backends require explicit CausalAssumptions with "
            "acknowledge_unconfoundedness and acknowledge_positivity. "
            "EDA never satisfies this gate."
        ),
    }


def _default_backend_when_installed() -> str:
    if econml_available():
        return "econml"
    if dowhy_available():
        return "dowhy"
    return "native"


def list_causal_methods(*, backend: CausalBackendName | None = None) -> list[str]:
    """List estimation methods for a backend (or all when backend is None)."""
    matrix = causal_capability_matrix()["backends"]
    if backend is not None:
        entry = matrix.get(backend)
        if entry is None:
            return []
        return list(entry.get("methods") or [])
    out: list[str] = []
    for entry in matrix.values():
        for method in entry.get("methods") or []:
            if method not in out:
                out.append(method)
    return out


def list_refute_kinds(*, backend: CausalBackendName | None = None) -> list[str]:
    """List refutation kinds supported by a backend."""
    matrix = causal_capability_matrix()["backends"]
    if backend is not None:
        entry = matrix.get(backend)
        if entry is None:
            return []
        return list(entry.get("refute_kinds") or [])
    out: list[str] = []
    for entry in matrix.values():
        for kind in entry.get("refute_kinds") or []:
            if kind not in out:
                out.append(kind)
    return out


def backend_available(name: CausalBackendName) -> bool:
    matrix = causal_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: CausalBackendName | None,
    method: str,
) -> tuple[CausalBackendName, str]:
    """Validate backend/method pairing and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    method_key = str(method).lower().replace("-", "_")
    native = {"t_learner", "ipw", "aipw"}
    dowhy = {
        "backdoor_linear",
        "backdoor_propensity_score",
        "backdoor_propensity_weighting",
    }
    econml = {"dml", "causal_forest", "policy_tree"}

    resolved_backend: CausalBackendName
    if backend is None:
        if method_key in native:
            resolved_backend = "native"
        elif method_key in dowhy:
            resolved_backend = "dowhy"
        elif method_key in econml:
            resolved_backend = "econml"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
            # If default is native but method unknown, fall back to aipw later.
    else:
        resolved_backend = backend

    allowed = list_causal_methods(backend=resolved_backend)
    if method_key not in allowed:
        if backend is None and method_key in native:
            resolved_backend = "native"
            allowed = list_causal_methods(backend="native")
        if method_key not in allowed:
            raise ValidationError(
                f"method='{method}' is not valid for backend='{resolved_backend}'. "
                f"Choose from {allowed}."
            )
    if not backend_available(resolved_backend):
        extra = causal_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(
            str(extra or "causal-industry"),
            f"backend='{resolved_backend}'",
        )
    return resolved_backend, method_key
