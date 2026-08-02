"""Symbolic / neuro-symbolic catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_spec_available
from buildml.symbolic.extras import (
    imodels_available,
    skope_rules_available,
    symbolic_industry_available,
    torch_neuro_available,
    z3_available,
)

SymbolicBackendName = Literal["sklearn", "industry"]
NeuroSymbolicBackendName = Literal["sklearn", "torch"]
IndustrySymbolicMethod = Literal["skope_rules", "rulefit", "boosted_rules"]
TorchNeuroMethod = Literal["concept_bottleneck_lite", "neural_additive_lite"]


def symbolic_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for symbolic backends and optional extras."""
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "sources": ["declared", "decision_tree", "decision_list"],
                "modality": "tabular",
                "notes": (
                    "Native rule induction: sklearn DecisionTree path export and "
                    "sequential-covering decision lists. Always available."
                ),
            },
            "industry": {
                "available": symbolic_industry_available(),
                "extra": "symbolic-industry",
                "methods": _industry_methods_available(),
                "modality": "tabular",
                "notes": (
                    "Interpretable-model rule export via skope-rules (SkopeRules) "
                    "and imodels (RuleFit, BoostedRules) when installed."
                ),
            },
        },
        "neuro_symbolic_backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "modes": [
                    "constraint_overlay",
                    "rules_as_features",
                    "constraint_repair",
                ],
                "base_estimators": [
                    "logistic_regression",
                    "ridge",
                    "random_forest",
                    "decision_tree",
                ],
                "notes": "sklearn base + symbolic rule hybrid (core fallback).",
            },
            "torch": {
                "available": torch_neuro_available(),
                "extra": "torch",
                "spec_present": torch_spec_available(),
                "modes": [
                    "constraint_overlay",
                    "rules_as_features",
                    "constraint_repair",
                ],
                "methods": ["concept_bottleneck_lite", "neural_additive_lite"],
                "notes": (
                    "Lite tabular concept-bottleneck and neural-additive models "
                    "with the same symbolic overlay / features / repair modes."
                ),
            },
        },
        "constraint_verification": {
            "available": z3_available(),
            "extra": "symbolic-industry",
            "scope": (
                "Optional SAT check on hard constraint rule antecedents via Z3 "
                "(verify_constraints=True on fit). Not a full SMT product."
            ),
        },
        "evaluation": {
            "metrics": ["accuracy", "f1_macro", "rmse", "r2", "rule_coverage"],
            "trace_fields": [
                "fired_rule_ids",
                "chosen_rule_id",
                "neural_prediction",
                "repaired",
            ],
        },
        "default_symbolic_backend_when_installed": _default_symbolic_backend(),
        "default_industry_method_when_installed": _default_industry_method(),
        "default_neuro_backend_when_installed": _default_neuro_backend(),
        "install_hints": {
            "symbolic-industry": (
                "pip install 'buildml[symbolic-industry]'  "
                "# skope-rules + imodels + optional Z3 constraint checks"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# concept-bottleneck / neural-additive neuro-symbolic bases"
            ),
        },
        "non_goals": [
            "Full Prolog / ASP logic programming product",
            "Logic Tensor Networks / differentiable theorem proving",
            "AGI symbolic reasoners",
            "Fuzzy-logic standalone product",
            "Complete Z3 SMT solver product (verification is optional lite check)",
        ],
        "skope_rules_present": skope_rules_available(),
        "imodels_present": imodels_available(),
        "z3_present": z3_available(),
        "torch_spec_present": torch_neuro_available(),
        "industry_extra_present": symbolic_industry_available(),
    }


def _industry_methods_available() -> list[str]:
    out: list[str] = []
    if skope_rules_available():
        out.append("skope_rules")
    if imodels_available():
        out.extend(["rulefit", "boosted_rules"])
    return out


def _default_symbolic_backend() -> str:
    if symbolic_industry_available():
        return "industry"
    return "sklearn"


def _default_industry_method() -> str:
    if skope_rules_available():
        return "skope_rules"
    if imodels_available():
        return "rulefit"
    return "skope_rules"


def _default_neuro_backend() -> str:
    if torch_spec_available():
        return "torch"
    return "sklearn"


def list_symbolic_methods(
    *,
    backend: SymbolicBackendName | None = None,
) -> list[str]:
    """List symbolic induction methods/sources for a backend."""
    matrix = symbolic_capability_matrix()
    if backend == "sklearn" or backend is None:
        sources = list(matrix["backends"]["sklearn"]["sources"])
        if backend == "sklearn":
            return sources
    if backend == "industry":
        entry = matrix["backends"]["industry"]
        if not entry.get("available"):
            return []
        return list(entry.get("methods") or [])
    methods: list[str] = list(matrix["backends"]["sklearn"]["sources"])
    if matrix["backends"]["industry"].get("available"):
        for method in matrix["backends"]["industry"].get("methods") or []:
            if method not in methods:
                methods.append(method)
    return methods


def list_neuro_symbolic_methods(
    *,
    backend: NeuroSymbolicBackendName | None = None,
) -> list[str]:
    matrix = symbolic_capability_matrix()
    if backend == "torch":
        entry = matrix["neuro_symbolic_backends"]["torch"]
        if not entry.get("available"):
            return []
        return list(entry.get("methods") or [])
    if backend == "sklearn" or backend is None:
        return list(
            matrix["neuro_symbolic_backends"]["sklearn"].get("base_estimators") or []
        )
    return []


def backend_available(name: SymbolicBackendName | NeuroSymbolicBackendName) -> bool:
    matrix = symbolic_capability_matrix()
    if name in matrix["backends"]:
        return bool(matrix["backends"][name].get("available"))
    if name in matrix["neuro_symbolic_backends"]:
        return bool(matrix["neuro_symbolic_backends"][name].get("available"))
    return False


def resolve_symbolic_backend_method(
    *,
    backend: SymbolicBackendName | None,
    source: str | None,
    method: str | None,
) -> tuple[SymbolicBackendName, str, str]:
    """Validate backend/source/method and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    source_key = str(source or "decision_tree").lower().replace("-", "_")
    method_key = str(method or "").lower().replace("-", "_") if method else ""

    resolved_backend: SymbolicBackendName
    if backend is None:
        if source_key == "declared":
            resolved_backend = "sklearn"
        elif source_key in {"decision_tree", "decision_list"}:
            resolved_backend = _default_symbolic_backend()  # type: ignore[assignment]
        elif method_key in {"skope_rules", "rulefit", "boosted_rules"}:
            resolved_backend = "industry"
        else:
            resolved_backend = _default_symbolic_backend()  # type: ignore[assignment]
    else:
        resolved_backend = backend

    if source_key == "declared":
        return resolved_backend, "declared", "declared"

    if resolved_backend == "sklearn":
        if source_key not in {"decision_tree", "decision_list"}:
            raise ValidationError(
                f"backend='sklearn' source={source!r} invalid; "
                "expected declared, decision_tree, or decision_list."
            )
        if not backend_available("sklearn"):
            raise MissingExtraError("core", "backend='sklearn'")
        return resolved_backend, source_key, source_key

    # industry backend
    if not backend_available("industry"):
        raise MissingExtraError("symbolic-industry", "backend='industry'")
    allowed = list_symbolic_methods(backend="industry")
    resolved_method = method_key or _default_industry_method()
    if resolved_method not in allowed:
        raise ValidationError(
            f"method='{resolved_method}' is not valid for backend='industry'. "
            f"Choose from {allowed}."
        )
    return resolved_backend, "industry", resolved_method


def resolve_neuro_symbolic_backend(
    *,
    backend: NeuroSymbolicBackendName | None,
    base_estimator: str,
    torch_method: str | None = None,
) -> tuple[NeuroSymbolicBackendName, str]:
    """Validate neuro-symbolic backend and resolve base estimator / torch method."""
    from buildml.core.errors import MissingExtraError, ValidationError

    base_key = str(base_estimator).lower().replace("-", "_")
    torch_key = (
        str(torch_method or base_key).lower().replace("-", "_")
        if torch_method
        else base_key
    )

    resolved_backend: NeuroSymbolicBackendName
    if backend is None:
        if base_key in {"concept_bottleneck_lite", "neural_additive_lite"}:
            resolved_backend = "torch"
        else:
            resolved_backend = "sklearn"
    else:
        resolved_backend = backend

    if resolved_backend == "sklearn":
        allowed = list(
            symbolic_capability_matrix()["neuro_symbolic_backends"]["sklearn"][
                "base_estimators"
            ]
        )
        if base_key not in allowed:
            raise ValidationError(
                f"base_estimator='{base_estimator}' invalid for backend='sklearn'. "
                f"Choose from {allowed}."
            )
        return resolved_backend, base_key

    if not backend_available("torch"):
        raise MissingExtraError("torch", "backend='torch' neuro-symbolic")
    allowed_torch = list_neuro_symbolic_methods(backend="torch")
    resolved_torch = torch_key
    if resolved_torch not in allowed_torch:
        if base_key in allowed_torch:
            resolved_torch = base_key
        else:
            resolved_torch = allowed_torch[0] if allowed_torch else "concept_bottleneck_lite"
    if resolved_torch not in allowed_torch:
        raise ValidationError(
            f"torch method '{resolved_torch}' unavailable. Choose from {allowed_torch}."
        )
    return resolved_backend, resolved_torch
