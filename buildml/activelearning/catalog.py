"""Active-learning catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.activelearning.extras import (
    activelearning_industry_available,
    scikit_activeml_available,
)
from buildml.dl.extras import torch_spec_available

ActiveLearningBackendName = Literal["sklearn", "industry", "torch"]

SKLEARN_STRATEGIES = (
    "least_confidence",
    "margin",
    "entropy",
    "committee",
    "expected_model_change_lite",
)
INDUSTRY_STRATEGIES = (
    "core_set",
    "qbc_kl",
    "qbc_variation_ratios",
)
TORCH_STRATEGIES = (
    "bald",
    "mc_dropout",
)


def activelearning_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for active-learning backends and strategies."""
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "strategies": list(SKLEARN_STRATEGIES),
                "modality": "tabular",
                "notes": (
                    "Core sklearn classifiers + bagging committee; "
                    "uncertainty and vote-entropy query strategies — always available."
                ),
            },
            "industry": {
                "available": activelearning_industry_available(),
                "extra": "activelearning-industry",
                "strategies": list(INDUSTRY_STRATEGIES),
                "modality": "tabular",
                "notes": (
                    "CoreSet + QBC KL/variation-ratio query strategies via native "
                    "numpy/sklearn scoring. Optional scikit-activeml in "
                    "buildml[activelearning-industry] for enhanced hosts."
                ),
            },
            "torch": {
                "available": torch_spec_available(),
                "extra": "torch",
                "strategies": list(TORCH_STRATEGIES),
                "modality": "tabular",
                "notes": (
                    "MC-dropout tabular classifier with BALD and MC-dropout entropy "
                    "query scoring (buildml[torch])."
                ),
            },
        },
        "human_label_boundary": {
            "label_rows": (
                "Session-primary human-in-the-loop operation — NOT AI-allowlisted."
            ),
            "suggest_query": "Returns indices only; never invents labels.",
        },
        "vs_semisupervised": {
            "active_learning": (
                "Interactive query loop: suggest_query → human label_rows → refit."
            ),
            "semi_supervised": (
                "Passive missing labels: propagation/pseudo-label without oracle loop."
            ),
        },
        "evaluation": {
            "metrics": [
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_macro",
                "recall_macro",
            ],
            "holdout_rule": "labeled holdout rows only; pool never scored as truth",
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_strategy_when_installed": _default_strategy_when_installed(),
        "install_hints": {
            "activelearning-industry": (
                "pip install 'buildml[activelearning-industry]'  "
                "# scikit-activeml CoreSet / QBC enhancements"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# BALD / MC-dropout deep query strategies"
            ),
        },
        "non_goals": [
            "Oracle label synthesis in library core",
            "Querying validation/test partitions",
            "Semi-supervised passive propagation (see buildml.semisupervised)",
            "Stream-based active learning product surfaces",
        ],
        "industry_extra_present": activelearning_industry_available(),
        "scikit_activeml_present": scikit_activeml_available(),
        "torch_spec_present": torch_spec_available(),
    }


def _default_backend_when_installed() -> str:
    if torch_spec_available():
        return "torch"
    return "industry"


def _default_strategy_when_installed() -> str:
    if torch_spec_available():
        return "bald"
    return "core_set"


def list_activelearning_strategies(
    *,
    backend: ActiveLearningBackendName | None = None,
) -> list[str]:
    """List query strategies for a backend (or all available when backend is None)."""
    matrix = activelearning_capability_matrix()
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None:
            return []
        if not entry.get("available"):
            return []
        return list(entry.get("strategies") or [])
    strategies: list[str] = []
    for entry in matrix["backends"].values():
        if not entry.get("available"):
            continue
        for strategy in entry.get("strategies") or []:
            if strategy not in strategies:
                strategies.append(strategy)
    return strategies


def backend_available(name: ActiveLearningBackendName) -> bool:
    matrix = activelearning_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_strategy(
    *,
    backend: ActiveLearningBackendName | None,
    strategy: str,
) -> tuple[ActiveLearningBackendName, str]:
    """Validate backend/strategy pairing and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_backend: ActiveLearningBackendName
    if backend is None:
        if strategy in SKLEARN_STRATEGIES:
            resolved_backend = "sklearn"
        elif strategy in INDUSTRY_STRATEGIES:
            resolved_backend = "industry"
        elif strategy in TORCH_STRATEGIES:
            resolved_backend = "torch"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
            allowed_default = list_activelearning_strategies(backend=resolved_backend)
            if allowed_default:
                strategy = allowed_default[0]
            else:
                resolved_backend = "sklearn"
                strategy = "margin"
    else:
        resolved_backend = backend

    allowed = list_activelearning_strategies(backend=resolved_backend)
    if strategy not in allowed:
        raise ValidationError(
            f"strategy='{strategy}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = activelearning_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "activelearning-industry"), f"backend='{resolved_backend}'")
    return resolved_backend, strategy
