"""Meta-learning catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.core.industry_markers import platform_skip_entry
from buildml.metalearning.extras import (
    learn2learn_available,
    metalearning_industry_available,
    metalearning_torch_available,
)
from buildml.dl.extras import torch_spec_available

MetaLearningBackendName = Literal["sklearn", "torch", "industry"]

SKLEARN_METHODS = (
    "prototypical",
    "warm_start",
)
TORCH_METHODS = (
    "prototypical_torch",
)
INDUSTRY_METHODS = (
    "maml",
    "reptile",
)


def metalearning_capability_matrix() -> dict[str, Any]:
    """Build the honest capability matrix for meta-learning backends and methods.

    Reports sklearn, torch, and industry paths, episodic protocol fields,
    evaluation metrics, install hints, and explicit non-goals for teaching
    overlays and Session walkthroughs.

    Returns
    -------
    dict[str, Any]
        Nested backend entries, episodic protocol, evaluation rules, and defaults.
    """
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": list(SKLEARN_METHODS),
                "modality": "tabular",
                "notes": (
                    "Nearest-centroid prototypical and warm-start sklearn adapt: "
                    "always available; honest fallback when torch/industry extras "
                    "are missing."
                ),
            },
            "torch": {
                "available": metalearning_torch_available(),
                "extra": "torch",
                "methods": list(TORCH_METHODS),
                "modality": "tabular",
                "notes": (
                    "PyTorch MLP encoder + episodic prototype loss for tabular "
                    "ProtoNet-style few-shot (buildml[torch]). Small-scale, not "
                    "image/foundation-model meta-learning."
                ),
            },
            "industry": {
                "available": metalearning_industry_available(),
                "extra": "metalearning-industry",
                "methods": list(INDUSTRY_METHODS),
                "modality": "tabular",
                "notes": (
                    "First-order tabular MAML/Reptile when buildml[torch] imports "
                    "cleanly. Uses learn2learn when installed "
                    "(buildml[metalearning-industry]); otherwise an honest native "
                    "first-order SGD meta-loop. Not second-order MAML-at-scale."
                ),
                **platform_skip_entry("learn2learn", extra="metalearning-industry"),
            },
        },
        "platform_markers": [
            platform_skip_entry("learn2learn", extra="metalearning-industry"),
        ],
        "episodic_protocol": {
            "task_column": "role='group' or task_column=",
            "metrics": [
                "meta_train_accuracy",
                "mean_accuracy",
                "mean_f1_macro",
            ],
            "held_out_task_ids": (
                "fit_metalearning may hold out train task ids internally; "
                "evaluate_metalearning prefers them on partition='train'."
            ),
        },
        "evaluation": {
            "metrics": ["mean_accuracy", "mean_f1_macro", "n_tasks_scored"],
            "holdout_rule": (
                "validation/test never used for meta-train; prefer novel task ids "
                "on holdout partitions."
            ),
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_method_when_installed": _default_method_when_installed(),
        "install_hints": {
            "torch": (
                "pip install 'buildml[torch]'  "
                "# deep tabular prototypical encoder"
            ),
            "metalearning-industry": (
                "pip install 'buildml[metalearning-industry,torch]'  "
                "# learn2learn MAML/Reptile tabular adapters"
            ),
        },
        "non_goals": [
            "Foundation-model or vision MAML-at-scale",
            "EconML-style causal meta (see buildml.causal)",
            "Regression few-shot (classification-focused surface)",
            "Neural architecture search across meta-learners",
        ],
        "learn2learn_present": learn2learn_available(),
        "torch_spec_present": torch_spec_available(),
        "industry_extra_present": metalearning_industry_available(),
    }


def _default_backend_when_installed() -> str:
    if metalearning_industry_available():
        return "industry"
    if metalearning_torch_available():
        return "torch"
    return "sklearn"


def _default_method_when_installed() -> str:
    if metalearning_industry_available():
        return "maml"
    if metalearning_torch_available():
        return "prototypical_torch"
    return "prototypical"


def list_metalearning_methods(
    *,
    backend: MetaLearningBackendName | None = None,
) -> list[str]:
    """List meta-learning method names available for one or all backends.

    Filters to backends that are actually installed when ``backend`` is omitted.

    Parameters
    ----------
    backend:
        Optional backend name; when set, returns methods only if that backend
        is available.

    Returns
    -------
    list[str]
        Sorted unique method identifiers (e.g. ``prototypical``, ``maml``).
    """
    matrix = metalearning_capability_matrix()
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None or not entry.get("available"):
            return []
        return list(entry.get("methods") or [])
    methods: list[str] = []
    for entry in matrix["backends"].values():
        if not entry.get("available"):
            continue
        for method in entry.get("methods") or []:
            if method not in methods:
                methods.append(method)
    return methods


def backend_available(name: MetaLearningBackendName) -> bool:
    """Return whether a meta-learning backend is installed and usable.

    Consults :func:`metalearning_capability_matrix` rather than probing imports
    directly so availability matches teaching disclosures.

    Parameters
    ----------
    name:
        Backend identifier: ``sklearn``, ``torch``, or ``industry``.

    Returns
    -------
    bool
        ``True`` when the capability matrix marks the backend as available.
    """
    entry = metalearning_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: MetaLearningBackendName | None,
    method: str,
) -> tuple[MetaLearningBackendName, str]:
    """Validate backend/method pairing and apply honest defaults.

    Normalizes method aliases, infers backend when omitted, and raises when the
    requested pair requires a missing extra.

    Parameters
    ----------
    backend:
        Optional backend override; when ``None``, inferred from ``method``.
    method:
        Meta-learning method name (case-insensitive, hyphens normalized).

    Returns
    -------
    tuple[MetaLearningBackendName, str]
        Resolved ``(backend, method)`` pair ready for fit routing.

    Raises
    ------
    ValidationError
        When the method is unknown or incompatible with the backend.
    MissingExtraError
        When the resolved backend requires an optional extra that is not installed.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    method_key = str(method).lower().replace("-", "_")
    all_methods = set(SKLEARN_METHODS) | set(TORCH_METHODS) | set(INDUSTRY_METHODS)
    if method_key not in all_methods:
        raise ValidationError(
            f"Unknown meta-learning method={method!r}. "
            f"Supported: {sorted(all_methods)}."
        )

    resolved_backend: MetaLearningBackendName
    if backend is None:
        if method_key in SKLEARN_METHODS:
            resolved_backend = "sklearn"
        elif method_key in TORCH_METHODS:
            resolved_backend = "torch"
        elif method_key in INDUSTRY_METHODS:
            resolved_backend = "industry"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
            allowed = list_metalearning_methods(backend=resolved_backend)
            if allowed:
                method_key = allowed[0]
            else:
                resolved_backend = "sklearn"
                method_key = "prototypical"
    else:
        resolved_backend = backend

    allowed = list_metalearning_methods(backend=resolved_backend)
    if method_key not in allowed:
        raise ValidationError(
            f"method='{method_key}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = metalearning_capability_matrix()["backends"][resolved_backend].get("extra")
        hint = str(extra or "torch")
        if resolved_backend == "industry":
            hint = "metalearning-industry,torch"
        raise MissingExtraError(hint, f"backend='{resolved_backend}'")
    return resolved_backend, method_key
