"""Online / continual catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_available, torch_spec_available
from buildml.online.extras import (
    online_industry_available,
    river_available,
    river_spec_present,
)

OnlineBackendName = Literal["sklearn", "industry", "torch"]

SKLEARN_ESTIMATORS = (
    "sgd_classifier",
    "sgd_regressor",
    "passive_aggressive_classifier",
    "passive_aggressive_regressor",
    "perceptron",
    "multinomial_nb",
    "bernoulli_nb",
)
INDUSTRY_ESTIMATORS = (
    "river_logistic",
    "river_hoeffding",
    "river_pa",
    "river_linear_regression",
    "river_hoeffding_regressor",
)
TORCH_ESTIMATORS = (
    "replay_mlp",
    "ewc_mlp",
)
DRIFT_DETECTORS = (
    "mean_shift",
    "adwin",
    "page_hinkley",
    "none",
)


def online_capability_matrix() -> dict[str, Any]:
    """Build the honest capability matrix for online / continual backends.

    Reports sklearn, industry (River), and torch paths, chunk ingestion modes,
    evaluation metrics, install hints, and explicit non-goals for teaching
    overlays and Session walkthroughs.

    Returns
    -------
    dict[str, Any]
        Nested backend entries, chunk ingestion rules, evaluation metrics, and defaults.
    """
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "estimators": list(SKLEARN_ESTIMATORS),
                "modality": "tabular",
                "drift_detectors": ["mean_shift"],
                "notes": (
                    "Core sklearn partial_fit family (SGD, PA, Perceptron, NB). "
                    "Optional init-chunk mean-shift disclosure on updates."
                ),
            },
            "industry": {
                "available": online_industry_available(),
                "extra": "online-industry",
                "estimators": list(INDUSTRY_ESTIMATORS),
                "modality": "tabular",
                "drift_detectors": ["mean_shift", "adwin", "page_hinkley"],
                "notes": (
                    "River streaming sklearn-compatible models with ADWIN / "
                    "Page-Hinkley drift hooks on updates and evaluate "
                    "(buildml[online-industry])."
                ),
            },
            "torch": {
                "available": torch_available(),
                "extra": "torch",
                "estimators": list(TORCH_ESTIMATORS),
                "modality": "tabular",
                "drift_detectors": ["mean_shift"],
                "notes": (
                    "Lite replay-buffer or EWC tabular MLP continual learner: "
                    "honest small-scale, not a lifelong-learning research suite "
                    "(buildml[torch])."
                ),
            },
        },
        "chunk_ingestion": {
            "session_train_cursor": "Default: carve unused train rows after cursor",
            "explicit_indices": "Train-partition dataset indices only",
            "external_frame": "Role-aligned user frame; cursor not advanced",
        },
        "evaluation": {
            "metrics_classification": ["accuracy", "f1_macro", "f1_weighted"],
            "metrics_regression": ["mae", "rmse", "r2"],
            "holdout_rule": "validation/test never used for partial_fit updates",
            "drift_hooks": (
                "River ADWIN/Page-Hinkley on holdout error stream when "
                "backend=industry; mean_shift disclosure on all backends."
            ),
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_estimator_when_installed": _default_estimator_when_installed(),
        "install_hints": {
            "online-industry": (
                "pip install 'buildml[online-industry]'  "
                "# River streaming models + ADWIN/Page-Hinkley drift"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# replay / EWC tabular continual MLP"
            ),
        },
        "non_goals": [
            "Distributed streaming platforms",
            "Full lifelong-learning research suites",
            "Silent full refits (allow_refit_fallback must disclose)",
            "Using holdout rows for incremental updates",
        ],
        "river_present": river_available(),
        "torch_spec_present": torch_spec_available(),
        "industry_extra_present": river_spec_present(),
        "industry_runtime_present": online_industry_available(),
        "industry_import_honesty": (
            "industry backend 'available' and industry_runtime_present require "
            "a successful subprocess import of river. industry_extra_present / "
            "river_spec_present are find_spec only."
        ),
    }


def _default_backend_when_installed() -> str:
    if online_industry_available():
        return "industry"
    if torch_available():
        return "torch"
    return "sklearn"


def _default_estimator_when_installed() -> str:
    if online_industry_available():
        return "river_logistic"
    if torch_available():
        return "replay_mlp"
    return "sgd_classifier"


def list_online_estimators(
    *,
    backend: OnlineBackendName | None = None,
) -> list[str]:
    """List online estimators for one backend or all installed backends.

    Filters to backends marked available in :func:`online_capability_matrix`.

    Parameters
    ----------
    backend:
        Optional backend name (``sklearn``, ``industry``, ``torch``). When
        ``None``, returns the union of estimators from all available backends.

    Returns
    -------
    list[str]
        Estimator keys valid for the requested backend(s).
    """
    matrix = online_capability_matrix()
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None:
            return []
        if not entry.get("available"):
            return []
        return list(entry.get("estimators") or [])
    estimators: list[str] = []
    for entry in matrix["backends"].values():
        if not entry.get("available"):
            continue
        for name in entry.get("estimators") or []:
            if name not in estimators:
                estimators.append(name)
    return estimators


def backend_available(name: OnlineBackendName) -> bool:
    """Return whether an online backend is installed and usable.

    Consults :func:`online_capability_matrix` rather than probing imports
    directly, so availability matches catalog disclosure.

    Parameters
    ----------
    name:
        Backend key (``sklearn``, ``industry``, or ``torch``).

    Returns
    -------
    bool
        ``True`` when the capability matrix marks the backend as available.
    """
    matrix = online_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_estimator(
    *,
    backend: OnlineBackendName | None,
    estimator: str,
) -> tuple[OnlineBackendName, str]:
    """Validate backend/estimator pairing and apply honest defaults.

    Infers the backend from the estimator name when ``backend`` is ``None`` and
    refuses pairings that require missing extras.

    Parameters
    ----------
    backend:
        Optional explicit backend; inferred from ``estimator`` when ``None``.
    estimator:
        Estimator key (normalized to lowercase with hyphens as underscores).

    Returns
    -------
    tuple[OnlineBackendName, str]
        Resolved ``(backend, estimator)`` pair.

    Raises
    ------
    ValidationError
        When the estimator is unknown or invalid for the resolved backend.
    MissingExtraError
        When the resolved backend requires an optional extra that is not installed.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    est_key = str(estimator).lower().replace("-", "_")
    resolved_backend: OnlineBackendName
    if backend is None:
        if est_key in SKLEARN_ESTIMATORS:
            resolved_backend = "sklearn"
        elif est_key in INDUSTRY_ESTIMATORS:
            resolved_backend = "industry"
        elif est_key in TORCH_ESTIMATORS:
            resolved_backend = "torch"
        else:
            raise ValidationError(
                f"Unknown online estimator={est_key!r}. "
                "See online_capability_matrix() for valid names."
            )
    else:
        resolved_backend = backend

    if not backend_available(resolved_backend):
        extra = online_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "online-industry"), f"backend='{resolved_backend}'")

    allowed = list_online_estimators(backend=resolved_backend)
    if est_key not in allowed:
        raise ValidationError(
            f"estimator='{est_key}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    return resolved_backend, est_key


def resolve_drift_detector(
    *,
    backend: OnlineBackendName,
    drift_detector: str | None,
) -> str:
    """Pick a drift disclosure mode valid for the backend.

    Defaults to ``adwin`` on industry backends when River is available, otherwise
    ``mean_shift``. River-specific detectors require ``backend='industry'``.

    Parameters
    ----------
    backend:
        Online backend in use (``sklearn``, ``industry``, or ``torch``).
    drift_detector:
        Optional explicit detector key; ``None`` selects a backend-appropriate default.

    Returns
    -------
    str
        Normalized drift detector key from :data:`DRIFT_DETECTORS`.

    Raises
    ------
    ValidationError
        When the detector is unknown or incompatible with the backend.
    MissingExtraError
        When River drift detectors are requested but ``online-industry`` is missing.
    """
    from buildml.core.errors import ValidationError

    if drift_detector is None:
        if backend == "industry" and online_industry_available():
            return "adwin"
        return "mean_shift"
    key = str(drift_detector).lower().replace("-", "_")
    if key not in DRIFT_DETECTORS:
        raise ValidationError(
            f"drift_detector='{drift_detector}' is unknown. Choose from {DRIFT_DETECTORS}."
        )
    if key in {"adwin", "page_hinkley"} and backend != "industry":
        raise ValidationError(
            f"drift_detector='{key}' requires backend='industry' (River). "
            "Use mean_shift on sklearn/torch or install buildml[online-industry]."
        )
    if key in {"adwin", "page_hinkley"} and not online_industry_available():
        from buildml.core.errors import MissingExtraError

        raise MissingExtraError(
            "online-industry",
            f"River drift detector '{key}'",
        )
    return key
