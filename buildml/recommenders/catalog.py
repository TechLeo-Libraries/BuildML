"""Recommender backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.core.industry_markers import platform_skip_entry
from buildml.recommenders.extras import (
    implicit_available,
    lightfm_available,
    recommenders_industry_available,
)
from buildml.recommenders.types import FeedbackMode, RecommenderMethod

RecommenderBackendName = Literal["sklearn", "implicit", "lightfm"]

_SKLEARN_METHODS = ("item_knn", "user_knn", "svd", "nmf", "content")
_IMPLICIT_METHODS = ("als", "bpr")
_LIGHTFM_METHODS = ("lightfm",)


def recommender_capability_matrix() -> dict[str, Any]:
    """Return an honest capability matrix for recommender backends.

    Summarises which sklearn, implicit, and LightFM methods are available,
    default routing per feedback mode, install hints, and non-goals. Used by
    walkthrough status and catalog helpers.

    Returns
    -------
    dict[str, Any]
        Nested backend availability, methods, ranking metrics, and install
        guidance suitable for Session disclosure.
    """
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": list(_SKLEARN_METHODS),
                "feedback": ["explicit", "implicit"],
                "features": "dense numpy/sklearn CF + content profiles",
            },
            "implicit": {
                "available": implicit_available(),
                "extra": "recommenders-industry",
                "methods": list(_IMPLICIT_METHODS),
                "feedback": ["implicit"],
                "features": "ALS / BPR on sparse implicit-feedback matrices",
                "notes": (
                    "Default backend for feedback='implicit' when the extra is "
                    "installed. Explicit ratings are not supported on this path."
                ),
            },
            "lightfm": {
                "available": lightfm_available(),
                "extra": "recommenders-lightfm",
                "methods": list(_LIGHTFM_METHODS),
                "feedback": ["explicit", "implicit"],
                "features": "Hybrid WARP/BPR with optional user/item side features",
                **platform_skip_entry("lightfm", extra="recommenders-lightfm"),
            },
        },
        "platform_markers": [
            platform_skip_entry("lightfm", extra="recommenders-lightfm"),
        ],
        "ranking_metrics": [
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "map_at_k",
        ],
        "cold_start_policies": ["popularity", "skip"],
        "default_method_for_feedback": {
            "explicit": _default_explicit_method(),
            "implicit": _default_implicit_method(),
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "recommenders-industry": (
                "pip install 'buildml[recommenders-industry]'  "
                "# implicit (ALS/BPR)"
            ),
            "recommenders-lightfm": (
                "pip install 'buildml[recommenders-lightfm]'  "
                "# LightFM hybrid (skipped on Win/Py3.13: no reliable wheels)"
            ),
        },
        "non_goals": [
            "Netflix-scale feature store / multi-stage cascade",
            "Streaming online recsys product",
            "Full surprise/recommenders library zoo",
            "RAG document retrieve/generate",
        ],
        "industry_extra_present": recommenders_industry_available(),
    }


def _default_explicit_method() -> str:
    return "item_knn"


def _default_implicit_method() -> str:
    if implicit_available():
        return "als"
    return "nmf"


def _default_backend_when_installed() -> str:
    if implicit_available():
        return "implicit"
    if lightfm_available():
        return "lightfm"
    return "sklearn"


def default_method_for_feedback(feedback: FeedbackMode) -> RecommenderMethod:
    """Return the default recommender method for a feedback mode.

    Explicit feedback defaults to item-item kNN; implicit feedback prefers ALS
    when the industry extra is installed, otherwise sklearn NMF.

    Parameters
    ----------
    feedback:
        ``"explicit"`` for rated interactions or ``"implicit"`` for presence-only
        signals.

    Returns
    -------
    RecommenderMethod
        Method name suitable for :func:`resolve_backend_method`.
    """
    if feedback == "implicit":
        return _default_implicit_method()  # type: ignore[return-value]
    return _default_explicit_method()  # type: ignore[return-value]


def list_recommender_methods(
    *,
    backend: RecommenderBackendName | None = None,
) -> list[str]:
    """List recommender method names, optionally filtered by backend.

    When ``backend`` is omitted, returns the union of methods across all
    registered backends without duplicates.

    Parameters
    ----------
    backend:
        Optional backend name (``"sklearn"``, ``"implicit"``, or ``"lightfm"``).
        When ``None``, all known methods are returned.

    Returns
    -------
    list[str]
        Method identifiers valid for the requested backend, or all methods.
    """
    matrix = recommender_capability_matrix()
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


def backend_available(name: RecommenderBackendName) -> bool:
    """Return whether a recommender backend is currently importable.

    Consults :func:`recommender_capability_matrix` so catalog probes stay
    consistent with walkthrough status and fit-time routing.

    Parameters
    ----------
    name:
        Backend identifier from the capability matrix.

    Returns
    -------
    bool
        ``True`` when the backend's optional dependencies are satisfied.
    """
    entry = recommender_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: RecommenderBackendName | None,
    method: RecommenderMethod | None,
    feedback: FeedbackMode,
) -> tuple[RecommenderBackendName, RecommenderMethod]:
    """Validate backend/method pairing and apply honest defaults.

    Fills in ``None`` backend or method from feedback-aware defaults, checks
    that the method is allowed for the backend, and verifies optional extras
    are installed before returning the resolved pair.

    Parameters
    ----------
    backend:
        Explicit backend override; inferred from ``method`` when ``None``.
    method:
        Recommender algorithm name; defaults via
        :func:`default_method_for_feedback` when ``None``.
    feedback:
        ``"explicit"`` or ``"implicit"``; constrains implicit-only backends.

    Returns
    -------
    tuple[RecommenderBackendName, RecommenderMethod]
        Resolved ``(backend, method)`` ready for :func:`fit_recommender`.

    Raises
    ------
    ValidationError
        When the method is invalid for the backend or implicit backend is
        paired with explicit feedback.
    MissingExtraError
        When the resolved backend requires an optional extra that is not
        installed.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_method: RecommenderMethod
    if method is None:
        resolved_method = default_method_for_feedback(feedback)
    else:
        resolved_method = method

    resolved_backend: RecommenderBackendName
    if backend is None:
        if resolved_method in _SKLEARN_METHODS:
            resolved_backend = "sklearn"
        elif resolved_method in _IMPLICIT_METHODS:
            resolved_backend = "implicit"
        elif resolved_method in _LIGHTFM_METHODS:
            resolved_backend = "lightfm"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
    else:
        resolved_backend = backend

    allowed = list_recommender_methods(backend=resolved_backend)
    if resolved_method not in allowed:
        raise ValidationError(
            f"method='{resolved_method}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )

    if resolved_backend == "implicit" and feedback != "implicit":
        raise ValidationError(
            "backend='implicit' (ALS/BPR) requires feedback='implicit'. "
            "Use sklearn svd/nmf/item_knn for explicit ratings, or lightfm for hybrid."
        )

    if not backend_available(resolved_backend):
        extra = recommender_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(
            str(extra or "recommenders-industry"),
            f"backend='{resolved_backend}' method='{resolved_method}'",
        )

    return resolved_backend, resolved_method
