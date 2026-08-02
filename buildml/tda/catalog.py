"""TDA backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.tda.extras import giotto_available, tda_available, tda_industry_available

TdaBackendName = Literal["native", "giotto"]
NativeVectorization = Literal["persistence_image", "landscape", "silhouette"]
GiottoVectorization = Literal[
    "persistence_image", "landscape", "betti_curve", "persistence_landscape"
]
VectorizationName = Literal[
    "persistence_image",
    "landscape",
    "silhouette",
    "betti_curve",
    "persistence_landscape",
]


def tda_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for TDA backends and optional extras."""
    return {
        "backends": {
            "native": {
                "available": tda_available(),
                "extra": "tda",
                "homology_engine": "ripser (Vietoris–Rips)",
                "vectorizations": [
                    "persistence_image",
                    "landscape",
                    "silhouette",
                ],
                "diagram_distances": ["wasserstein", "bottleneck"],
                "mapper": False,
                "notes": (
                    "Light ripser + persim stack; landscapes and silhouettes "
                    "vectorized in-tree. Default when buildml[tda] is installed."
                ),
            },
            "giotto": {
                "available": giotto_available(),
                "extra": "tda-industry",
                "homology_engine": "gtda.homology.VietorisRipsPersistence",
                "vectorizations": [
                    "persistence_image",
                    "persistence_landscape",
                    "landscape",
                    "betti_curve",
                ],
                "diagram_distances": ["wasserstein", "bottleneck"],
                "mapper": giotto_available(),
                "notes": (
                    "giotto-tda sklearn-style PH + BettiCurve / PersistenceImage / "
                    "PersistenceLandscape. Optional KeplerMapper summary on train "
                    "when mapper=True (buildml[tda-industry])."
                ),
            },
        },
        "subsample_strategies": ["error", "random", "stratified"],
        "evaluate_diagram_metrics": ["wasserstein", "bottleneck"],
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "tda": (
                "pip install 'buildml[tda]'  "
                "# ripser + persim — native VR PH + persistence images"
            ),
            "tda-industry": (
                "pip install 'buildml[tda-industry]'  "
                "# buildml[tda] + giotto-tda (Betti curves, gtda vectorizers, Mapper)"
            ),
        },
        "non_goals": [
            "Full Mapper research / interactive visualization suite",
            "Multiparameter, zigzag, or sheaf persistence",
            "Domain-specific credit-risk product surface",
            "Every TDA paper implementation",
        ],
        "tda_extra_present": tda_available(),
        "tda_industry_extra_present": tda_industry_available(),
    }


def _default_backend_when_installed() -> str:
    if giotto_available():
        return "giotto"
    if tda_available():
        return "native"
    return "native"


def list_tda_vectorizations(
    *,
    backend: TdaBackendName | None = None,
) -> list[str]:
    """List vectorization methods for a backend (or union when backend is None)."""
    matrix = tda_capability_matrix()
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None:
            return []
        return list(entry.get("vectorizations") or [])
    out: list[str] = []
    for entry in matrix["backends"].values():
        for name in entry.get("vectorizations") or []:
            if name not in out:
                out.append(name)
    return out


def backend_available(name: TdaBackendName) -> bool:
    entry = tda_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_vectorization(
    *,
    backend: TdaBackendName | None,
    vectorization: str,
) -> tuple[TdaBackendName, str]:
    """Validate backend/vectorization pairing and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    key = str(vectorization).lower().replace("-", "_")
    resolved_backend: TdaBackendName
    if backend is None:
        if key in {"silhouette"}:
            resolved_backend = "native"
        elif key in {"betti_curve", "persistence_landscape"}:
            resolved_backend = "giotto"
        elif key in {"persistence_image", "landscape"}:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
        else:
            resolved_backend = "native"
    else:
        resolved_backend = backend

    allowed = list_tda_vectorizations(backend=resolved_backend)
    if key not in allowed:
        raise ValidationError(
            f"vectorization='{key}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = tda_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "tda"), f"backend='{resolved_backend}'")
    return resolved_backend, key
