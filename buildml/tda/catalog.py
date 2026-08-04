"""TDA backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.core.industry_markers import platform_skip_entry
from buildml.tda.extras import (
    giotto_available,
    giotto_spec_present,
    persim_spec_present,
    ripser_spec_present,
    tda_available,
    tda_industry_available,
)

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
    """Report which TDA backends and vectorizations are available on this machine.

    Call before :func:`fit_tda` or Session :meth:`~buildml.session.session.Session.fit_tda`
    to confirm ripser/persim (``buildml[tda]``) or giotto-tda (``buildml[tda-industry]``)
    imported successfully. Read-only introspection: no point cloud required.

    Returns
    -------
    dict[str, Any]
        Nested ``backends`` for ``native`` and ``giotto``, default backend selection,
        subsample strategies, diagram distance metrics, ``install_hints``, and
        ``non_goals`` separating Session TDA from Mapper research tools.
    """
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
                **platform_skip_entry("giotto_tda", extra="tda-industry"),
            },
        },
        "platform_markers": [
            platform_skip_entry("giotto_tda", extra="tda-industry"),
        ],
        "subsample_strategies": ["error", "random", "stratified"],
        "evaluate_diagram_metrics": ["wasserstein", "bottleneck"],
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "tda": (
                "pip install 'buildml[tda]'  "
                "# ripser + persim: native VR PH + persistence images"
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
        "tda_extra_present": ripser_spec_present() and persim_spec_present(),
        "tda_runtime_present": tda_available(),
        "tda_industry_extra_present": giotto_spec_present(),
        "tda_industry_runtime_present": tda_industry_available(),
        "industry_import_honesty": (
            "tda_*_runtime_present and backend 'available' flags use subprocess "
            "import probes. tda_*_extra_present / *_spec_present are find_spec only."
        ),
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
    """List persistence-diagram vectorization methods for a TDA backend.

    When ``backend`` is ``None``, returns the union of vectorizations advertised
    by every backend in :func:`tda_capability_matrix`.

    Parameters
    ----------
    backend:
        ``native`` or ``giotto``. When ``None``, aggregate across backends.

    Returns
    -------
    list[str]
        Vectorization names such as ``persistence_image`` or ``betti_curve``.
    """
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
    """Return whether a TDA backend imported successfully on this machine.

    Looks up the ``available`` flag for ``native`` or ``giotto`` in
    :func:`tda_capability_matrix`. Use before hard-coding a backend in shared
    code.

    Parameters
    ----------
    name:
        ``native`` (ripser/persim) or ``giotto`` (giotto-tda).

    Returns
    -------
    bool
        ``True`` when the backend's optional extra imported successfully.
    """
    entry = tda_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_vectorization(
    *,
    backend: TdaBackendName | None,
    vectorization: str,
) -> tuple[TdaBackendName, str]:
    """Validate a backend/vectorization pair and apply honest defaults.

    Normalises hyphenated names, picks a backend when ``None``, and refuses
    pairings that are not advertised in :func:`tda_capability_matrix`.

    Parameters
    ----------
    backend:
        Explicit ``native`` or ``giotto``. When ``None``, inferred from the
        vectorization (e.g. ``silhouette`` → native, ``betti_curve`` → giotto).
    vectorization:
        Persistence diagram vectorizer name.

    Returns
    -------
    tuple[TdaBackendName, str]
        Resolved backend and normalised vectorization key.

    Raises
    ------
    ValidationError
        When the vectorization is unknown for the resolved backend.
    MissingExtraError
        When the resolved backend's optional extra is not installed.
    """
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
