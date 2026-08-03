"""Knowledge-graph backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.kg.extras import kg_industry_available, pykeen_available

KgBackendName = Literal["native", "pykeen"]


def kg_capability_matrix() -> dict[str, Any]:
    """Build the honest capability matrix for knowledge-graph backends.

    Reports native vs PyKEEN methods, evaluation metrics, query modes, install
    hints, and explicit non-goals for teaching overlays and Session walkthroughs.

    Returns
    -------
    dict[str, Any]
        Nested backend entries, link-prediction modes, and boundary disclosures.
    """
    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "methods": ["transe", "distmult"],
                "embedding_engine": "pure-numpy SGD",
                "negative_sampling": "uniform head/tail corruption (disclosed neg_ratio)",
                "symbolic_query": True,
                "filtered_mrr_hits": True,
            },
            "pykeen": {
                "available": pykeen_available(),
                "extra": "kg-industry",
                "methods": ["transe", "distmult", "rotate", "complex"],
                "embedding_engine": "PyKEEN pipeline (torch)",
                "negative_sampling": "PyKEEN sLCWA / LCWA (disclosed on fit)",
                "symbolic_query": True,
                "filtered_mrr_hits": True,
                "notes": (
                    "RotatE and ComplEx require buildml[kg-industry]. Train-only "
                    "triples materialized from Session train partition; holdout "
                    "never enters PyKEEN training factory."
                ),
            },
        },
        "evaluation_metrics": [
            "mrr",
            "hits_at_1",
            "hits_at_3",
            "hits_at_k",
            "mean_rank",
        ],
        "link_prediction_modes": ["tail", "head", "relation"],
        "query_modes": ["neighbors", "path", "typed"],
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "kg-industry": (
                "pip install 'buildml[kg-industry]'  "
                "# PyKEEN RotatE / ComplEx / TransE / DistMult pipeline"
            ),
        },
        "non_goals": [
            "Neo4j / Cypher graph-database product",
            "Graph ML node classification (set_graph / fit_graph)",
            "RAG retrieve/generate",
            "Full PyG graph learning product scope",
            "Automatic ontology / schema inference",
        ],
        "industry_extra_present": kg_industry_available(),
        "pykeen_import_honesty": (
            "pykeen backend 'available' reflects package install (find_spec). "
            "PyKEEN training also requires a working torch install — broken "
            "wheels may fail at require_pykeen / require_torch."
        ),
        "train_only_honesty": (
            "All backends fit on Session train triples only. Vocabularies, "
            "embeddings, and symbolic adjacency exclude holdout triples."
        ),
    }


def _default_backend_when_installed() -> str:
    if pykeen_available():
        return "pykeen"
    return "native"


def list_kg_methods(*, backend: KgBackendName | None = None) -> list[str]:
    """List embedding methods for a knowledge-graph backend.

    Reads :func:`kg_capability_matrix` so callers only offer methods installed
    for the requested backend.

    Parameters
    ----------
    backend:
        ``native``, ``pykeen``, or ``None`` for the combined deduplicated list.

    Returns
    -------
    list[str]
        Valid method names such as ``transe``, ``distmult``, ``rotate``.
    """
    matrix = kg_capability_matrix()["backends"]
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


def backend_available(name: KgBackendName) -> bool:
    """Return whether a knowledge-graph backend is available on this machine.

    Checks the ``available`` flag in :func:`kg_capability_matrix` for native
    or pykeen entries.

    Parameters
    ----------
    name:
        Backend key such as ``native`` or ``pykeen``.

    Returns
    -------
    bool
        ``True`` when the backend can be used for fit without missing extras.
    """
    matrix = kg_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: KgBackendName | None,
    method: str,
) -> tuple[KgBackendName, str]:
    """Validate backend/method pairing and apply honest defaults.

    Infers backend from method when omitted, checks install status, and
    normalises method aliases before fit proceeds.

    Parameters
    ----------
    backend:
        Explicit backend or ``None`` to infer from ``method``.
    method:
        Embedding method key from the catalog.

    Returns
    -------
    tuple[str, str]
        Resolved ``(backend, method_key)`` pair.

    Raises
    ------
    ValidationError
        When the method is not valid for the resolved backend.
    MissingExtraError
        When the resolved backend requires ``kg-industry`` and it is missing.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    method_key = str(method).lower().replace("-", "_")
    native = {"transe", "distmult"}
    pykeen_only = {"rotate", "complex"}
    pykeen_all = native | pykeen_only

    resolved_backend: KgBackendName
    if backend is None:
        if method_key in pykeen_only:
            resolved_backend = "pykeen"
        elif method_key in native:
            resolved_backend = "native"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
    else:
        resolved_backend = backend

    allowed = list_kg_methods(backend=resolved_backend)
    if method_key not in allowed:
        if backend is None and method_key in native:
            resolved_backend = "native"
            allowed = list_kg_methods(backend="native")
        if method_key not in allowed:
            raise ValidationError(
                f"method='{method}' is not valid for backend='{resolved_backend}'. "
                f"Choose from {allowed}."
            )
    if not backend_available(resolved_backend):
        extra = kg_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(
            str(extra or "kg-industry"),
            f"backend='{resolved_backend}'",
        )
    return resolved_backend, method_key
