"""Report which CBR backends this installation has, and pair them with metrics.

Two jobs, both about being straight with the caller. The capability matrix says
what is actually available here, since retrieval behaviour depends on optional
dependencies and the same code searches differently on different machines. The
resolver pairs a requested backend with a requested metric and refuses
combinations that cannot be honoured.

That refusal is the important part. Approximate index libraries implement
Euclidean and cosine distance and nothing else: there is no Manhattan or
Gower-style mixed distance in an HNSW graph. Silently substituting a metric the
backend does support would change what "similar" means without telling anyone,
so an impossible pairing raises instead.

See Also
--------
buildml.cbr.extras : The availability probes underneath.
buildml.cbr.types.CbrConfig : Where backend and metric are chosen.
"""

from __future__ import annotations

from typing import Any, Literal

from buildml.cbr.extras import (
    cbr_industry_available,
    faiss_available,
    hnswlib_available,
    text_embedding_available,
)
from buildml.dl.extras import torch_available, torch_spec_available

CbrBackendName = Literal["sklearn", "industry", "embedding", "torch"]

SKLEARN_METRICS = ("euclidean", "manhattan", "cosine", "mixed")
INDUSTRY_METRICS = ("euclidean", "cosine")
EMBEDDING_METRICS = ("cosine", "euclidean")
TORCH_METRICS = ("euclidean", "cosine")


def cbr_capability_matrix() -> dict[str, Any]:
    """Describe the CBR retrieval capabilities available in this environment.

    Probes the optional dependencies and assembles a plain dictionary: which
    backends work here, which metrics each supports, what the defaults resolve
    to, how to install what is missing, and what this module deliberately does
    not attempt.

    The first thing to reach for when CBR behaves differently than expected, and
    worth logging alongside results so a run can be explained later.

    Returns
    -------
    dict
        Keys:

        ``backends``
            Per backend (``sklearn``, ``industry``, ``embedding``, ``torch``):
            availability, the extra that provides it, the metrics it supports,
            and a note on what it is for.
        ``ann_library``
            Which approximate library would be used, or ``None``.
        ``defaults``
            The backend and metric chosen when the caller specifies neither.
        ``install_hints``
            Copy-paste pip commands per extra.
        ``non_goals``
            What this is not trying to be.

    Notes
    -----
    **Availability comes from real imports where it is affordable**, so an
    installed-but-broken compiled library reports as unavailable: which matches
    what the user would experience.

    **The metric lists are narrower for the optional backends, and that is
    intrinsic.** Exact search can compute any distance you can write down;
    approximate indexes are built around a specific one.

    Examples
    --------
    Check before requesting a backend::

        matrix = cbr_capability_matrix()
        if not matrix["backends"]["industry"]["available"]:
            print(matrix["install_hints"]["cbr-industry"])
    """
    ann_lib = None
    if hnswlib_available():
        ann_lib = "hnswlib"
    elif faiss_available():
        ann_lib = "faiss"
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "metrics": list(SKLEARN_METRICS),
                "modality": "tabular",
                "retrieval": "exact kNN (numpy/sklearn distances)",
                "notes": (
                    "Native brute-force case memory retrieval: always available "
                    "fallback when industry extras are absent."
                ),
            },
            "industry": {
                "available": cbr_industry_available(),
                "extra": "cbr-industry",
                "metrics": list(INDUSTRY_METRICS),
                "modality": "tabular",
                "retrieval": f"approximate NN via {ann_lib or 'hnswlib|faiss'}",
                "ann_library": ann_lib,
                "notes": (
                    "Fast approximate retrieval on standardized numeric features "
                    "using hnswlib (preferred) or faiss-cpu when installed."
                ),
            },
            "embedding": {
                "available": text_embedding_available(),
                "extra": "rag or ssl",
                "metrics": list(EMBEDDING_METRICS),
                "modality": "text / hybrid",
                "retrieval": (
                    "sentence-transformer case embeddings + "
                    f"{'ANN' if cbr_industry_available() else 'exact cosine kNN'}"
                ),
                "notes": (
                    "Embed text (or hybrid text+numeric concat) case features with "
                    "sentence-transformers (buildml[rag] or buildml[ssl]). Uses ANN "
                    "when buildml[cbr-industry] is also installed."
                ),
            },
            "torch": {
                "available": torch_available(),
                "extra": "torch",
                "spec_present": torch_spec_available(),
                "import_probe": "available uses torch_available(); require_torch at fit time.",
                "metrics": list(TORCH_METRICS),
                "modality": "tabular",
                "retrieval": (
                    "learned metric encoder + kNN "
                    f"({'ANN' if cbr_industry_available() else 'exact'})"
                ),
                "notes": (
                    "Lite supervised metric encoder (MLP) on train cases; retrieve "
                    "neighbors in embedding space (buildml[torch])."
                ),
            },
        },
        "case_influence_traces": {
            "preserved": True,
            "fields": [
                "neighbor_case_ids",
                "neighbor_row_indices",
                "distances",
                "weights",
                "neighbor_solutions",
                "prediction",
            ],
        },
        "cbr_vs_rag": {
            "cbr": (
                "Tabular case memory (features + solution/label); retrieve similar "
                "cases and reuse/adapt outcomes for supervised-style prediction."
            ),
            "rag": (
                "Text corpus chunks; retrieve passages to ground LLM generation / "
                "citations: not case→solution reuse."
            ),
            "boundary": (
                "Sharing nearest-neighbor retrieval does not make CBR a RAG "
                "submodule. CBR bundles (buildml.cbr_bundle.v1) ≠ RAG bundles."
            ),
        },
        "evaluation": {
            "metrics": ["accuracy", "f1_macro", "rmse", "r2", "mean_neighbor_distance"],
            "holdout_rule": "train-built memory; holdout retrieve/predict/evaluate only",
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_metric_when_installed": _default_metric_when_installed(),
        "install_hints": {
            "cbr-industry": (
                "pip install 'buildml[cbr-industry]'  "
                "# hnswlib (preferred) or faiss-cpu approximate case retrieval"
            ),
            "rag": (
                "pip install 'buildml[rag]'  "
                "# sentence-transformer text case embedding backend"
            ),
            "ssl": (
                "pip install 'buildml[ssl]'  "
                "# alternate path for sentence-transformer text embeddings"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# learned metric encoder + kNN retrieval"
            ),
        },
        "non_goals": [
            "RAG document retrieval for generation",
            "Vector DB / Pinecone / Weaviate products",
            "Full cognitive CBR revise/retain research suites",
            "Cross-session federated case bases",
        ],
        "hnswlib_present": hnswlib_available(),
        "faiss_present": faiss_available(),
        "text_embedding_present": text_embedding_available(),
        "torch_spec_present": torch_spec_available(),
        "torch_import_honesty": (
            "torch backend matrix 'available' uses a real import probe "
            "(torch_available); torch_spec_present is find_spec only."
        ),
        "industry_extra_present": cbr_industry_available(),
    }


def _default_backend_when_installed() -> str:
    if cbr_industry_available():
        return "industry"
    return "sklearn"


def _default_metric_when_installed() -> str:
    """Return the metric used when the caller names none.

    Euclidean, unconditionally. It is supported by every backend, needs no
    categorical handling, and behaves predictably on standardised numeric
    features: so the default never constrains which backend can be chosen.

    Returns
    -------
    str
        Always ``'euclidean'``.
    """
    return "euclidean"


def list_cbr_backends(*, available_only: bool = True) -> list[str]:
    """List the retrieval backends, by default only those that work here.

    Filtering to what is actually installed is the useful default: a list that
    includes backends the environment cannot run turns a configuration choice
    into a failed import later on.

    Parameters
    ----------
    available_only:
        Restrict to backends whose dependencies are installed. Pass ``False``
        to see the full set, which is useful when telling a user what they
        could have.

    Returns
    -------
    list of str
        Backend names.

    Notes
    -----
    **``'sklearn'`` is always present**, so this never returns an empty list.

    See Also
    --------
    cbr_capability_matrix : The detail behind these names.
    """
    matrix = cbr_capability_matrix()
    out: list[str] = []
    for name, entry in matrix["backends"].items():
        if available_only and not entry.get("available"):
            continue
        out.append(name)
    return out


def backend_available(name: CbrBackendName) -> bool:
    """Report whether one named backend can be used here.

    The single-backend form of :func:`cbr_capability_matrix`, for branching on
    one capability without reading the whole picture.

    Parameters
    ----------
    name:
        ``'sklearn'``, ``'industry'``, ``'embedding'``, or ``'torch'``.

    Returns
    -------
    bool
        True when the backend is usable. An unrecognised name returns ``False``
        rather than raising, so a typo degrades to "unavailable".

    Notes
    -----
    **This builds the full matrix internally**, so it probes every dependency
    rather than only the one asked about. Fine for a setup check; avoid it in a
    loop.
    """
    entry = cbr_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_metric(
    *,
    backend: CbrBackendName | None,
    metric: str,
    text_columns: list[str] | None = None,
) -> tuple[CbrBackendName, str]:
    """Settle on a backend and metric, refusing pairings that cannot be honoured.

    Chooses a backend when none is named, then checks that the requested metric
    is one that backend can actually compute. An impossible pairing raises
    rather than substituting: quietly swapping in a metric the backend does
    support would change what "similar" means, and every downstream neighbour,
    prediction, and score would be answering a different question than the one
    asked.

    When no backend is given, the choice follows from the other settings. Text
    columns imply the embedding backend, since nothing else can compare text.
    Manhattan and mixed distances imply exact search, since no approximate index
    implements them. Otherwise the best installed option is used.

    Parameters
    ----------
    backend:
        The requested backend, or ``None`` to infer.
    metric:
        The requested distance function.
    text_columns:
        Text columns, whose presence forces the embedding backend.

    Returns
    -------
    tuple
        ``(backend, metric)``: both resolved and validated.

    Raises
    ------
    ValidationError
        If the metric is not one the resolved backend supports.
    MissingExtraError
        If a backend was named explicitly and its dependency is not installed.

    Notes
    -----
    **Torch is never probed while inferring a backend.** Importing torch is slow
    and, on some Windows configurations with antivirus in the path, can hang
    outright. It is only touched when explicitly requested.

    **Naming an unavailable backend raises; inferring one falls back.** An
    explicit request is a decision worth honouring or failing on, whereas
    inference is by definition a best-effort choice.

    See Also
    --------
    cbr_capability_matrix : Which pairings are possible here.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    metric_key = str(metric).lower().replace("-", "_")
    resolved_backend: CbrBackendName
    if backend is None:
        if text_columns:
            resolved_backend = "embedding"
        elif metric_key in {"manhattan", "mixed"}:
            resolved_backend = "sklearn"
        else:
            # Prefer industry/sklearn defaults: never probe torch here.
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
    else:
        resolved_backend = backend

    # Short-circuit non-torch backends so resolve never probes torch (Windows AV).
    if resolved_backend == "sklearn":
        if metric_key not in SKLEARN_METRICS:
            raise ValidationError(
                f"metric='{metric}' is not valid for backend='sklearn'. "
                f"Choose from {list(SKLEARN_METRICS)}."
            )
        return resolved_backend, metric_key
    if resolved_backend == "industry":
        from buildml.cbr.extras import cbr_industry_available

        if metric_key not in INDUSTRY_METRICS:
            raise ValidationError(
                f"metric='{metric}' is not valid for backend='industry'. "
                f"Choose from {list(INDUSTRY_METRICS)}."
            )
        if not cbr_industry_available():
            raise MissingExtraError("cbr-industry", "backend='industry'")
        return resolved_backend, metric_key
    if resolved_backend == "embedding":
        from buildml.cbr.extras import text_embedding_available

        if not text_columns:
            raise ValidationError(
                "backend='embedding' requires text_columns=... "
                "(one or more text feature columns)."
            )
        if metric_key not in EMBEDDING_METRICS:
            raise ValidationError(
                f"metric='{metric}' is not valid for backend='embedding'. "
                f"Choose from {list(EMBEDDING_METRICS)}."
            )
        if not text_embedding_available():
            raise MissingExtraError("rag or ssl", "backend='embedding'")
        return resolved_backend, metric_key

    matrix = cbr_capability_matrix()["backends"]
    entry = matrix.get(resolved_backend)
    if entry is None:
        raise ValidationError(f"Unknown CBR backend {resolved_backend!r}.")
    allowed = list(entry.get("metrics") or [])
    if metric_key not in allowed:
        raise ValidationError(
            f"metric='{metric}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed} or use backend='sklearn'."
        )
    if not entry.get("available"):
        extra = entry.get("extra") or "torch"
        raise MissingExtraError(str(extra), f"backend='{resolved_backend}'")
    return resolved_backend, metric_key
