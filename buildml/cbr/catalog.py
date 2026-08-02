"""CBR catalog and honest capability matrix."""

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
    """Honest capability matrix for CBR retrieval backends."""
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
                    "Native brute-force case memory retrieval — always available "
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
                "citations — not case→solution reuse."
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
    return "euclidean"


def list_cbr_backends(*, available_only: bool = True) -> list[str]:
    matrix = cbr_capability_matrix()
    out: list[str] = []
    for name, entry in matrix["backends"].items():
        if available_only and not entry.get("available"):
            continue
        out.append(name)
    return out


def backend_available(name: CbrBackendName) -> bool:
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
    """Validate backend/metric pairing and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    metric_key = str(metric).lower().replace("-", "_")
    resolved_backend: CbrBackendName
    if backend is None:
        if text_columns:
            resolved_backend = "embedding"
        elif metric_key in {"manhattan", "mixed"}:
            resolved_backend = "sklearn"
        else:
            # Prefer industry/sklearn defaults — never probe torch here.
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
