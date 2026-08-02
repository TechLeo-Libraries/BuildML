"""Case memory objects and distance helpers for tabular CBR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError


@dataclass(slots=True)
class Case:
    """One retained episode: features + solution/label/outcome."""

    case_id: str
    row_index: Any
    solution: Any
    numeric_features: tuple[float, ...] = ()
    categorical_features: tuple[Any, ...] = ()
    source: str = "train"  # train | retained
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "row_index": self.row_index,
            "solution": self.solution,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "source": self.source,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class CaseTrace:
    """Explanation: which cases influenced a query prediction."""

    query_index: Any
    neighbor_case_ids: tuple[str, ...]
    neighbor_row_indices: tuple[Any, ...]
    distances: tuple[float, ...]
    weights: tuple[float, ...]
    neighbor_solutions: tuple[Any, ...]
    prediction: Any
    reuse_mode: str
    adapt_mode: str = "none"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "neighbor_case_ids": list(self.neighbor_case_ids),
            "neighbor_row_indices": list(self.neighbor_row_indices),
            "distances": list(self.distances),
            "weights": list(self.weights),
            "neighbor_solutions": list(self.neighbor_solutions),
            "prediction": self.prediction,
            "reuse_mode": self.reuse_mode,
            "adapt_mode": self.adapt_mode,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class CaseBase:
    """Train-built (or retained-augmented) case memory.

    Honesty: tabular case memory for supervised-style CBR — **not** a RAG
    text corpus, vector DB product, or full cognitive CBR research suite.
    """

    cases: tuple[Case, ...]
    numeric_matrix: np.ndarray = field(repr=False)
    categorical_matrix: np.ndarray = field(repr=False)  # object/int codes
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    metric: str
    # Train-fit distance transforms (never refit on holdout).
    numeric_mean_: np.ndarray | None = field(repr=False, default=None)
    numeric_scale_: np.ndarray | None = field(repr=False, default=None)
    numeric_ranges_: np.ndarray | None = field(repr=False, default=None)
    cat_vocabularies_: tuple[tuple[Any, ...], ...] = ()
    # Industry / embedding / torch retrieval artifacts (train-fit; never refit).
    search_matrix_: np.ndarray | None = field(repr=False, default=None)
    ann_index_: Any = field(repr=False, default=None)
    ann_library_: str | None = None
    text_embedder_id_: str | None = None
    torch_encoder_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    n_retained: int = 0

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "n_retained": self.n_retained,
            "metric": self.metric,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "disclosures": list(self.disclosures),
            "cases_preview": [c.to_dict() for c in self.cases[:5]],
        }


def pairwise_distances(
    query: np.ndarray,
    memory: np.ndarray,
    *,
    metric: str,
    query_cat: np.ndarray | None = None,
    memory_cat: np.ndarray | None = None,
    numeric_ranges: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute distances from one query (or batch) to every memory row.

    Parameters
    ----------
    query:
        Shape ``(n_query, n_num)`` or ``(n_num,)``.
    memory:
        Shape ``(n_cases, n_num)``.
    metric:
        ``euclidean`` | ``manhattan`` | ``cosine`` | ``mixed``.
    query_cat / memory_cat:
        Required for ``mixed`` — integer-coded categoricals.
    numeric_ranges:
        Per-feature ranges for Gower-style numeric term (``mixed``).
    """
    q = np.atleast_2d(np.asarray(query, dtype=float))
    m = np.asarray(memory, dtype=float)
    if q.shape[1] != m.shape[1]:
        raise ValidationError(
            f"Query numeric width {q.shape[1]} != case memory width {m.shape[1]}."
        )
    key = str(metric).lower().replace("-", "_")
    if key == "euclidean":
        # (n_q, n_m)
        diff = q[:, None, :] - m[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if key == "manhattan":
        diff = q[:, None, :] - m[None, :, :]
        return np.sum(np.abs(diff), axis=-1)
    if key == "cosine":
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        mn = np.linalg.norm(m, axis=1, keepdims=True)
        qn = np.maximum(qn, eps)
        mn = np.maximum(mn, eps)
        sim = (q / qn) @ (m / mn).T
        return 1.0 - sim
    if key == "mixed":
        if query_cat is None or memory_cat is None:
            raise ValidationError(
                "metric='mixed' requires categorical codes for query and memory."
            )
        qc = np.atleast_2d(np.asarray(query_cat))
        mc = np.asarray(memory_cat)
        n_num = m.shape[1]
        n_cat = mc.shape[1] if mc.ndim == 2 else 0
        n_parts = n_num + n_cat
        if n_parts == 0:
            raise ValidationError("mixed metric needs at least one feature.")
        # Numeric: range-normalized absolute difference.
        if n_num > 0:
            ranges = (
                np.asarray(numeric_ranges, dtype=float)
                if numeric_ranges is not None
                else np.ones(n_num, dtype=float)
            )
            ranges = np.maximum(ranges, eps)
            diff = np.abs(q[:, None, :] - m[None, :, :]) / ranges[None, None, :]
            num_term = np.mean(np.clip(diff, 0.0, 1.0), axis=-1)
        else:
            num_term = np.zeros((q.shape[0], m.shape[0]), dtype=float)
        if n_cat > 0:
            # Mismatch rate over categorical columns.
            mism = (qc[:, None, :] != mc[None, :, :]).astype(float)
            cat_term = np.mean(mism, axis=-1)
        else:
            cat_term = np.zeros_like(num_term)
        # Weighted average by feature count (Gower).
        w_num = n_num / n_parts
        w_cat = n_cat / n_parts
        return w_num * num_term + w_cat * cat_term
    raise ValidationError(
        f"Unknown CBR metric {metric!r}; expected euclidean, manhattan, "
        "cosine, or mixed."
    )


def top_k_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k smallest distances (stable for ties)."""
    if distances.ndim != 1:
        raise ValidationError("top_k_indices expects a 1-d distance vector.")
    n = int(distances.shape[0])
    if n == 0:
        raise ValidationError("Case memory is empty; cannot retrieve neighbors.")
    kk = min(int(k), n)
    if kk < 1:
        raise ValidationError("k must be >= 1.")
    # argpartition then sort the shortlist for stable ordering.
    part = np.argpartition(distances, kk - 1)[:kk]
    order = part[np.argsort(distances[part], kind="stable")]
    return order


def distance_weights(
    distances: Sequence[float] | np.ndarray, *, eps: float = 1e-8
) -> np.ndarray:
    """Inverse-distance weights; exact matches get weight 1/eps."""
    d = np.asarray(distances, dtype=float)
    return 1.0 / (d + float(eps))


def encode_categoricals(
    values: Sequence[Any] | np.ndarray,
    vocabulary: Sequence[Any],
) -> np.ndarray:
    """Map categorical values to integer codes; unknown → -1."""
    vocab = {str(v): i for i, v in enumerate(vocabulary)}
    out = np.empty(len(values), dtype=int)
    for i, v in enumerate(values):
        out[i] = vocab.get(str(v), -1)
    return out
