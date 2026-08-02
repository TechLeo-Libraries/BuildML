"""Configuration types for Session-facing case-based reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CbrTask = Literal["classification", "regression"]

# Retrieval backend for case memory (honest defaults when extras installed).
CbrBackend = Literal["sklearn", "industry", "embedding", "torch"]

# Distance / similarity metrics over the case-memory feature space.
# - euclidean / manhattan / cosine: numeric features only (after optional train scaling)
# - mixed: Gower-style mix of range-normalized numeric |Δ| and categorical mismatch
CbrMetric = Literal["euclidean", "manhattan", "cosine", "mixed"]

# How neighbor solutions become a prediction.
CbrReuseMode = Literal[
    "majority",  # classification: unweighted majority vote
    "distance_weighted",  # classification vote / regression average with 1/(d+ε)
    "local_mean",  # regression: unweighted mean of neighbor solutions
    "local_ridge",  # regression: tiny Ridge on the k neighbors' features→solution
]

# Optional post-reuse adaptation (lite).
CbrAdaptMode = Literal["none", "offset"]


@dataclass(slots=True)
class CbrConfig:
    """User-facing CBR knobs (serializable summary)."""

    task: CbrTask = "classification"
    backend: CbrBackend = "sklearn"
    metric: CbrMetric = "euclidean"
    reuse: CbrReuseMode = "distance_weighted"
    adapt: CbrAdaptMode = "none"
    k: int = 5
    columns: tuple[str, ...] | None = None
    categorical_columns: tuple[str, ...] | None = None
    text_columns: tuple[str, ...] | None = None
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    standardize: bool = True
    distance_eps: float = 1e-8
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    disclosures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "backend": self.backend,
            "metric": self.metric,
            "reuse": self.reuse,
            "adapt": self.adapt,
            "k": self.k,
            "columns": None if self.columns is None else list(self.columns),
            "categorical_columns": (
                None
                if self.categorical_columns is None
                else list(self.categorical_columns)
            ),
            "text_columns": (
                None if self.text_columns is None else list(self.text_columns)
            ),
            "text_model_name": self.text_model_name,
            "standardize": self.standardize,
            "distance_eps": self.distance_eps,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "disclosures": list(self.disclosures),
        }
