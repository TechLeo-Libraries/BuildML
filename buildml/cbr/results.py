"""Typed results for case-based reasoning Session path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.cbr.cases import CaseBase, CaseTrace


@dataclass(slots=True)
class CbrPlan:
    """Train-built case memory + metric / reuse config.

    Persist via ``buildml.cbr_bundle.v1``. Honesty: tabular kNN / case-memory
    reasoning for supervised-style tasks — **not** RAG (document retrieval for
    generation), not a vector DB product, not a full cognitive CBR suite.
    """

    task: str
    metric: str
    reuse: str
    adapt: str
    k: int
    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    case_base: CaseBase
    classes_: tuple[Any, ...] | None
    label_encoder_: Any = field(repr=False, default=None)
    distance_eps: float = 1e-8
    standardize: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "cbr",
            "task": self.task,
            "metric": self.metric,
            "reuse": self.reuse,
            "adapt": self.adapt,
            "k": self.k,
            "columns": list(self.columns),
            "categorical_columns": list(self.categorical_columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_cases": self.case_base.n_cases,
            "n_retained": self.case_base.n_retained,
            "classes": None if self.classes_ is None else list(self.classes_),
            "distance_eps": self.distance_eps,
            "standardize": self.standardize,
            "case_base": self.case_base.to_dict(),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class CbrFitResult:
    """Outcome of building a case base on Session train."""

    task: str
    metric: str
    reuse: str
    k: int
    n_train_rows: int
    n_cases: int
    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target_column: str
    classes: tuple[Any, ...] | None = None
    train_score: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "metric": self.metric,
            "reuse": self.reuse,
            "k": self.k,
            "n_train_rows": self.n_train_rows,
            "n_cases": self.n_cases,
            "columns": list(self.columns),
            "categorical_columns": list(self.categorical_columns),
            "target_column": self.target_column,
            "classes": None if self.classes is None else list(self.classes),
            "train_score": self.train_score,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrEvalResult:
    """Holdout evaluation for CBR predictions."""

    partition: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    mean_neighbor_distance: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "mean_neighbor_distance": self.mean_neighbor_distance,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrPredictResult:
    """Predictions with optional case-influence explanation traces."""

    partition: str
    task: str
    n_rows: int
    predictions: tuple[Any, ...]
    traces: tuple[CaseTrace, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "n_traces": len(self.traces),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrRetrieveResult:
    """Neighbor retrieval without reuse (inspection / teaching surface)."""

    partition: str
    k: int
    metric: str
    n_queries: int
    traces: tuple[CaseTrace, ...]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "k": self.k,
            "metric": self.metric,
            "n_queries": self.n_queries,
            "n_traces": len(self.traces),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrRetainResult:
    """Outcome of retaining new labeled cases into the case base."""

    n_added: int
    n_cases_after: int
    n_skipped: int
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_added": self.n_added,
            "n_cases_after": self.n_cases_after,
            "n_skipped": self.n_skipped,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
