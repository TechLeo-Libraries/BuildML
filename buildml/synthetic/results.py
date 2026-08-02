"""Typed results for synthetic-data systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buildml.synthetic.types import ColumnSchemaSpec, SynthesizerConfig


@dataclass(slots=True)
class SynthesizerPlan:
    """Fitted train-only generator (schema + method state).

    Persist via ``buildml.synthetic_bundle.v1``. Distinct from Session
    checkpoints and from ``Session.resample`` (class-balance lineage only).
    """

    method: str
    partition_fitted: str
    columns: tuple[str, ...]
    column_specs: tuple[ColumnSchemaSpec, ...]
    n_rows_fitted: int
    random_state: int
    config: dict[str, Any] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    # Opaque fitted generator state (joblib-friendly)
    generator_: Any = field(default=None, repr=False)
    target_column: str | None = None
    roles_snapshot: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "partition_fitted": self.partition_fitted,
            "columns": list(self.columns),
            "column_specs": [spec.to_dict() for spec in self.column_specs],
            "n_rows_fitted": self.n_rows_fitted,
            "random_state": self.random_state,
            "config": dict(self.config),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "target_column": self.target_column,
            "roles_snapshot": dict(self.roles_snapshot),
            "has_generator": self.generator_ is not None,
        }


@dataclass(slots=True)
class SynthesizerFitResult:
    """Outcome of fitting a synthesizer on train."""

    method: str
    partition: str
    n_rows: int
    n_columns: int
    column_kinds: dict[str, str] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "partition": self.partition,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "column_kinds": dict(self.column_kinds),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }

    def show(self) -> None:
        print(
            f"SynthesizerFit · {self.method} · partition={self.partition} · "
            f"n={self.n_rows} · cols={self.n_columns}"
        )
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class SyntheticSampleResult:
    """Rows produced by a frozen synthesizer."""

    method: str
    n_rows: int
    columns: tuple[str, ...]
    frame: pd.DataFrame | None = field(default=None, repr=False)
    merged: bool = False
    merge_mode: str = "none"
    provenance_column: str | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "merged": self.merged,
            "merge_mode": self.merge_mode,
            "provenance_column": self.provenance_column,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "has_frame": self.frame is not None,
        }

    def show(self) -> None:
        print(
            f"SyntheticSample · {self.method} · n={self.n_rows} · "
            f"merged={self.merged} ({self.merge_mode})"
        )


@dataclass(slots=True)
class SyntheticEvalResult:
    """Utility / fidelity evaluation of a frozen synthesizer."""

    mode: str
    partition: str
    method: str
    n_real: int
    n_synthetic: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_column: dict[str, dict[str, float]] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "partition": self.partition,
            "method": self.method,
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "metrics": dict(self.metrics),
            "per_column": {k: dict(v) for k, v in self.per_column.items()},
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"SyntheticEval · mode={self.mode} · method={self.method} · "
            f"partition={self.partition}"
        )
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


# Re-export config for typing convenience
__all__ = [
    "SynthesizerPlan",
    "SynthesizerFitResult",
    "SyntheticSampleResult",
    "SyntheticEvalResult",
    "SynthesizerConfig",
]
