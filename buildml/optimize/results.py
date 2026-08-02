"""Typed results for decision / optimisation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.optimize.types import CostModel


@dataclass(slots=True)
class DecisionPlan:
    """Fitted decision policy (thresholds, cost matrix, or allocation rules).

    Persist via ``buildml.decision_bundle.v1``. Distinct from Session
    checkpoints and from classical ``DiagnosticReport`` threshold sweeps —
    this plan is the reusable operating policy; ``tune_threshold`` remains
    the diagnostic explorer.
    """

    method: str
    partition_fitted: str
    allow_test_tuning: bool
    # Threshold policy
    threshold: float | None = None
    positive_class: str | None = None
    recommendation_basis: str | None = None
    # Cost specification
    cost_model: CostModel | None = None
    class_labels: tuple[str, ...] = ()
    # Allocation
    capacity: int | None = None
    budget: float | None = None
    score_source: str = "model_proba"
    score_column: str | None = None
    cost_column: str | None = None
    value_column: str | None = None
    id_column: str | None = None
    knapsack_solver: str | None = None
    objective: str = "maximize_score"
    min_score: float | None = None
    lp_max_fraction: float = 1.0
    # Fit-partition diagnostics (not reapplied blindly — stored for audit)
    n_rows_fitted: int = 0
    expected_cost_at_fit: float | None = None
    selected_value_at_fit: float | None = None
    selected_cost_at_fit: float | None = None
    n_selected_at_fit: int = 0
    operating_points: dict[str, Any] = field(default_factory=dict)
    # Feature columns expected when scoring via the Session estimator
    feature_columns: tuple[str, ...] = ()
    target_column: str | None = None
    task: str | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    # Optional action index map for multiclass (label -> column of cost matrix)
    action_labels: tuple[str, ...] = ()
    # Stored for LP / knapsack unit scaling disclosure
    cost_scale_: float = 1.0
    # Bayes action lookup is recomputed from cost_matrix + proba at apply time
    cost_matrix_: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "partition_fitted": self.partition_fitted,
            "allow_test_tuning": self.allow_test_tuning,
            "threshold": self.threshold,
            "positive_class": self.positive_class,
            "recommendation_basis": self.recommendation_basis,
            "cost_model": None if self.cost_model is None else self.cost_model.to_dict(),
            "class_labels": list(self.class_labels),
            "action_labels": list(self.action_labels),
            "capacity": self.capacity,
            "budget": self.budget,
            "score_source": self.score_source,
            "score_column": self.score_column,
            "cost_column": self.cost_column,
            "value_column": self.value_column,
            "id_column": self.id_column,
            "knapsack_solver": self.knapsack_solver,
            "objective": self.objective,
            "min_score": self.min_score,
            "lp_max_fraction": self.lp_max_fraction,
            "n_rows_fitted": self.n_rows_fitted,
            "expected_cost_at_fit": self.expected_cost_at_fit,
            "selected_value_at_fit": self.selected_value_at_fit,
            "selected_cost_at_fit": self.selected_cost_at_fit,
            "n_selected_at_fit": self.n_selected_at_fit,
            "operating_points": dict(self.operating_points),
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "task": self.task,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
            "cost_scale": self.cost_scale_,
        }


@dataclass(slots=True)
class DecisionFitResult:
    """Outcome of fitting a decision policy on a tuning partition."""

    method: str
    partition: str
    n_rows: int
    threshold: float | None = None
    recommendation_basis: str | None = None
    expected_cost: float | None = None
    n_selected: int = 0
    selected_value: float | None = None
    selected_cost: float | None = None
    capacity: int | None = None
    budget: float | None = None
    allow_test_tuning: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "partition": self.partition,
            "n_rows": self.n_rows,
            "threshold": self.threshold,
            "recommendation_basis": self.recommendation_basis,
            "expected_cost": self.expected_cost,
            "n_selected": self.n_selected,
            "selected_value": self.selected_value,
            "selected_cost": self.selected_cost,
            "capacity": self.capacity,
            "budget": self.budget,
            "allow_test_tuning": self.allow_test_tuning,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }

    def show(self) -> None:
        print(
            f"DecisionFit · {self.method} · partition={self.partition} · "
            f"n={self.n_rows}"
        )
        if self.threshold is not None:
            print(f"  threshold={self.threshold:.4f} ({self.recommendation_basis})")
        if self.expected_cost is not None:
            print(f"  expected_cost={self.expected_cost:.6f}")
        if self.n_selected:
            print(
                f"  selected={self.n_selected} · value={self.selected_value} · "
                f"cost={self.selected_cost}"
            )
        for tip in self.disclosures[:6]:
            print(f"  · {tip}")


@dataclass(slots=True)
class ApplyDecisionsResult:
    """Decisions produced by applying a frozen DecisionPlan."""

    method: str
    partition: str | None
    n_rows: int
    n_selected: int
    # For threshold / cost_matrix: predicted labels / actions
    decisions: tuple[Any, ...] = ()
    scores: tuple[float, ...] = ()
    # For allocation: selected ids / indices and fractions (LP)
    selected_ids: tuple[Any, ...] = ()
    selected_indices: tuple[int, ...] = ()
    fractions: tuple[float, ...] = ()
    selected_value: float | None = None
    selected_cost: float | None = None
    threshold: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "partition": self.partition,
            "n_rows": self.n_rows,
            "n_selected": self.n_selected,
            "n_decisions": len(self.decisions),
            "threshold": self.threshold,
            "selected_value": self.selected_value,
            "selected_cost": self.selected_cost,
            "n_selected_ids": len(self.selected_ids),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"ApplyDecisions · {self.method} · partition={self.partition} · "
            f"n={self.n_rows} · selected={self.n_selected}"
        )


@dataclass(slots=True)
class DecisionEvalResult:
    """Holdout evaluation of a frozen decision policy."""

    partition: str
    method: str
    n_rows: int
    metrics: dict[str, float] = field(default_factory=dict)
    realized_cost: float | None = None
    n_selected: int = 0
    selected_value: float | None = None
    selected_cost: float | None = None
    threshold: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "method": self.method,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "realized_cost": self.realized_cost,
            "n_selected": self.n_selected,
            "selected_value": self.selected_value,
            "selected_cost": self.selected_cost,
            "threshold": self.threshold,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        print(
            f"DecisionEval · {self.method} · partition={self.partition} · "
            f"n={self.n_rows}"
        )
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
