"""Configuration types for Session-facing decision / optimisation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DecisionMethod = Literal[
    "threshold",
    "cost_matrix",
    "topk",
    "knapsack",
    "lp_allocate",
]
DecisionBackend = Literal["native", "pulp", "ortools", "cvxpy", "calibrated", "xgb"]
TuningPartition = Literal["train", "validation", "test"]
ScoreSource = Literal["model_proba", "model_decision_function", "column"]
KnapsackSolver = Literal["dp", "greedy"]
AllocationObjective = Literal["maximize_score", "maximize_value", "minimize_cost"]


@dataclass(slots=True)
class DecisionConfig:
    """User-facing decision-policy knobs (serializable summary)."""

    method: DecisionMethod = "threshold"
    backend: DecisionBackend | None = None
    partition: TuningPartition = "validation"
    allow_test_tuning: bool = False
    # Binary threshold / expected-cost (wraps classical threshold sweep)
    fp_cost: float | None = None
    fn_cost: float | None = None
    tp_benefit: float = 0.0
    tn_benefit: float = 0.0
    # Multi-class Bayes decision under a cost matrix C[true, action]
    cost_matrix: list[list[float]] | None = None
    class_labels: list[str] | None = None
    # Allocation helpers
    capacity: int | None = None
    budget: float | None = None
    score_source: ScoreSource = "model_proba"
    score_column: str | None = None
    cost_column: str | None = None
    value_column: str | None = None
    id_column: str | None = None
    knapsack_solver: KnapsackSolver = "dp"
    objective: AllocationObjective = "maximize_score"
    min_score: float | None = None
    # LP continuous allocation (fractional knapsack / budget share)
    lp_max_fraction: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": self.backend,
            "partition": self.partition,
            "allow_test_tuning": self.allow_test_tuning,
            "fp_cost": self.fp_cost,
            "fn_cost": self.fn_cost,
            "tp_benefit": self.tp_benefit,
            "tn_benefit": self.tn_benefit,
            "cost_matrix": None if self.cost_matrix is None else [list(r) for r in self.cost_matrix],
            "class_labels": None if self.class_labels is None else list(self.class_labels),
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
        }


@dataclass(slots=True)
class CostModel:
    """Serializable cost / benefit specification used by a DecisionPlan."""

    kind: Literal["binary_expected_cost", "multiclass_matrix", "selection_unit_cost"]
    fp_cost: float | None = None
    fn_cost: float | None = None
    tp_benefit: float = 0.0
    tn_benefit: float = 0.0
    matrix: list[list[float]] | None = None
    class_labels: tuple[str, ...] = ()
    formula: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fp_cost": self.fp_cost,
            "fn_cost": self.fn_cost,
            "tp_benefit": self.tp_benefit,
            "tn_benefit": self.tn_benefit,
            "matrix": None if self.matrix is None else [list(r) for r in self.matrix],
            "class_labels": list(self.class_labels),
            "formula": self.formula,
            "extras": dict(self.extras),
        }
