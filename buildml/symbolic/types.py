"""Configuration types for Session-facing symbolic / neuro-symbolic ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SymbolicTask = Literal["classification", "regression"]

SymbolicSource = Literal["declared", "decision_tree", "decision_list"]

SymbolicBackend = Literal["sklearn", "industry"]

IndustrySymbolicMethod = Literal["skope_rules", "rulefit", "boosted_rules"]

NeuroSymbolicBackend = Literal["sklearn", "torch"]

TorchNeuroMethod = Literal["concept_bottleneck_lite", "neural_additive_lite"]

NeuroSymbolicMode = Literal[
    "constraint_overlay",
    "rules_as_features",
    "constraint_repair",
]

BaseEstimatorName = Literal[
    "logistic_regression",
    "ridge",
    "random_forest",
    "decision_tree",
    "concept_bottleneck_lite",
    "neural_additive_lite",
]

PredicateOp = Literal[
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
    "in",
    "not_in",
    "isna",
    "notna",
]

RuleSource = Literal[
    "declared",
    "induced_tree",
    "induced_list",
    "induced_skope",
    "induced_rulefit",
    "induced_boosted_rules",
]


@dataclass(slots=True)
class SymbolicConfig:
    """User-facing symbolic / rule-learning knobs (serializable summary)."""

    backend: SymbolicBackend = "sklearn"
    source: SymbolicSource = "decision_tree"
    method: IndustrySymbolicMethod | None = None
    task: SymbolicTask = "classification"
    verify_constraints: bool = False
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    max_depth: int = 4
    min_samples_leaf: int = 5
    max_rules: int = 32
    default_consequent: Any = None
    prefer_reduce_components: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "source": self.source,
            "method": self.method,
            "task": self.task,
            "verify_constraints": self.verify_constraints,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_rules": self.max_rules,
            "default_consequent": self.default_consequent,
            "prefer_reduce_components": self.prefer_reduce_components,
        }


@dataclass(slots=True)
class NeuroSymbolicConfig:
    """User-facing neuro-symbolic hybrid knobs."""

    backend: NeuroSymbolicBackend = "sklearn"
    mode: NeuroSymbolicMode = "constraint_overlay"
    base_estimator: BaseEstimatorName = "logistic_regression"
    torch_method: TorchNeuroMethod | None = None
    task: SymbolicTask = "classification"
    torch_epochs: int = 60
    device: str = "cpu"
    columns: tuple[str, ...] | None = None
    random_state: int | None = 0
    soft_strength: float = 0.5
    rule_source: SymbolicSource = "decision_tree"
    max_depth: int = 3
    min_samples_leaf: int = 5
    max_rules: int = 24
    prefer_reduce_components: bool = True
    disclosures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "base_estimator": self.base_estimator,
            "torch_method": self.torch_method,
            "task": self.task,
            "torch_epochs": self.torch_epochs,
            "device": self.device,
            "columns": None if self.columns is None else list(self.columns),
            "random_state": self.random_state,
            "soft_strength": self.soft_strength,
            "rule_source": self.rule_source,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_rules": self.max_rules,
            "prefer_reduce_components": self.prefer_reduce_components,
            "disclosures": list(self.disclosures),
        }
