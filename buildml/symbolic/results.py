"""Typed results for symbolic / neuro-symbolic Session path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.symbolic.rules import RuleKnowledgeBase, RuleTrace


@dataclass(slots=True)
class SymbolicPlan:
    """Compiled / induced rule knowledge base.

    Persist via ``buildml.symbolic_bundle.v1``. Honesty: structured if-then
    rules over tabular columns (declared or train-induced decision tree /
    decision list) — **not** an AGI symbolic reasoner, Prolog engine, or Z3
    SMT solver.
    """

    source: str
    task: str
    columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    n_rules: int
    knowledge_base: RuleKnowledgeBase
    classes_: tuple[Any, ...] | None
    backend: str = "sklearn"
    method: str | None = None
    tree_estimator_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "symbolic",
            "backend": self.backend,
            "method": self.method,
            "source": self.source,
            "task": self.task,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_rules": self.n_rules,
            "knowledge_base": self.knowledge_base.to_dict(),
            "classes": None if self.classes_ is None else list(self.classes_),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class NeuroSymbolicPlan:
    """Hybrid neural/sklearn estimator + symbolic rule overlay.

    Persist via ``buildml.symbolic_bundle.v1`` (same format family; tagged
    ``kind=neuro_symbolic``). Honesty: sklearn base model + rule constraints /
    features — **not** a deep neuro-symbolic research platform or logic
    programming stack.
    """

    mode: str
    base_estimator_name: str
    task: str
    columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    knowledge_base: RuleKnowledgeBase
    estimator_: Any = field(repr=False)
    backend: str = "sklearn"
    torch_method: str | None = None
    label_encoder_: Any = field(repr=False, default=None)
    classes_: tuple[Any, ...] | None = None
    rule_feature_names_: tuple[str, ...] = ()
    soft_strength: float = 0.5
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "neuro_symbolic",
            "backend": self.backend,
            "mode": self.mode,
            "base_estimator_name": self.base_estimator_name,
            "torch_method": self.torch_method,
            "task": self.task,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_rules": len(self.knowledge_base.rules),
            "knowledge_base": self.knowledge_base.to_dict(),
            "classes": None if self.classes_ is None else list(self.classes_),
            "rule_feature_names": list(self.rule_feature_names_),
            "soft_strength": self.soft_strength,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class SymbolicFitResult:
    """Outcome of fitting / compiling a symbolic rule base on train."""

    source: str
    task: str
    n_train_rows: int
    n_rules: int
    columns: tuple[str, ...]
    target_column: str
    provenance: str
    backend: str = "sklearn"
    method: str | None = None
    classes: tuple[Any, ...] | None = None
    train_accuracy: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "backend": self.backend,
            "method": self.method,
            "task": self.task,
            "n_train_rows": self.n_train_rows,
            "n_rules": self.n_rules,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "provenance": self.provenance,
            "classes": None if self.classes is None else list(self.classes),
            "train_accuracy": self.train_accuracy,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NeuroSymbolicFitResult:
    """Outcome of fitting a neuro-symbolic hybrid on train."""

    mode: str
    base_estimator_name: str
    task: str
    n_train_rows: int
    n_rules: int
    columns: tuple[str, ...]
    target_column: str
    rule_provenance: str
    backend: str = "sklearn"
    torch_method: str | None = None
    classes: tuple[Any, ...] | None = None
    train_score: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "base_estimator_name": self.base_estimator_name,
            "torch_method": self.torch_method,
            "task": self.task,
            "n_train_rows": self.n_train_rows,
            "n_rules": self.n_rules,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "rule_provenance": self.rule_provenance,
            "classes": None if self.classes is None else list(self.classes),
            "train_score": self.train_score,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class SymbolicEvalResult:
    """Holdout evaluation for symbolic / neuro-symbolic predictions."""

    partition: str
    path: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    rule_coverage: float | None = None
    mean_rules_fired: float | None = None
    repair_rate: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "path": self.path,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "rule_coverage": self.rule_coverage,
            "mean_rules_fired": self.mean_rules_fired,
            "repair_rate": self.repair_rate,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class SymbolicPredictResult:
    """Predictions with optional rule-firing explanation traces."""

    partition: str
    path: str
    task: str
    n_rows: int
    predictions: tuple[Any, ...]
    traces: tuple[RuleTrace, ...] = ()
    neural_predictions: tuple[Any, ...] | None = None
    n_repaired: int = 0
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "path": self.path,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "n_traces": len(self.traces),
            "has_neural_predictions": self.neural_predictions is not None,
            "n_repaired": self.n_repaired,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
