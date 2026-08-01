"""Typed results for the Torch thin slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.dl.types import DeviceSpec, FeatureContract, TrainConfig


@dataclass(slots=True)
class LoaderReport:
    """Summary of partition → DataLoader construction."""

    batch_size: int
    shuffle_train: bool
    normalize: bool
    feature_columns: tuple[str, ...]
    target_column: str
    task: Literal["classification", "regression"]
    n_train: int
    n_validation: int
    n_test: int
    class_labels: tuple[Any, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle_train": self.shuffle_train,
            "normalize": self.normalize,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "task": self.task,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "class_labels": list(self.class_labels),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class TorchLoaderBundle:
    """In-memory loaders plus the feature contract used to build them."""

    loaders: dict[str, Any]
    contract: FeatureContract
    report: LoaderReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "partitions": sorted(self.loaders),
            "contract": self.contract.to_dict(),
            "report": self.report.to_dict(),
        }


@dataclass(slots=True)
class TrainResult:
    """Outcome of a Torch supervised train loop."""

    module: Any
    task: Literal["classification", "regression"]
    config: TrainConfig
    device: DeviceSpec
    contract: FeatureContract
    optimizer_state: dict[str, Any] | None
    history: list[dict[str, float]]
    n_train_rows: int
    n_epochs_ran: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": type(self.module).__name__,
            "task": self.task,
            "config": self.config.to_dict(),
            "device": self.device.to_dict(),
            "contract": self.contract.to_dict(),
            "history": list(self.history),
            "n_train_rows": self.n_train_rows,
            "n_epochs_ran": self.n_epochs_ran,
            "warnings": list(self.warnings),
            "has_optimizer_state": self.optimizer_state is not None,
        }


@dataclass(slots=True)
class DLEvaluateResult:
    """Metrics for a Torch module on one partition."""

    partition: str
    task: Literal["classification", "regression"]
    metrics: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0
    device: str = "cpu"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "metrics": dict(self.metrics),
            "n_rows": self.n_rows,
            "device": self.device,
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        """Print a short evaluation digest."""
        print(f"evaluate_torch · {self.task} · partition={self.partition} · n={self.n_rows}")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for tip in self.recommendations[:10]:
            print(f"  - {tip}")
