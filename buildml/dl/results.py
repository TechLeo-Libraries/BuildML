"""Typed results for the Torch supervised path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.dl.types import DeviceSpec, FeatureContract, TrainConfig


@dataclass(slots=True)
class EarlyStopInfo:
    """Why training stopped and which validation monitor drove the decision."""

    enabled: bool
    triggered: bool
    monitor: str
    mode: str
    patience: int | None
    best_epoch: int | None
    best_value: float | None
    stopped_epoch: int
    restore_best_weights: bool
    partition: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "triggered": self.triggered,
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "stopped_epoch": self.stopped_epoch,
            "restore_best_weights": self.restore_best_weights,
            "partition": self.partition,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EarlyStopInfo:
        return cls(
            enabled=bool(payload.get("enabled", False)),
            triggered=bool(payload.get("triggered", False)),
            monitor=str(payload.get("monitor") or "val_loss"),
            mode=str(payload.get("mode") or "min"),
            patience=payload.get("patience"),
            best_epoch=payload.get("best_epoch"),
            best_value=payload.get("best_value"),
            stopped_epoch=int(payload.get("stopped_epoch") or 0),
            restore_best_weights=bool(payload.get("restore_best_weights", True)),
            partition=str(payload.get("partition") or "validation"),
            reason=str(payload.get("reason") or ""),
        )


@dataclass(slots=True)
class TrainingCurveReport:
    """Structured epoch curves plus interpretation and honesty limits."""

    epochs: list[int]
    train_loss: list[float]
    val_loss: list[float | None]
    learning_rates: list[float | None]
    monitor: str | None
    monitor_values: list[float | None]
    early_stop_epoch: int | None
    device_resolved: str
    early_stop_partition: str | None
    interpretation: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    disclosures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": list(self.epochs),
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "learning_rates": list(self.learning_rates),
            "monitor": self.monitor,
            "monitor_values": list(self.monitor_values),
            "early_stop_epoch": self.early_stop_epoch,
            "device_resolved": self.device_resolved,
            "early_stop_partition": self.early_stop_partition,
            "interpretation": list(self.interpretation),
            "limitations": list(self.limitations),
            "disclosures": list(self.disclosures),
        }


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
    split_kind: str | None = None
    group_column: str | None = None
    time_column: str | None = None
    groups_disjoint: bool | None = None
    time_order_ok: bool | None = None

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
            "split_kind": self.split_kind,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "groups_disjoint": self.groups_disjoint,
            "time_order_ok": self.time_order_ok,
        }


@dataclass(slots=True)
class TorchLoaderBundle:
    """In-memory loaders plus the feature contract used to build them."""

    loaders: dict[str, Any]
    contract: FeatureContract
    report: LoaderReport
    # Optional modality metadata (text / multimodal / image). Kept on the
    # slotted dataclass so Session factories can read train-only fit artifacts.
    text_vocab: Any | None = None
    text_contract: Any | None = None
    multimodal_contract: Any | None = None
    modality: str | None = None
    input_layout: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "partitions": sorted(self.loaders),
            "contract": self.contract.to_dict(),
            "report": self.report.to_dict(),
        }
        if self.modality is not None:
            payload["modality"] = self.modality
        if self.input_layout is not None:
            payload["input_layout"] = list(self.input_layout)
        return payload


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
    early_stop: EarlyStopInfo | None = None
    scheduler_name: str = "none"
    scheduler_state: dict[str, Any] | None = None
    resumed_from_epochs: int = 0
    training_curve: TrainingCurveReport | None = None

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
            "early_stop": None if self.early_stop is None else self.early_stop.to_dict(),
            "scheduler_name": self.scheduler_name,
            "has_scheduler_state": self.scheduler_state is not None,
            "resumed_from_epochs": self.resumed_from_epochs,
            "training_curve": None
            if self.training_curve is None
            else self.training_curve.to_dict(),
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
    confusion_matrix: list[list[int]] | None = None
    class_labels: tuple[Any, ...] = ()
    residuals_summary: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "task": self.task,
            "metrics": dict(self.metrics),
            "n_rows": self.n_rows,
            "device": self.device,
            "recommendations": list(self.recommendations),
            "confusion_matrix": self.confusion_matrix,
            "class_labels": list(self.class_labels),
            "residuals_summary": self.residuals_summary,
        }

    def show(self) -> None:
        """Print a short evaluation digest."""
        print(f"evaluate_torch · {self.task} · partition={self.partition} · n={self.n_rows}")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for tip in self.recommendations[:10]:
            print(f"  - {tip}")
