"""Typed configuration for the Torch thin slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

TaskSpec = Literal["classification", "regression", "auto"]
DeviceName = Literal["cpu", "cuda", "mps", "auto"]


@dataclass(slots=True)
class DeviceSpec:
    """Resolved compute device with fallback disclosure."""

    requested: str
    resolved: str
    fallback_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainConfig:
    """Epoch-loop knobs for :func:`buildml.dl.train.train_supervised_module`."""

    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    normalize: bool = True
    seed: int = 0
    device: DeviceName = "auto"
    grad_clip_norm: float | None = None
    log_every: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureContract:
    """Feature / label schema carried with a trainer bundle."""

    feature_columns: tuple[str, ...]
    target_column: str
    task: Literal["classification", "regression"]
    class_labels: tuple[Any, ...] = ()
    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": list(self.class_labels),
            "normalize_mean": None
            if self.normalize_mean is None
            else list(self.normalize_mean),
            "normalize_std": None if self.normalize_std is None else list(self.normalize_std),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureContract:
        mean = payload.get("normalize_mean")
        std = payload.get("normalize_std")
        labels = payload.get("class_labels") or ()
        return cls(
            feature_columns=tuple(payload["feature_columns"]),
            target_column=str(payload["target_column"]),
            task=payload["task"],
            class_labels=tuple(labels),
            normalize_mean=None if mean is None else tuple(float(v) for v in mean),
            normalize_std=None if std is None else tuple(float(v) for v in std),
        )


@dataclass(slots=True)
class LoaderConfig:
    """DataLoader factory knobs."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    normalize: bool = True
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
