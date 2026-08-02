"""Typed configuration for the Torch supervised path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

TaskSpec = Literal["classification", "regression", "auto"]
DeviceName = Literal["cpu", "cuda", "mps", "auto"]
# Runtime also accepts ``cuda:N`` device strings for single-node DDP ranks.
SchedulerName = Literal["none", "step", "plateau", "cosine"]
EarlyStopMode = Literal["min", "max"]

# Documented TrainConfig defaults (M2). Change with care — tests and catalog cite these.
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_GRAD_CLIP_NORM: float | None = None  # disabled
DEFAULT_SCHEDULER: SchedulerName = "none"
DEFAULT_EARLY_STOPPING_PATIENCE: int | None = None  # disabled
DEFAULT_EARLY_STOPPING_MONITOR = "val_loss"
DEFAULT_EARLY_STOPPING_MODE: EarlyStopMode = "min"
DEFAULT_RESTORE_BEST_WEIGHTS = True
DEFAULT_SCHEDULER_STEP_SIZE = 10
DEFAULT_SCHEDULER_GAMMA = 0.1
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_SCHEDULER_FACTOR = 0.1
DEFAULT_SCHEDULER_THRESHOLD = 1e-4


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
    """Epoch-loop knobs for :func:`buildml.dl.train.train_supervised_module`.

    Defaults
    --------
    - ``epochs=5``, ``learning_rate=1e-3``, Adam when no optimizer factory is passed.
    - ``grad_clip_norm=None`` (no clipping). When set, applies
      ``clip_grad_norm_`` after ``backward`` and before ``optimizer.step``.
    - ``scheduler="none"``. ``step`` → StepLR; ``plateau`` → ReduceLROnPlateau on
      the early-stop monitor (or ``val_loss`` / ``train_loss``); ``cosine`` →
      CosineAnnealingLR over ``scheduler_t_max`` (default: ``epochs``).
    - ``early_stopping_patience=None`` (disabled). When set, monitors
      ``early_stopping_monitor`` on the validation loader (requires a validation
      partition). ``restore_best_weights=True`` reloads the best monitored epoch.
    - ``mixed_precision=False``. When True on a CUDA device, uses autocast +
      GradScaler. On CPU/MPS this is a documented no-op with a warning.
    """

    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    normalize: bool = True
    seed: int = 0
    device: str = "auto"
    grad_clip_norm: float | None = DEFAULT_GRAD_CLIP_NORM
    log_every: int = 1
    early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_monitor: str = DEFAULT_EARLY_STOPPING_MONITOR
    early_stopping_mode: EarlyStopMode = DEFAULT_EARLY_STOPPING_MODE
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = DEFAULT_RESTORE_BEST_WEIGHTS
    scheduler: SchedulerName = DEFAULT_SCHEDULER
    scheduler_step_size: int = DEFAULT_SCHEDULER_STEP_SIZE
    scheduler_gamma: float = DEFAULT_SCHEDULER_GAMMA
    scheduler_t_max: int | None = None
    scheduler_factor: float = DEFAULT_SCHEDULER_FACTOR
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE
    scheduler_threshold: float = DEFAULT_SCHEDULER_THRESHOLD
    mixed_precision: bool = False

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
