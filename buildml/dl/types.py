"""Name and default every knob on the Torch training path.

Deep learning has more settings than classical machine learning, and most of
them interact. Collecting them in dataclasses rather than spreading them across
function signatures buys three things: a single place to look up what a setting
does, a value that can be serialised into a bundle so a run is reproducible, and
defaults that are stated once rather than repeated at every call site.

The defaults are conservative on purpose. Training runs with no scheduler, no
gradient clipping, no early stopping, and no mixed precision: a plain loop that
works everywhere and does nothing surprising. Each of those is worth enabling
when a specific problem calls for it, and the attribute documentation says which
problem.

Module constants such as :data:`DEFAULT_EPOCHS` exist so the catalog, the tests,
and the documentation all read the same numbers. Changing one changes all three.

See Also
--------
buildml.dl.train : The loop these configure.
buildml.dl.results : What the loop produces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

TaskSpec = Literal["classification", "regression", "auto"]
DeviceName = Literal["cpu", "cuda", "mps", "auto"]
# Runtime also accepts ``cuda:N`` device strings for single-node DDP ranks.
SchedulerName = Literal["none", "step", "plateau", "cosine"]
EarlyStopMode = Literal["min", "max"]

# Documented TrainConfig defaults (M2). Change with care: tests and catalog cite these.
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
    """What device you asked for, what you got, and whether they differ.

    Requesting ``'cuda'`` on a machine without a GPU falls back to CPU rather
    than failing: training slowly beats not training. But a silent fallback is
    how someone ends up waiting hours for a run they believed was on a GPU, so
    the substitution is recorded here and surfaced as a warning.

    Attributes
    ----------
    requested:
        What the caller asked for: ``'cpu'``, ``'cuda'``, ``'mps'``, ``'auto'``,
        or a specific ``'cuda:N'``.
    resolved:
        What is actually being used.
    fallback_warning:
        An explanation when the two differ; ``None`` when the request was
        honoured.

    See Also
    --------
    buildml.dl.metrics.resolve_device : Produces this.
    """

    requested: str
    resolved: str
    fallback_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the device decision as JSON-safe values.

        Recorded in bundles and results, so that a run on unexpectedly slow
        hardware can be diagnosed after the fact.

        Returns
        -------
        dict
            ``requested``, ``resolved``, and ``fallback_warning``.
        """
        return asdict(self)


@dataclass(slots=True)
class TrainConfig:
    """Everything that governs how a Torch training run behaves.

    The defaults train: five epochs of Adam at 1e-3, no scheduler, no clipping,
    no early stopping. That is a deliberate floor rather than a recommendation :
    it runs anywhere and does nothing unexpected, and each of the disabled
    features is worth turning on for a reason given below.

    Attributes
    ----------
    epochs:
        Passes over the training data. When resuming, this counts as
        *additional* epochs rather than a new total.
    learning_rate:
        Optimiser step size, and the setting most likely to be the problem. Loss
        that jumps around means it is too high; loss that barely moves means it
        is too low.
    batch_size:
        Rows per gradient step. Larger batches give steadier gradients and use
        more memory; smaller ones add noise, which sometimes helps
        generalisation and always slows throughput.
    num_workers:
        Background data-loading processes. Zero loads in the main process,
        which is slower but avoids multiprocessing problems: a sensible
        default in notebooks and on Windows.
    pin_memory:
        Pin host memory for faster GPU transfer. Only useful with CUDA.
    shuffle_train:
        Reshuffle training rows each epoch. Leave this on unless order carries
        meaning the model must see.
    drop_last:
        Discard a final undersized batch. Useful when a layer needs a fixed
        batch size.
    normalize:
        Standardise features using training statistics. The statistics are
        computed on train alone and stored in the contract, so validation and
        test are transformed rather than re-fitted.
    seed:
        Seeds initialisation and shuffling. Deep learning varies noticeably
        across seeds; one run is not a measurement.
    device:
        ``'auto'`` picks the best available. Explicit values are honoured when
        possible and fall back with a warning when not.
    grad_clip_norm:
        Cap the gradient norm before each step. ``None`` disables it. Set it
        when loss suddenly becomes ``NaN`` or spikes: that is usually one
        outsized gradient destroying the weights, and clipping bounds the
        damage.
    log_every:
        Record every Nth epoch in the history. Only affects what is recorded.
    early_stopping_patience:
        Stop after this many epochs without improvement. ``None`` disables it.
        Requires a validation partition: stopping on training loss would only
        detect that the model stopped memorising.
    early_stopping_monitor:
        Which recorded metric to watch, normally ``'val_loss'``.
    early_stopping_mode:
        ``'min'`` when lower is better (losses), ``'max'`` when higher is
        (accuracies).
    early_stopping_min_delta:
        How much of a change counts as improvement. Raise it to ignore noise.
    restore_best_weights:
        After stopping, reload the best monitored epoch. On by default, because
        the last epoch of an early-stopped run is by construction one of the
        worse ones.
    scheduler:
        ``'none'`` keeps the rate fixed. ``'step'`` cuts it on a fixed
        timetable. ``'cosine'`` decays it smoothly to near zero over
        ``scheduler_t_max``, which suits a known epoch budget. ``'plateau'``
        cuts it only when the monitored metric stops improving, which suits a
        run whose length you cannot predict.
    scheduler_step_size, scheduler_gamma:
        For ``'step'``: multiply the rate by ``gamma`` every ``step_size``
        epochs.
    scheduler_t_max:
        For ``'cosine'``: the decay horizon. Defaults to ``epochs``.
    scheduler_factor, scheduler_patience, scheduler_threshold:
        For ``'plateau'``: how much to cut, how long to wait first, and how
        much change counts as improvement.
    mixed_precision:
        Use half precision where it is safe, which is faster and lighter on
        memory. CUDA only: on CPU or MPS it is a no-op that records a warning
        rather than a silent nothing.

    See Also
    --------
    buildml.dl.train.train_supervised_module : The loop this configures.
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
        """Return every training setting as JSON-safe values.

        Written into bundles and results, so a run months old still says what
        it was trained with: which is most of what reproducing it requires.

        Returns
        -------
        dict
            All fields, including those the active configuration does not use.
        """
        return asdict(self)


@dataclass(slots=True)
class FeatureContract:
    """The data shape a trained module expects, travelling with the module.

    A ``state_dict`` is a bag of tensors. It does not record which columns fed
    it, in what order, what the class labels were, or how the features were
    scaled: and every one of those is needed to use the module correctly. Get
    the column order wrong and inference produces confident nonsense rather than
    an error.

    Attributes
    ----------
    feature_columns:
        The input columns, in the order the module expects them.
    target_column:
        What it predicts.
    task:
        ``'classification'`` or ``'regression'``. Decides the default loss and
        how outputs are interpreted.
    class_labels:
        The original labels behind the integer class indices, so predictions
        come back as the values the user supplied.
    normalize_mean, normalize_std:
        Per-feature training statistics, ``None`` when normalisation was off.
        Stored rather than recomputed: recomputing them on inference data would
        scale each batch by its own statistics, which is both wrong and
        undetectable from the output.

    See Also
    --------
    buildml.dl.dataset : Builds the contract from a Dataset.
    """

    feature_columns: tuple[str, ...]
    target_column: str
    task: Literal["classification", "regression"]
    class_labels: tuple[Any, ...] = ()
    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the contract as JSON-safe values.

        Written to a bundle manifest as plain text, so the expected columns can
        be read without loading the model.

        Returns
        -------
        dict
            Feature columns, target, task, class labels, and normalisation
            statistics. Class labels are converted from NumPy scalars: which
            pandas produces and ``json`` refuses: to Python values.

        See Also
        --------
        from_dict : The inverse.
        """
        def _jsonable(value: Any) -> Any:
            # numpy scalars (common from pandas unique) are not JSON-serializable.
            item = getattr(value, "item", None)
            if callable(item):
                try:
                    return item()
                except Exception:  # pragma: no cover - non-scalar edge
                    return value
            return value

        return {
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": [_jsonable(v) for v in self.class_labels],
            "normalize_mean": None
            if self.normalize_mean is None
            else [float(v) for v in self.normalize_mean],
            "normalize_std": None
            if self.normalize_std is None
            else [float(v) for v in self.normalize_std],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureContract:
        """Rebuild a contract from its serialised form.

        Used when loading a bundle, to restore what the module expects before
        any data is fed to it.

        Parameters
        ----------
        payload:
            A mapping produced by :meth:`to_dict`.

        Returns
        -------
        FeatureContract
            The restored contract, with lists narrowed back to tuples so it
            stays immutable.

        Raises
        ------
        KeyError
            If ``feature_columns``, ``target_column``, or ``task`` is absent.
            These have no safe default: guessing any of them would produce a
            contract that silently disagrees with the module.

        See Also
        --------
        to_dict : The forward direction.
        """
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
    """The subset of training settings that affect how data is fed in.

    A narrower :class:`TrainConfig` for the paths that build loaders without
    running a training loop. The fields mean the same things.

    Attributes
    ----------
    batch_size:
        Rows per batch.
    num_workers:
        Background loading processes; ``0`` loads in the main process.
    pin_memory:
        Pin host memory for faster GPU transfer.
    shuffle_train:
        Reshuffle training rows each epoch. Never applied to validation or
        test, where order does not affect the result and stability aids
        debugging.
    drop_last:
        Discard a final undersized batch.
    normalize:
        Standardise features using training statistics only.
    seed:
        Seeds the shuffle, so epoch order is reproducible.

    See Also
    --------
    TrainConfig : The full set, for the training loop.
    """

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    normalize: bool = True
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the loader settings as JSON-safe values.

        Recorded alongside a loader bundle so the batching that produced a
        result is part of the record.

        Returns
        -------
        dict
            All seven fields.
        """
        return asdict(self)
