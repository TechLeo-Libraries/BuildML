"""What the Torch training path hands back.

A trained module on its own is not enough to know what happened. These types
carry the surrounding facts: what the loop saw each epoch, why it stopped, what
device it ran on, what data shape the module expects, and what quietly went
differently from what was asked for.

Two of them deserve attention beyond their fields.
:class:`TrainingCurveReport` carries not just the loss curves but an
interpretation of their shape and a statement of the limits of that reading —
because a curve invites confident conclusions it often cannot support.
:class:`EarlyStopInfo` records the partition the stopping decision was made on,
so a stopping claim can always be traced to the data behind it.

``to_dict`` on each returns JSON-safe values for bundles and history. Live
objects — the module, optimiser state — are described rather than embedded.

See Also
--------
buildml.dl.train : Produces most of these.
buildml.dl.checkpoint : Persists them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.dl.types import DeviceSpec, FeatureContract, TrainConfig


@dataclass(slots=True)
class EarlyStopInfo:
    """Why training stopped, and on what evidence.

    "Training stopped at epoch 12" is ambiguous: it could mean the budget ran
    out, or that validation loss had not improved for three epochs. Those are
    very different situations — the first suggests training longer, the second
    suggests the model had already peaked — so the distinction is recorded
    rather than left to be inferred.

    Attributes
    ----------
    enabled:
        Whether early stopping was configured at all.
    triggered:
        Whether it actually fired. ``enabled`` and not ``triggered`` means the
        epoch budget ran out first, which usually means more epochs were
        available to use.
    monitor:
        Which metric was watched.
    mode:
        ``'min'`` or ``'max'``.
    patience:
        Epochs without improvement tolerated before stopping.
    best_epoch, best_value:
        Where the monitored metric was at its best. When
        ``restore_best_weights`` is set, these describe the weights you are
        holding — not the last epoch trained.
    stopped_epoch:
        The last epoch run.
    restore_best_weights:
        Whether the best epoch's weights were reloaded at the end.
    partition:
        Which data the decision was made on. Always ``'validation'`` — recorded
        so the claim can be traced rather than trusted.
    reason:
        A readable explanation of why the loop ended.

    Notes
    -----
    **``best_epoch`` and ``stopped_epoch`` normally differ**, and the gap is
    ``patience``. That is not a bug: patience exists to let a metric wander
    before concluding it has stopped improving.

    See Also
    --------
    buildml.dl.types.TrainConfig : The settings behind these fields.
    """

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
        """Return the stopping record as JSON-safe values.

        Persisted in bundles so a resumed run can pick up the best-so-far
        bookkeeping rather than starting its patience count from scratch.

        Returns
        -------
        dict
            All eleven fields.

        See Also
        --------
        from_dict : The inverse.
        """
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
        """Rebuild a stopping record from its serialised form.

        Used when loading a bundle, so that resuming continues the same
        patience count instead of restarting it.

        Parameters
        ----------
        payload:
            A mapping produced by :meth:`to_dict`.

        Returns
        -------
        EarlyStopInfo
            The restored record.

        Notes
        -----
        Every field has a default, so a manifest written by an older version
        loads rather than failing. The defaults describe a disabled monitor,
        which is the safe reading of an absent field.

        See Also
        --------
        to_dict : The forward direction.
        """
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
    """The loss curves, what they appear to say, and what they cannot.

    A training curve is the most-read diagnostic in deep learning and the most
    over-read. Falling training loss with rising validation loss looks like
    overfitting and usually is; a flat curve looks like a learning rate problem
    and often is not. This report carries the numbers, a plain reading of their
    shape, and — deliberately — the limits of that reading.

    Attributes
    ----------
    epochs:
        Epoch numbers, aligned with every list below.
    train_loss:
        Training loss per epoch.
    val_loss:
        Validation loss per epoch, or ``None`` entries when no validation
        loader was present.
    learning_rates:
        The rate in force at each epoch, which makes a scheduler's effect
        visible against the curve.
    monitor:
        The early-stopping metric, or ``None`` when it was disabled.
    monitor_values:
        That metric per epoch.
    early_stop_epoch:
        Where stopping fired, or ``None``.
    device_resolved:
        What the run actually ran on.
    early_stop_partition:
        Which data the stopping decision used.
    interpretation:
        Plain-language readings of the curve's shape.
    limitations:
        What those readings cannot establish. Present because an
        interpretation without its limits is more dangerous than no
        interpretation.
    disclosures:
        Facts about the run that affect how the curve should be read.

    Notes
    -----
    **A curve describes the run, not the model's worth.** Smooth convergence to
    a low training loss is consistent with a model that has memorised the
    training set. Only holdout evaluation distinguishes the two.

    See Also
    --------
    buildml.dl.curves.build_training_curve : Produces this from a result.
    """

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
        """Return the curves and their reading as JSON-safe values.

        Complete rather than summarised, so the numbers can be replotted and
        the interpretation stays attached to the data it describes.

        Returns
        -------
        dict
            Every list and scalar, including the interpretation, limitations,
            and disclosures.
        """
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
    """What went into the loaders, including whether the split was sound.

    Records both the mechanics of batching and the integrity of the split
    behind it. The second matters more: a group split that let the same subject
    appear in train and test produces a holdout score that means nothing, and
    that is far easier to check here than to detect later.

    Attributes
    ----------
    batch_size, shuffle_train, normalize:
        The batching settings in force.
    feature_columns, target_column, task:
        The data shape, mirroring the contract.
    n_train, n_validation, n_test:
        Rows per partition. A zero here explains a missing loader.
    class_labels:
        The original labels behind the class indices.
    warnings:
        Anything noticed while building — tiny partitions, degenerate classes,
        constant features.
    split_kind:
        How the data was split: random, grouped, or time-based.
    group_column, time_column:
        The column the split was organised around, when relevant.
    groups_disjoint:
        For a group split, whether no group appears in two partitions.
        ``False`` here invalidates the holdout score.
    time_order_ok:
        For a time split, whether every training timestamp precedes the holdout
        ones. ``False`` means the model was trained on the future.

    Notes
    -----
    **Check ``groups_disjoint`` and ``time_order_ok`` before reading any
    metric.** They are the two ways a split silently stops being a holdout, and
    a model evaluated on a leaked split will look excellent right up until it
    is deployed.

    See Also
    --------
    buildml.dl.dataset : Builds loaders and this report.
    """

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
        """Return the loader summary as JSON-safe values.

        Includes the split-integrity flags, deliberately: a recorded result
        should carry the evidence that its holdout was actually held out.

        Returns
        -------
        dict
            Batching settings, data shape, partition sizes, warnings, and the
            split-integrity fields.
        """
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
    """The loaders to train on, together with what they are feeding.

    A bare ``DataLoader`` yields tensors and says nothing about which columns
    they came from or how they were scaled. Keeping the loaders, the contract,
    and the build report together means the training loop can validate what it
    is being handed rather than assuming.

    Attributes
    ----------
    loaders:
        Partition name to ``DataLoader``. A ``'train'`` key is required; a
        ``'validation'`` key enables validation loss and early stopping.
    contract:
        The data shape, carried forward into the trained model.
    report:
        How the loaders were built, and whether the split holds up.
    text_vocab, text_contract:
        Train-fitted text artefacts, for the text path. The vocabulary is built
        from training documents alone, so holdout text meets genuine unknown
        tokens rather than ones the vocabulary quietly learned.
    multimodal_contract:
        Per-modality preprocessing for combined inputs.
    speech_contract:
        Audio preprocessing settings.
    modality:
        Which path built this bundle: tabular, text, image, audio, or
        multimodal.
    input_layout:
        The order inputs arrive in for multi-input batches, so a module knows
        which tensor is which.

    Notes
    -----
    **The modality artefacts are fitted on train only.** A vocabulary or
    normalisation statistic learned from all partitions is leakage, and the
    kind that inflates a holdout score without ever raising an error.

    See Also
    --------
    buildml.dl.dataset : Builds these bundles.
    """

    loaders: dict[str, Any]
    contract: FeatureContract
    report: LoaderReport
    # Optional modality metadata (text / multimodal / image). Kept on the
    # slotted dataclass so Session factories can read train-only fit artifacts.
    text_vocab: Any | None = None
    text_contract: Any | None = None
    multimodal_contract: Any | None = None
    speech_contract: Any | None = None
    modality: str | None = None
    input_layout: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Describe the bundle as JSON-safe values.

        The loaders themselves are named rather than embedded — they hold data
        and worker processes, neither of which belongs in a record.

        Returns
        -------
        dict
            The partition names, the contract, the build report, and the
            modality and input layout when set.
        """
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
    """Everything a training run produced, not just the trained module.

    The module is what you use; the rest is what makes it trustworthy and
    resumable. Optimiser and scheduler state are here because resuming without
    them makes a run stumble for several epochs — Adam's momentum estimates are
    part of where training had got to, not incidental.

    Attributes
    ----------
    module:
        The trained module, already on the resolved device.
    task:
        ``'classification'`` or ``'regression'``.
    config:
        The settings the run used.
    device:
        What was requested and what was actually used.
    contract:
        The data shape, which travels with the module into inference.
    optimizer_state:
        Optimiser internals, needed for a clean resume.
    history:
        Per-epoch metrics: epoch, training loss, learning rate, and validation
        loss when available.
    n_train_rows:
        How many rows the run trained on.
    n_epochs_ran:
        The last epoch reached, cumulative across resumes.
    warnings:
        Everything that quietly differed from what was asked for — a device
        fallback, mixed precision disabled, a scheduler that could not be
        restored.
    early_stop:
        Why the loop ended.
    scheduler_name, scheduler_state:
        Which schedule ran and where it had reached.
    resumed_from_epochs:
        Epochs already completed when this run started; ``0`` for a fresh run.
    training_curve:
        The curves with their interpretation and limits.
    multimodal_preprocess:
        Frozen per-modality preprocessing, persisted so a reloaded model
        transforms inputs the way it was trained to.

    Notes
    -----
    **Read ``warnings`` on every run.** A CUDA request that fell back to CPU
    trains correctly and slowly, and the only evidence is here.

    **``n_epochs_ran`` is cumulative.** Resuming for five more epochs after ten
    reports fifteen, not five.

    See Also
    --------
    buildml.dl.train.train_supervised_module : Produces this.
    buildml.dl.checkpoint : Persists it for resuming.
    """

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
    # Frozen multimodal preprocess meta (audio/image/text stats + layout).
    # Persisted in torch bundles for honesty; load does not rebuild loaders.
    multimodal_preprocess: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Describe the run as JSON-safe values.

        The module and the optimiser and scheduler state are tensors, so they
        are reported by name and presence rather than embedded. The bundle
        format carries the real weights.

        Returns
        -------
        dict
            Module class name, task, config, device, contract, full epoch
            history, row and epoch counts, warnings, early-stop record,
            scheduler name, whether optimiser and scheduler state are present,
            resume offset, the training curve, and multimodal preprocessing.

        See Also
        --------
        buildml.dl.checkpoint : Saves the weights this omits.
        """
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
            "multimodal_preprocess": self.multimodal_preprocess,
        }


@dataclass(slots=True)
class DLEvaluateResult:
    """How a trained module scored on one partition.

    The headline metrics plus the detail that gives them meaning: a confusion
    matrix for classification, a residual summary for regression, and
    recommendations pointing at what the numbers suggest doing next.

    Attributes
    ----------
    partition:
        Which data was scored.
    task:
        ``'classification'`` or ``'regression'``.
    metrics:
        The headline numbers for the task.
    n_rows:
        How many rows were scored. A small holdout makes every metric noisy,
        and this is where you notice.
    device:
        Where evaluation ran.
    recommendations:
        Concrete suggestions drawn from the metrics.
    confusion_matrix:
        For classification: predicted against actual counts. Reading it is how
        you find that overall accuracy is being carried by one dominant class.
    class_labels:
        The labels indexing the confusion matrix, in order.
    residuals_summary:
        For regression: the distribution of errors. Read it to find that errors
        are skewed, or concentrated at one end of the target range, rather than
        evenly spread as a single RMSE implies.

    Notes
    -----
    **The confusion matrix and residual summary are where the useful
    information is.** A single accuracy or RMSE compresses every kind of
    failure into one number; these show which kind you have.

    See Also
    --------
    buildml.dl.metrics : Computes these.
    """

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
        """Return the evaluation as JSON-safe values.

        Complete, including the confusion matrix and residual summary — a
        recorded evaluation should support the same reading later that it
        supported when it was produced.

        Returns
        -------
        dict
            Partition, task, metrics, row count, device, recommendations,
            confusion matrix, class labels, and residual summary.
        """
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
        """Print the metrics and top recommendations to the console.

        A quick look for interactive work. Prints the task, partition, and row
        count, then each metric, then up to ten recommendations — the confusion
        matrix and residual summary stay on the object, since neither reads
        well as console output.
        """
        print(f"evaluate_torch · {self.task} · partition={self.partition} · n={self.n_rows}")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        for tip in self.recommendations[:10]:
            print(f"  - {tip}")
