"""Thin Session facades over buildml.dl."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

from buildml.session._imports import (
    ValidationError,
)


def _attached_classical_plans(session) -> dict[str, Any]:
    """Return non-null classical plan objects currently attached to the Session."""
    if hasattr(session, "_plan_objects"):
        return {k: v for k, v in session._plan_objects().items() if v is not None}
    plans: dict[str, Any] = {}
    for key, attr in (
        ("impute", "_impute_plan"),
        ("encode", "_encode_plan"),
        ("scale", "_scale_plan"),
        ("outliers", "_outlier_plan"),
        ("binning", "_binning_plan"),
        ("feature_select", "_feature_select_plan"),
        ("dates", "_date_plan"),
        ("text_features", "_text_plan"),
        ("reduce_dimensions", "_reduce_plan"),
    ):
        value = getattr(session, attr, None)
        if value is not None:
            plans[key] = value
    return plans


def _module_needs_non_tabular_loaders(module: Any) -> str | None:
    """Return 'multimodal' / 'text' / 'speech' when auto tabular rebuild would be wrong."""
    modality = getattr(module, "modality", None) or ""
    layout = getattr(module, "input_layout", None)
    if str(modality) in {"speech_classify", "speech_encoder"} or (
        hasattr(module, "encoder") and hasattr(module, "head") and hasattr(module, "embed_dim")
    ):
        return "speech"
    if (
        str(modality).endswith("_fusion")
        or layout is not None
        or (
            hasattr(module, "n_numeric")
            and (
                hasattr(module, "embedding")
                or hasattr(module, "image_net")
                or hasattr(module, "audio_net")
            )
        )
    ):
        return "multimodal"
    if hasattr(module, "vocab_size") and hasattr(module, "embedding"):
        return "text"
    return None


def _refuse_silent_tabular_loader_rebuild(session, *, operation: str) -> None:
    """Raise when loaders are missing after a non-tabular Torch fit."""
    if session._torch_loaders is not None:
        return
    train_result = getattr(session, "_dl_train_result", None)
    if train_result is None:
        return
    kind = _module_needs_non_tabular_loaders(train_result.module)
    if kind == "multimodal":
        raise ValidationError(
            f"{operation} needs active multimodal loaders after multimodal fit. "
            "Call make_multimodal_torch_loaders(...) / "
            "make_image_multimodal_torch_loaders(...) / "
            "make_audio_multimodal_torch_loaders(...) again. "
            "Refusing silent tabular loader rebuild."
        )
    if kind == "text":
        raise ValidationError(
            f"{operation} needs active text loaders after text fit. "
            "Call make_text_torch_loaders(...) again. "
            "Refusing silent tabular loader rebuild."
        )
    if kind == "speech":
        raise ValidationError(
            f"{operation} needs active speech loaders after speech fit. "
            "Call make_speech_torch_loaders(...) again. "
            "Refusing silent tabular loader rebuild."
        )


def make_torch_loaders(
    session,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
    drop_last: bool = False,
    normalize: bool = True,
    seed: int = 0,
    task: Literal["classification", "regression", "auto"] = "auto",
    apply_plans: bool = False,
) -> Any:
    """Build Torch DataLoaders from current roles and split partitions.

    Requires ``pip install 'buildml[torch]'`` (or ``buildml[dl]``). Shuffle
    applies to the train loader only. When ``normalize`` is True, mean/std
    are fit on train and frozen for validation/test.

    Classical preprocess: "Session" ``impute`` / ``encode`` / ``scale`` already
    mutate the attached frame with train-fitted plans. Attached plans are
    disclosed on the loader report. Pass ``apply_plans=True`` to explicitly
    re-apply fitted plans via :meth:`apply_preprocess_plans` before building
    loaders (score-time replay; does not refit).

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    batch_size:
        Minibatch size for all loaders.
    num_workers:
        DataLoader worker processes.
    pin_memory:
        When True, pin CPU memory for faster GPU transfer.
    shuffle_train:
        When True, shuffle the train loader each epoch.
    drop_last:
        When True, drop the final partial train batch.
    normalize:
        When True, fit normalize stats on train only.
    seed:
        Seed for shuffling and sampling.
    task:
        Supervised task (``classification``, ``regression``, or ``auto``).
    apply_plans:
        When True, re-apply Session preprocess plans before building loaders.

    Returns
    -------
    TorchLoaderBundle
        Loaders keyed by partition plus the feature contract.
    """
    from buildml.dl.loaders import make_loaders
    from buildml.dl.types import LoaderConfig

    session.assert_can_fit("train")
    if apply_plans and _attached_classical_plans(session):
        session.apply_preprocess_plans(inplace=True, use_session_plans=True)
    classical = _attached_classical_plans(session)
    bundle = make_loaders(
        session.dataset,
        session._split_plan,
        config=LoaderConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
            drop_last=drop_last,
            normalize=normalize,
            seed=seed,
        ),
        task=task,
        classical_plans=classical or None,
    )
    session._torch_loaders = bundle
    session._record(
        "make_torch_loaders",
        {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "shuffle_train": shuffle_train,
            "drop_last": drop_last,
            "normalize": normalize,
            "seed": seed,
            "task": task,
            "apply_plans": apply_plans,
            "classical_plans": sorted(classical),
        },
        result_summary=bundle.report.to_dict(),
        warnings=tuple(bundle.report.warnings),
    )
    return bundle


def make_text_torch_loaders(
    session,
    *,
    text_column: str | None = None,
    batch_size: int = 16,
    max_len: int = 64,
    max_vocab: int = 5000,
    min_freq: int = 1,
    shuffle_train: bool = True,
    seed: int = 0,
) -> Any:
    """Build token-id DataLoaders for text classification (non-tabular modality).

    Vocabulary is fit on the train partition only. Requires ``buildml[torch]``.
    Delegates to :func:`buildml.dl.text.make_text_loaders`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    text_column:
        Optional text column; auto-detected when omitted.
    batch_size:
        Minibatch size for all loaders.
    max_len:
        Maximum token sequence length.
    max_vocab:
        Maximum vocabulary size fit on train.
    min_freq:
        Minimum token frequency to enter the vocabulary.
    shuffle_train:
        When True, shuffle the train loader each epoch.
    seed:
        Seed for shuffling and vocabulary sampling.

    Returns
    -------
    TorchLoaderBundle
        Text loaders plus vocabulary and text contract metadata.
    """
    from buildml.dl.text import TextLoaderConfig, make_text_loaders

    session.assert_can_fit("train")
    bundle = make_text_loaders(
        session.dataset,
        session._split_plan,
        text_column=text_column,
        config=TextLoaderConfig(
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            seed=seed,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
        ),
    )
    session._torch_loaders = bundle
    session._record(
        "make_text_torch_loaders",
        {
            "text_column": text_column
            or getattr(getattr(bundle, "text_contract", None), "text_column", None),
            "batch_size": batch_size,
            "max_len": max_len,
            "max_vocab": max_vocab,
            "seed": seed,
        },
        result_summary=bundle.report.to_dict(),
        warnings=tuple(bundle.report.warnings),
    )
    return bundle


def fit_torch(
    session,
    module: Any | None = None,
    *,
    loss_fn: Any | None = None,
    optimizer_factory: Any | None = None,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    grad_clip_norm: float | None = None,
    log_every: int = 1,
    early_stopping_patience: int | None = None,
    early_stopping_monitor: str = "val_loss",
    scheduler: Literal["none", "step", "plateau", "cosine"] = "none",
    resume: bool = False,
    config: Any | None = None,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
    mixed_precision: bool = False,
) -> "Session":
    """Train an ``nn.Module`` on the train Torch loader.

    Requires ``pip install 'buildml[torch]'``. When ``module`` is omitted, builds
    a tabular MLP, text classifier, or multimodal fusion module from the active
    loader contract so the happy path does not require a hand-rolled network.

    Does not replace classical :meth:`fit` / :attr:`fit_result`.
    Delegates to :func:`buildml.dl.train.train_supervised_module`.

    Parameters
    ----------
    session:
        Active Session with torch loaders attached or auto-built.
    module:
        Optional ``nn.Module`` to train; auto-built when omitted.
    loss_fn:
        Optional custom loss function.
    optimizer_factory:
        Optional factory returning a torch optimizer.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    device:
        Compute device (``cpu``, ``cuda``, ``mps``, or ``auto``).
    grad_clip_norm:
        Optional gradient clipping norm.
    log_every:
        Log training metrics every N epochs.
    early_stopping_patience:
        Optional validation patience for early stopping.
    early_stopping_monitor:
        Metric name monitored for early stopping.
    scheduler:
        Learning-rate scheduler kind.
    resume:
        When True, resume from the prior ``dl_train_result``.
    config:
        Optional full :class:`~buildml.dl.types.TrainConfig` override.
    hidden:
        Hidden layer sizes for auto-built tabular MLPs.
    dropout:
        Dropout rate for auto-built modules.
    mixed_precision:
        When True, enable autocast mixed precision where supported.

    Returns
    -------
    Session
        ``session`` with ``dl_train_result`` attached for chaining.

    Raises
    ------
    ValidationError
        When resume is requested without a prior trainer or multimodal
        contracts are incomplete.
    """
    from buildml.dl.models import build_tabular_mlp, build_text_classifier
    from buildml.dl.train import train_supervised_module
    from buildml.dl.types import TrainConfig

    session.assert_can_fit("train")
    _refuse_silent_tabular_loader_rebuild(session, operation="fit_torch")
    if session._torch_loaders is None:
        session.make_torch_loaders()
    assert session._torch_loaders is not None
    if config is None:
        config = TrainConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            grad_clip_norm=grad_clip_norm,
            log_every=log_every,
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor=early_stopping_monitor,
            scheduler=scheduler,
            batch_size=getattr(session._torch_loaders.report, "batch_size", 32),
            normalize=getattr(session._torch_loaders.report, "normalize", True),
            mixed_precision=mixed_precision,
        )
    prior = None
    if resume:
        if session._dl_train_result is None:
            raise ValidationError(
                "resume=True requires dl_train_result. Call load_torch_bundle(...) or fit_torch(...) first."
            )
        prior = session._dl_train_result
    if module is None:
        if resume and prior is not None:
            module = prior.module
        else:
            from buildml.dl.labels import n_classes_from_labels
            from buildml.dl.multimodal import build_multimodal_fusion

            text_vocab = getattr(session._torch_loaders, "text_vocab", None)
            multimodal = getattr(session._torch_loaders, "multimodal_contract", None)
            speech = getattr(session._torch_loaders, "speech_contract", None)
            contract = session._torch_loaders.contract
            modality = getattr(session._torch_loaders, "modality", None) or ""
            n_classes = n_classes_from_labels(contract.class_labels)
            if speech is not None or modality == "speech_classify":
                from buildml.dl.speech import build_speech_classifier

                embed_dim = int(getattr(speech, "encoder_dim", 64) or 64) if speech else 64
                sample_rate = (
                    int(getattr(speech, "sample_rate", 16_000) or 16_000) if speech else 16_000
                )
                module = build_speech_classifier(
                    n_classes=n_classes,
                    embed_dim=embed_dim,
                    sample_rate=sample_rate,
                )
            elif multimodal is not None or str(modality).endswith("_fusion"):
                mm = multimodal
                if mm is None:
                    raise ValidationError(
                        "Multimodal loaders are missing multimodal_contract for fit_torch"
                    )
                has_text = bool(mm.text_column) or bool(mm.vocab)
                has_image = bool(mm.image_column)
                has_audio = bool(getattr(mm, "audio_column", None))
                vocab_size = 0
                if has_text:
                    vocab_size = (
                        int(getattr(text_vocab, "vocab_size", 0))
                        or int((mm.vocab or {}).get("vocab_size") or 0)
                        or len((mm.vocab or {}).get("id_to_token") or [])
                    )
                    if vocab_size < 2:
                        raise ValidationError(
                            "Multimodal loaders are missing vocabulary metadata for fit_torch"
                        )
                module = build_multimodal_fusion(
                    len(mm.numeric_columns),
                    vocab_size,
                    image_channels=int(mm.image_channels) if has_image else 0,
                    image_size=tuple(mm.image_size) if has_image else (32, 32),
                    audio_channels=1 if has_audio else 0,
                    audio_samples=int(getattr(mm, "audio_max_samples", 16_000) or 16_000)
                    if has_audio
                    else 16_000,
                    task=contract.task,
                    n_classes=n_classes,
                    dropout=dropout,
                )
            elif text_vocab is not None and multimodal is None:
                module = build_text_classifier(
                    text_vocab.vocab_size,
                    n_classes=n_classes,
                    dropout=dropout,
                )
            else:
                in_features = len(contract.feature_columns)
                module = build_tabular_mlp(
                    in_features,
                    task=contract.task,
                    n_classes=n_classes,
                    hidden=hidden,
                    dropout=dropout,
                )
    result = train_supervised_module(
        module,
        session._torch_loaders,
        config=config,
        loss_fn=loss_fn,
        optimizer_factory=optimizer_factory,
        resume_from=prior,
    )
    session._dl_train_result = result
    session._record(
        "fit_torch",
        {
            "module": type(module).__name__,
            "epochs": result.n_epochs_ran,
            "device": result.device.to_dict(),
            "task": result.task,
            "resume": resume,
            "scheduler": result.scheduler_name,
            "early_stopping_patience": result.config.early_stopping_patience,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return cast("Session", session)
def cross_validate_torch(
    session,
    *,
    n_folds: int = 3,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    normalize: bool = True,
    seed: int = 0,
    stratify: bool = True,
    task: Literal["classification", "regression", "auto"] = "auto",
    module_factory: Any | None = None,
) -> Any:
    """Fold-local Torch CV on the attached numeric tabular dataset.

    Normalize stats are fit per fold. Classical Session plans are disclosed as
    a limitation unless you supply a custom factory path: this helper does not
    silently refit Session-global plans inside each fold.
    Delegates to :func:`buildml.dl.cv.cross_validate_torch`.

    Parameters
    ----------
    session:
        Active Session with an attached tabular dataset.
    n_folds:
        Number of cross-validation folds.
    epochs:
        Training epochs per fold.
    batch_size:
        Minibatch size per fold.
    learning_rate:
        Optimizer learning rate per fold.
    device:
        Compute device for fold-local training.
    normalize:
        When True, fit normalize stats per fold on train only.
    seed:
        Seed for fold splitting and training.
    stratify:
        When True, stratify folds for classification tasks.
    task:
        Supervised task (``classification``, ``regression``, or ``auto``).
    module_factory:
        Optional factory returning a fresh module per fold.

    Returns
    -------
    TorchCVResult
        Per-fold metrics and mean summary.

    Raises
    ------
    ValidationError
        When no dataset is attached to the Session.
    """
    from buildml.dl.cv import cross_validate_torch as _cv

    if session._dataset is None:
        raise ValidationError("No dataset attached. Call ingest(...) first.")
    result = _cv(
        session.dataset,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        normalize=normalize,
        seed=seed,
        stratify=stratify,
        task=task,
        module_factory=module_factory,
    )
    session._dl_cv_result = result
    session._record(
        "cross_validate_torch",
        {
            "n_folds": n_folds,
            "epochs": epochs,
            "task": result.task,
            "mean_metrics": result.mean_metrics,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def torch_training_curve(session) -> Any:
    """Return structured training-curve teaching data for the last Torch run.

    Requires a prior :meth:`fit_torch` / :meth:`load_torch_bundle`. Torch-free
    to read once :attr:`dl_train_result` exists.
    Delegates to :func:`buildml.dl.curves.build_training_curve`.

    Parameters
    ----------
    session:
        Active Session with a prior torch training result.

    Returns
    -------
    TrainingCurve
        Epoch-wise loss/metric series for visualization or reporting.

    Raises
    ------
    ValidationError
        When no torch trainer exists on the Session.
    """
    from buildml.dl.curves import build_training_curve

    if session._dl_train_result is None:
        raise ValidationError(
            "No Torch trainer. Call fit_torch(...) or load_torch_bundle(...) first."
        )
    curve = session._dl_train_result.training_curve
    if curve is None:
        curve = build_training_curve(session._dl_train_result)
        session._dl_train_result.training_curve = curve
    session._record("torch_training_curve", {}, result_summary=curve.to_dict())
    return curve


def evaluate_torch(
    session,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    device: str | None = None,
) -> Any:
    """Evaluate the last Torch trainer on a named partition.

    Requires ``pip install 'buildml[torch]'``. Uses loaders from
    :meth:`make_torch_loaders` (rebuilds them if missing).
    Delegates to :func:`buildml.dl.metrics.evaluate_module`.

    Parameters
    ----------
    session:
        Active Session with a prior torch training result.
    partition:
        Partition to evaluate (``train``, ``validation``, or ``test``).
    device:
        Optional device override for evaluation.

    Returns
    -------
    TorchEvalResult
        Partition metrics for the trained module.

    Raises
    ------
    ValidationError
        When no torch trainer exists or non-tabular loaders are missing.
    """
    from buildml.dl.metrics import evaluate_module

    if session._dl_train_result is None:
        raise ValidationError(
            "No Torch trainer. Call fit_torch(...) or load_torch_bundle(...) first."
        )
    _refuse_silent_tabular_loader_rebuild(session, operation="evaluate_torch")
    if session._torch_loaders is None:
        session.make_torch_loaders(
            normalize=session._dl_train_result.contract.normalize_mean is not None,
            task=session._dl_train_result.task,
        )
    assert session._torch_loaders is not None
    result = evaluate_module(
        session._dl_train_result, session._torch_loaders, partition=partition, device=device
    )
    session._record(
        "evaluate_torch",
        {"partition": partition, "device": device},
        result_summary=result.to_dict(),
    )
    return result


def save_torch_bundle(session, path: str | Path) -> Path:
    """Persist the last Torch trainer as ``buildml.torch_bundle.v1``.

    Distinct from Session checkpoints and classical pipeline bundles.
    See :data:`buildml.dl.checkpoint.CHECKPOINT_BOUNDARY`.
    Delegates to :func:`buildml.dl.checkpoint.save_torch_bundle`.

    Parameters
    ----------
    session:
        Active Session with a prior torch training result.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no torch trainer exists on the Session.
    """
    from buildml.dl.checkpoint import save_torch_bundle

    if session._dl_train_result is None:
        raise ValidationError("No Torch trainer. Call fit_torch(...) first.")
    destination = save_torch_bundle(path, session._dl_train_result)
    session._record("save_torch_bundle", {"path": str(destination)})
    return destination


def load_torch_bundle(
    session,
    path: str | Path,
    module: Any,
    *,
    map_location: str | None = None,
    trusted: bool = False,
) -> "Session":
    """Load a Torch trainer bundle into this Session.

    Restores weights plus optional ``multimodal_preprocess`` meta (frozen
    image/audio stats and layout). Does **not** rebuild DataLoaders: remake
    multimodal/text loaders before fit/evaluate/export.
    Delegates to :func:`buildml.dl.checkpoint.load_torch_bundle`.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded trainer.
    path:
        Bundle directory with ``meta.json`` and ``trainer.pt``.
    module:
        Compatible ``nn.Module`` shell that receives ``load_state_dict``.
    map_location:
        Optional device for ``torch.load`` (default CPU).
    trusted:
        Required ``True`` to deserialize ``trainer.pt`` (torch pickle). Pass
        only for bundles you created or fully trust.

    Returns
    -------
    Session
        ``session`` with ``dl_train_result`` attached for chaining.
    """
    from buildml.dl.checkpoint import load_torch_bundle as _load_torch_bundle

    session._dl_train_result = _load_torch_bundle(
        path, module, map_location=map_location, trusted=trusted
    )
    session._record(
        "load_torch_bundle",
        {"path": str(path), "module": type(module).__name__, "map_location": map_location},
        result_summary=session._dl_train_result.to_dict(),
    )
    return cast("Session", session)
def make_multimodal_torch_loaders(
    session,
    *,
    text_column: str | None = None,
    numeric_columns: list[str] | None = None,
    image_column: str | None = None,
    audio_column: str | None = None,
    batch_size: int = 16,
    max_len: int = 64,
    max_vocab: int = 5000,
    min_freq: int = 1,
    normalize: bool = True,
    normalize_images: bool = True,
    normalize_audio: bool = True,
    image_size: tuple[int, int] = (32, 32),
    image_channels: int = 3,
    audio_sample_rate: int = 16_000,
    audio_max_samples: int = 16_000,
    audio_source_sample_rate: int | None = None,
    shuffle_train: bool = True,
    seed: int = 0,
    task: Literal["classification", "regression", "auto"] = "auto",
    preprocess: Any | None = None,
    use_saved_preprocess: bool = False,
) -> Any:
    """Build fused multimodal DataLoaders (tabular/text/image/audio mixes).

    Requires ``buildml[torch]``. Fit stats (vocab, numeric mean/std, image
    channel mean/std, audio amplitude mean/std) use the train partition only.
    Batches follow ``(numeric?, tokens?, image?, audio?, y)`` for present
    modalities. Audio fusion is a small 1D-CNN branch: not a speech foundation
    model.

    Pass ``preprocess=`` (contract / dict) to freeze restore stats, or
    ``use_saved_preprocess=True`` to reuse ``dl_train_result.multimodal_preprocess``.
    Delegates to :func:`buildml.dl.multimodal.make_multimodal_loaders`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    text_column:
        Optional text column for multimodal fusion.
    numeric_columns:
        Optional numeric columns for tabular branch.
    image_column:
        Optional image column/path column.
    audio_column:
        Optional audio column/path column.
    batch_size:
        Minibatch size for all loaders.
    max_len:
        Maximum token sequence length for text branch.
    max_vocab:
        Maximum vocabulary size fit on train.
    min_freq:
        Minimum token frequency for vocabulary.
    normalize:
        When True, normalize numeric features on train only.
    normalize_images:
        When True, normalize image channels on train only.
    normalize_audio:
        When True, normalize audio amplitude on train only.
    image_size:
        Target image height/width for image branch.
    image_channels:
        Number of image channels.
    audio_sample_rate:
        Target audio sample rate after resampling.
    audio_max_samples:
        Maximum audio samples per example.
    audio_source_sample_rate:
        Optional source sample rate before resampling.
    shuffle_train:
        When True, shuffle the train loader each epoch.
    seed:
        Seed for shuffling and preprocessing.
    task:
        Supervised task (``classification``, ``regression``, or ``auto``).
    preprocess:
        Optional frozen preprocess contract/dict to restore stats.
    use_saved_preprocess:
        When True, reuse preprocess meta from ``dl_train_result``.

    Returns
    -------
    TorchLoaderBundle
        Multimodal loaders plus contracts and preprocess disclosures.

    Raises
    ------
    ValidationError
        When both ``preprocess`` and ``use_saved_preprocess`` are supplied or
        saved preprocess meta is missing.
    """
    from buildml.dl.multimodal import MultimodalLoaderConfig, make_multimodal_loaders

    session.assert_can_fit("train")
    resolved_preprocess = preprocess
    if use_saved_preprocess:
        if resolved_preprocess is not None:
            raise ValidationError(
                "Pass preprocess= or use_saved_preprocess=True, not both."
            )
        train = getattr(session, "_dl_train_result", None)
        saved = None if train is None else getattr(train, "multimodal_preprocess", None)
        if saved is None:
            raise ValidationError(
                "use_saved_preprocess=True requires a prior fit_torch / "
                "load_torch_bundle with multimodal_preprocess meta."
            )
        resolved_preprocess = saved
    bundle = make_multimodal_loaders(
        session.dataset,
        session._split_plan,
        text_column=text_column,
        numeric_columns=numeric_columns,
        image_column=image_column,
        audio_column=audio_column,
        config=MultimodalLoaderConfig(
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            seed=seed,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            normalize=normalize,
            normalize_images=normalize_images,
            normalize_audio=normalize_audio,
            image_size=image_size,
            image_channels=image_channels,
            audio_sample_rate=audio_sample_rate,
            audio_max_samples=audio_max_samples,
            audio_source_sample_rate=audio_source_sample_rate,
        ),
        task=task,
        preprocess=resolved_preprocess,
    )
    session._torch_loaders = bundle
    mm = getattr(bundle, "multimodal_contract", None)
    session._record(
        "make_multimodal_torch_loaders",
        {
            "text_column": text_column or getattr(mm, "text_column", None),
            "image_column": image_column or getattr(mm, "image_column", None),
            "audio_column": audio_column or getattr(mm, "audio_column", None),
            "numeric_columns": list(getattr(mm, "numeric_columns", ()) or ()),
            "batch_size": batch_size,
            "normalize": normalize,
            "normalize_images": normalize_images,
            "normalize_audio": normalize_audio,
            "image_size": list(image_size),
            "image_channels": image_channels,
            "audio_sample_rate": audio_sample_rate,
            "audio_max_samples": audio_max_samples,
            "modality": getattr(bundle, "modality", None),
            "seed": seed,
            "task": task,
            "use_saved_preprocess": use_saved_preprocess,
            "preprocess": resolved_preprocess is not None,
        },
        result_summary=bundle.report.to_dict(),
        warnings=tuple(bundle.report.warnings),
    )
    return bundle


def make_image_multimodal_torch_loaders(
    session,
    *,
    image_column: str,
    text_column: str | None = None,
    numeric_columns: list[str] | None = None,
    audio_column: str | None = None,
    batch_size: int = 16,
    max_len: int = 64,
    max_vocab: int = 5000,
    min_freq: int = 1,
    normalize: bool = True,
    normalize_images: bool = True,
    normalize_audio: bool = True,
    image_size: tuple[int, int] = (32, 32),
    image_channels: int = 3,
    audio_sample_rate: int = 16_000,
    audio_max_samples: int = 16_000,
    audio_source_sample_rate: int | None = None,
    shuffle_train: bool = True,
    seed: int = 0,
    task: Literal["classification", "regression", "auto"] = "auto",
) -> Any:
    """Build image multimodal loaders (image ⊕ tabular and/or text and/or audio).

    Thin facade that requires ``image_column`` and delegates to the shared
    multimodal loader builder. Path cells need Pillow (bundled in
    ``buildml[torch]``); array/list cells work with Torch alone.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    image_column:
        Required image column or path column.
    text_column:
        Optional text column for multimodal fusion.
    numeric_columns:
        Optional numeric columns for tabular branch.
    audio_column:
        Optional audio column for audio branch.
    batch_size:
        Minibatch size for all loaders.
    max_len:
        Maximum token sequence length for text branch.
    max_vocab:
        Maximum vocabulary size fit on train.
    min_freq:
        Minimum token frequency for vocabulary.
    normalize:
        When True, normalize numeric features on train only.
    normalize_images:
        When True, normalize image channels on train only.
    normalize_audio:
        When True, normalize audio amplitude on train only.
    image_size:
        Target image height/width for image branch.
    image_channels:
        Number of image channels.
    audio_sample_rate:
        Target audio sample rate after resampling.
    audio_max_samples:
        Maximum audio samples per example.
    audio_source_sample_rate:
        Optional source sample rate before resampling.
    shuffle_train:
        When True, shuffle the train loader each epoch.
    seed:
        Seed for shuffling and preprocessing.
    task:
        Supervised task (``classification``, ``regression``, or ``auto``).

    Returns
    -------
    TorchLoaderBundle
        Image-centric multimodal loaders plus contracts.

    Raises
    ------
    ValidationError
        When ``image_column`` is missing or empty.
    """
    if not image_column:
        raise ValidationError("make_image_multimodal_torch_loaders requires image_column")
    from buildml.dl.multimodal import MultimodalLoaderConfig, make_multimodal_loaders

    session.assert_can_fit("train")
    bundle = make_multimodal_loaders(
        session.dataset,
        session._split_plan,
        text_column=text_column,
        numeric_columns=numeric_columns,
        image_column=image_column,
        audio_column=audio_column,
        config=MultimodalLoaderConfig(
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            seed=seed,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            normalize=normalize,
            normalize_images=normalize_images,
            normalize_audio=normalize_audio,
            image_size=image_size,
            image_channels=image_channels,
            audio_sample_rate=audio_sample_rate,
            audio_max_samples=audio_max_samples,
            audio_source_sample_rate=audio_source_sample_rate,
        ),
        task=task,
    )
    session._torch_loaders = bundle
    mm = getattr(bundle, "multimodal_contract", None)
    session._record(
        "make_image_multimodal_torch_loaders",
        {
            "image_column": image_column or getattr(mm, "image_column", None),
            "text_column": text_column or getattr(mm, "text_column", None),
            "audio_column": audio_column or getattr(mm, "audio_column", None),
            "numeric_columns": list(getattr(mm, "numeric_columns", ()) or ()),
            "batch_size": batch_size,
            "normalize": normalize,
            "normalize_images": normalize_images,
            "image_size": list(image_size),
            "image_channels": image_channels,
            "modality": getattr(bundle, "modality", None),
            "seed": seed,
            "task": task,
        },
        result_summary=bundle.report.to_dict(),
        warnings=tuple(bundle.report.warnings),
    )
    return bundle


def make_audio_multimodal_torch_loaders(
    session,
    *,
    audio_column: str,
    text_column: str | None = None,
    numeric_columns: list[str] | None = None,
    image_column: str | None = None,
    batch_size: int = 16,
    max_len: int = 64,
    max_vocab: int = 5000,
    min_freq: int = 1,
    normalize: bool = True,
    normalize_images: bool = True,
    normalize_audio: bool = True,
    image_size: tuple[int, int] = (32, 32),
    image_channels: int = 3,
    audio_sample_rate: int = 16_000,
    audio_max_samples: int = 16_000,
    audio_source_sample_rate: int | None = None,
    shuffle_train: bool = True,
    seed: int = 0,
    task: Literal["classification", "regression", "auto"] = "auto",
) -> Any:
    """Build audio multimodal loaders (audio ⊕ tabular and/or text and/or image).

    Thin facade that requires ``audio_column`` and delegates to the shared
    multimodal loader builder. Path cells need soundfile (bundled in
    ``buildml[torch]`` / ``buildml[audio]``); waveform array cells work with
    Torch alone. Uses a small 1D-CNN fusion branch: not a speech foundation
    model.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    audio_column:
        Required audio column or path column.
    text_column:
        Optional text column for multimodal fusion.
    numeric_columns:
        Optional numeric columns for tabular branch.
    image_column:
        Optional image column for image branch.
    batch_size:
        Minibatch size for all loaders.
    max_len:
        Maximum token sequence length for text branch.
    max_vocab:
        Maximum vocabulary size fit on train.
    min_freq:
        Minimum token frequency for vocabulary.
    normalize:
        When True, normalize numeric features on train only.
    normalize_images:
        When True, normalize image channels on train only.
    normalize_audio:
        When True, normalize audio amplitude on train only.
    image_size:
        Target image height/width for optional image branch.
    image_channels:
        Number of image channels.
    audio_sample_rate:
        Target audio sample rate after resampling.
    audio_max_samples:
        Maximum audio samples per example.
    audio_source_sample_rate:
        Optional source sample rate before resampling.
    shuffle_train:
        When True, shuffle the train loader each epoch.
    seed:
        Seed for shuffling and preprocessing.
    task:
        Supervised task (``classification``, ``regression``, or ``auto``).

    Returns
    -------
    TorchLoaderBundle
        Audio-centric multimodal loaders plus contracts.

    Raises
    ------
    ValidationError
        When ``audio_column`` is missing or empty.
    """
    if not audio_column:
        raise ValidationError("make_audio_multimodal_torch_loaders requires audio_column")
    from buildml.dl.multimodal import MultimodalLoaderConfig, make_multimodal_loaders

    session.assert_can_fit("train")
    bundle = make_multimodal_loaders(
        session.dataset,
        session._split_plan,
        text_column=text_column,
        numeric_columns=numeric_columns,
        image_column=image_column,
        audio_column=audio_column,
        config=MultimodalLoaderConfig(
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            seed=seed,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            normalize=normalize,
            normalize_images=normalize_images,
            normalize_audio=normalize_audio,
            image_size=image_size,
            image_channels=image_channels,
            audio_sample_rate=audio_sample_rate,
            audio_max_samples=audio_max_samples,
            audio_source_sample_rate=audio_source_sample_rate,
        ),
        task=task,
    )
    session._torch_loaders = bundle
    mm = getattr(bundle, "multimodal_contract", None)
    session._record(
        "make_audio_multimodal_torch_loaders",
        {
            "audio_column": audio_column or getattr(mm, "audio_column", None),
            "text_column": text_column or getattr(mm, "text_column", None),
            "image_column": image_column or getattr(mm, "image_column", None),
            "numeric_columns": list(getattr(mm, "numeric_columns", ()) or ()),
            "batch_size": batch_size,
            "normalize": normalize,
            "normalize_audio": normalize_audio,
            "audio_sample_rate": audio_sample_rate,
            "audio_max_samples": audio_max_samples,
            "modality": getattr(bundle, "modality", None),
            "seed": seed,
            "task": task,
        },
        result_summary=bundle.report.to_dict(),
        warnings=tuple(bundle.report.warnings),
    )
    return bundle


def search_torch(
    session,
    *,
    param_grid: dict[str, list[Any]] | None = None,
    param_distributions: dict[str, Any] | None = None,
    inner_search: Literal["grid", "randomized", "auto"] = "auto",
    n_iter: int = 5,
    n_folds: int = 3,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    normalize: bool = True,
    seed: int = 0,
    stratify: bool = True,
    task: Literal["classification", "regression", "auto"] = "auto",
    scoring_metric: str | None = None,
    module_factory: Any | None = None,
) -> Any:
    """Inner-fold Torch hyperparameter search on the Session train universe.

    Held-out validation/test partitions are never scored. For a nested outer
    estimate after search, use :meth:`nested_cv_torch`.
    Delegates to :func:`buildml.dl.search.search_torch`.

    Parameters
    ----------
    session:
        Active Session with an attached tabular dataset.
    param_grid:
        Optional grid of hyperparameter lists.
    param_distributions:
        Optional randomized search distributions.
    inner_search:
        Inner search strategy (``grid``, ``randomized``, or ``auto``).
    n_iter:
        Randomized search iterations when applicable.
    n_folds:
        Number of inner CV folds.
    epochs:
        Training epochs per candidate per fold.
    batch_size:
        Minibatch size for inner CV training.
    learning_rate:
        Optimizer learning rate for inner CV training.
    device:
        Compute device for inner CV training.
    normalize:
        When True, fit normalize stats per fold on train only.
    seed:
        Seed for fold splitting and search sampling.
    stratify:
        When True, stratify folds for classification tasks.
    task:
        Supervised task (``classification``, ``regression``, or ``auto``).
    scoring_metric:
        Optional metric name for ranking candidates.
    module_factory:
        Optional factory returning a fresh module per candidate/fold.

    Returns
    -------
    TorchSearchResult
        Best params, inner CV scores, and search disclosures.

    Raises
    ------
    ValidationError
        When no dataset is attached to the Session.
    """
    from buildml.dl.search import search_torch as _search

    if session._dataset is None:
        raise ValidationError("No dataset attached. Call ingest(...) first.")
    result = _search(
        session.dataset,
        split_plan=session._split_plan,
        param_grid=param_grid,
        param_distributions=param_distributions,
        inner_search=inner_search,
        n_iter=n_iter,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        normalize=normalize,
        seed=seed,
        stratify=stratify,
        task=task,
        scoring_metric=scoring_metric,
        module_factory=module_factory,
    )
    session._dl_search_result = result
    session._record(
        "search_torch",
        {
            "search_method": result.search_method,
            "n_folds": n_folds,
            "best_params": result.best_params,
            "scoring_metric": result.scoring_metric,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def nested_cv_torch(
    session,
    *,
    param_grid: dict[str, list[Any]] | None = None,
    param_distributions: dict[str, Any] | None = None,
    inner_search: Literal["grid", "randomized", "auto"] = "auto",
    n_iter: int = 5,
    outer_cv: int = 3,
    inner_cv: int = 2,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    normalize: bool = True,
    seed: int = 0,
    stratify: bool = True,
    task: Literal["classification", "regression", "auto"] = "auto",
    scoring_metric: str | None = None,
    module_factory: Any | None = None,
) -> Any:
    """Nested Torch CV: outer evaluation after fold-local inner hyperparameter search.

    Outer-eval rows never enter inner ranking. Session validation/test stay
    untouched. Normalize stats are fit fold-locally.
    Delegates to :func:`buildml.dl.search.nested_cv_torch`.

    Parameters
    ----------
    session:
        Active Session with an attached tabular dataset.
    param_grid:
        Optional grid of hyperparameter lists for inner search.
    param_distributions:
        Optional randomized search distributions for inner search.
    inner_search:
        Inner search strategy (``grid``, ``randomized``, or ``auto``).
    n_iter:
        Randomized search iterations when applicable.
    outer_cv:
        Number of outer evaluation folds.
    inner_cv:
        Number of inner CV folds per outer fold.
    epochs:
        Training epochs per candidate per inner fold.
    batch_size:
        Minibatch size for nested CV training.
    learning_rate:
        Optimizer learning rate for nested CV training.
    device:
        Compute device for nested CV training.
    normalize:
        When True, fit normalize stats per fold on train only.
    seed:
        Seed for fold splitting and search sampling.
    stratify:
        When True, stratify folds for classification tasks.
    task:
        Supervised task (``classification``, ``regression``, or ``auto``).
    scoring_metric:
        Optional metric name for inner ranking and outer reporting.
    module_factory:
        Optional factory returning a fresh module per candidate/fold.

    Returns
    -------
    TorchNestedCVResult
        Outer-fold metrics, inner search summaries, and disclosures.

    Raises
    ------
    ValidationError
        When no dataset is attached to the Session.
    """
    from buildml.dl.search import nested_cv_torch as _nested

    if session._dataset is None:
        raise ValidationError("No dataset attached. Call ingest(...) first.")
    result = _nested(
        session.dataset,
        split_plan=session._split_plan,
        param_grid=param_grid,
        param_distributions=param_distributions,
        inner_search=inner_search,
        n_iter=n_iter,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        normalize=normalize,
        seed=seed,
        stratify=stratify,
        task=task,
        scoring_metric=scoring_metric,
        module_factory=module_factory,
    )
    session._dl_nested_cv_result = result
    session._record(
        "nested_cv_torch",
        {
            "outer_cv": outer_cv,
            "inner_cv": inner_cv,
            "search_method": result.search_method,
            "mean_metrics": result.mean_metrics,
            "scoring_metric": result.scoring_metric,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def export_torch(
    session,
    path: str | Path,
    *,
    format: Literal["torchscript", "onnx"] = "torchscript",
    opset: int = 17,
    dynamic_batch: bool = True,
    example_input: Any | None = None,
) -> Any:
    """Export the last Torch trainer to TorchScript or ONNX.

    Uses train-loader example inputs when ``example_input`` is omitted.
    Alpha-quality escape hatch: see export result limitations.
    Delegates to :func:`buildml.dl.export.export_train_result`.

    Parameters
    ----------
    session:
        Active Session with a prior torch training result.
    path:
        Destination file path for the exported artifact.
    format:
        Export format (``torchscript`` or ``onnx``).
    opset:
        ONNX opset version when ``format='onnx'``.
    dynamic_batch:
        When True, declare dynamic batch axes where supported.
    example_input:
        Optional explicit example input matching module layout.

    Returns
    -------
    TorchExportResult
        Export path, format, and limitation disclosures.

    Raises
    ------
    ValidationError
        When no torch trainer exists or non-tabular loaders/example inputs
        are missing.
    """
    from buildml.dl.export import export_train_result

    if session._dl_train_result is None:
        raise ValidationError("No Torch trainer. Call fit_torch(...) first.")
    if session._torch_loaders is None and example_input is None:
        kind = _module_needs_non_tabular_loaders(session._dl_train_result.module)
        if kind == "multimodal":
            raise ValidationError(
                "export_torch needs active multimodal loaders or an explicit "
                "example_input matching the fusion input_layout after multimodal fit. "
                "Call make_multimodal_torch_loaders(...) / "
                "make_image_multimodal_torch_loaders(...) / "
                "make_audio_multimodal_torch_loaders(...) again or pass example_input=. "
                "Refusing silent tabular loader rebuild."
            )
        if kind == "text":
            raise ValidationError(
                "export_torch needs active text loaders or an explicit example_input "
                "after text fit. Call make_text_torch_loaders(...) again or pass "
                "example_input=. Refusing silent tabular loader rebuild."
            )
        session.make_torch_loaders(
            normalize=session._dl_train_result.contract.normalize_mean is not None,
            task=session._dl_train_result.task,
        )
    result = export_train_result(
        session._dl_train_result,
        path,
        format=format,
        loader_bundle=session._torch_loaders,
        example_input=example_input,
        opset=opset,
        dynamic_batch=dynamic_batch,
    )
    session._dl_export_result = result
    session._record(
        "export_torch",
        {"path": str(result.path), "format": format, "opset": opset},
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def fit_torch_ddp(
    session,
    module_factory: Any,
    *,
    epochs: int = 5,
    learning_rate: float = 0.001,
    mixed_precision: bool = False,
    world_size: int | None = None,
    allow_cpu_ddp: bool = False,
    multi_node: bool = False,
    config: Any | None = None,
) -> Any:
    """DDP training via a fresh ``module_factory`` per process.

    * Single-node (default): spawn local ranks. Requires
      ``torch.cuda.device_count() >= 2`` unless ``allow_cpu_ddp=True`` (gloo smoke).
    * Multi-node: ``multi_node=True`` joins a ``torchrun`` rendezvous
      (``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` /
      ``MASTER_PORT``; ``LOCAL_RANK`` is required: global rank is not used as a
      local CUDA index). Not a Kubernetes multi-cluster orchestrator.
    Delegates to :func:`buildml.dl.ddp.train_supervised_module_ddp`.

    Parameters
    ----------
    session:
        Active Session with torch loaders attached or auto-built.
    module_factory:
        Callable returning a fresh ``nn.Module`` per DDP process.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    mixed_precision:
        When True, enable autocast mixed precision where supported.
    world_size:
        Optional explicit process/world size override.
    allow_cpu_ddp:
        When True, permit CPU gloo smoke tests with fewer GPUs.
    multi_node:
        When True, join an external torchrun rendezvous instead of spawning.
    config:
        Optional full :class:`~buildml.dl.types.TrainConfig` override.

    Returns
    -------
    DDPTrainResult
        DDP run summary and optional aggregated train result.
    """
    from buildml.dl.ddp import DDPConfig, train_supervised_module_ddp
    from buildml.dl.types import TrainConfig

    session.assert_can_fit("train")
    _refuse_silent_tabular_loader_rebuild(session, operation="fit_torch_ddp")
    if session._torch_loaders is None:
        session.make_torch_loaders()
    assert session._torch_loaders is not None
    if config is None:
        config = TrainConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            mixed_precision=mixed_precision,
            batch_size=getattr(session._torch_loaders.report, "batch_size", 32),
        )
    ddp_result = train_supervised_module_ddp(
        module_factory,
        session._torch_loaders,
        config=config,
        ddp_config=DDPConfig(
            world_size=world_size,
            allow_cpu_ddp=allow_cpu_ddp,
            multi_node=multi_node,
        ),
    )
    if ddp_result.train_result is not None:
        session._dl_train_result = ddp_result.train_result
    session._dl_ddp_result = ddp_result
    session._record(
        "fit_torch_ddp",
        {
            "world_size": ddp_result.world_size,
            "backend": ddp_result.backend,
            "allow_cpu_ddp": allow_cpu_ddp,
            "multi_node": multi_node,
        },
        result_summary=ddp_result.to_dict(),
        warnings=tuple(ddp_result.warnings),
    )
    return ddp_result


def make_speech_torch_loaders(
    session,
    *,
    audio_column: str | None = None,
    batch_size: int = 8,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
    normalize_audio: bool = True,
    encoder_dim: int = 64,
    shuffle_train: bool = True,
    seed: int = 0,
) -> Any:
    """Build speech classification loaders (finetune-lite encoder path).

    Requires ``buildml[torch]``. Amplitude stats fit on train only. This is an
    integration/finetune path: not training a foundation model from scratch.
    Delegates to :func:`buildml.dl.speech.make_speech_loaders`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    audio_column:
        Optional audio column; auto-detected when omitted.
    batch_size:
        Minibatch size for all loaders.
    sample_rate:
        Target audio sample rate after resampling.
    max_samples:
        Maximum audio samples per example.
    source_sample_rate:
        Optional source sample rate before resampling.
    normalize_audio:
        When True, normalize audio amplitude on train only.
    encoder_dim:
        Encoder embedding dimension for speech contract metadata.
    shuffle_train:
        When True, shuffle the train loader each epoch.
    seed:
        Seed for shuffling and preprocessing.

    Returns
    -------
    TorchLoaderBundle
        Speech loaders plus speech contract metadata.
    """
    from buildml.dl.speech import SpeechLoaderConfig, make_speech_loaders

    session.assert_can_fit("train")
    bundle = make_speech_loaders(
        session.dataset,
        session._split_plan,
        audio_column=audio_column,
        config=SpeechLoaderConfig(
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            seed=seed,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
            normalize_audio=normalize_audio,
            encoder_dim=encoder_dim,
        ),
    )
    session._torch_loaders = bundle
    session._record(
        "make_speech_torch_loaders",
        {
            "audio_column": audio_column
            or getattr(getattr(bundle, "speech_contract", None), "audio_column", None),
            "batch_size": batch_size,
            "sample_rate": sample_rate,
            "max_samples": max_samples,
            "normalize_audio": normalize_audio,
            "modality": getattr(bundle, "modality", None),
        },
        result_summary=bundle.report.to_dict(),
        warnings=tuple(bundle.report.warnings),
    )
    return bundle


def fit_speech_torch(
    session,
    *,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    freeze_encoder: bool = False,
    audio_column: str | None = None,
    batch_size: int = 8,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
    normalize_audio: bool = True,
    encoder_dim: int = 64,
    seed: int = 0,
) -> Any:
    """Fine-tune a tiny speech encoder + classifier head (finetune-lite).

    Builds speech loaders when missing. Honest alpha: not Whisper-scale FM
    training from scratch. Requires ``buildml[torch]``.
    Delegates to :func:`buildml.dl.train.train_supervised_module` after building
    a speech classifier module.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    device:
        Compute device (``cpu``, ``cuda``, ``mps``, or ``auto``).
    freeze_encoder:
        When True, freeze the speech encoder during finetuning.
    audio_column:
        Optional audio column when loaders must be built.
    batch_size:
        Minibatch size when loaders must be built.
    sample_rate:
        Target sample rate when loaders must be built.
    max_samples:
        Maximum samples per example when loaders must be built.
    source_sample_rate:
        Optional source sample rate when loaders must be built.
    normalize_audio:
        When True, normalize audio amplitude on train only.
    encoder_dim:
        Encoder embedding dimension for the speech classifier.
    seed:
        Seed for shuffling and training.

    Returns
    -------
    Session
        ``session`` with ``dl_train_result`` attached for chaining.
    """
    from buildml.dl.labels import n_classes_from_labels
    from buildml.dl.speech import build_speech_classifier
    from buildml.dl.train import train_supervised_module
    from buildml.dl.types import TrainConfig

    session.assert_can_fit("train")
    loaders = session._torch_loaders
    modality = getattr(loaders, "modality", None) if loaders is not None else None
    if loaders is None or modality != "speech_classify":
        make_speech_torch_loaders(
            session,
            audio_column=audio_column,
            batch_size=batch_size,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
            normalize_audio=normalize_audio,
            encoder_dim=encoder_dim,
            seed=seed,
        )
    assert session._torch_loaders is not None
    contract = session._torch_loaders.contract
    speech = getattr(session._torch_loaders, "speech_contract", None)
    n_classes = n_classes_from_labels(contract.class_labels)
    embed = int(getattr(speech, "encoder_dim", encoder_dim) or encoder_dim)
    sr = int(getattr(speech, "sample_rate", sample_rate) or sample_rate)
    module = build_speech_classifier(
        n_classes=n_classes,
        embed_dim=embed,
        sample_rate=sr,
        freeze_encoder=freeze_encoder,
    )
    config = TrainConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        batch_size=getattr(session._torch_loaders.report, "batch_size", batch_size),
        seed=seed,
    )
    result = train_supervised_module(module, session._torch_loaders, config=config)
    session._dl_train_result = result
    session._record(
        "fit_speech_torch",
        {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "device": device,
            "freeze_encoder": freeze_encoder,
            "n_classes": n_classes,
            "embed_dim": embed,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return cast("Session", session)
def transcribe_speech(
    session,
    *,
    audio_column: str,
    backend: Literal["stub", "transformers", "auto"] | None = None,
    model_id: str | None = None,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
    partition: Literal["train", "validation", "test", "all"] = "all",
) -> Any:
    """ASR transcription for an audio feature column.

    Default prefers ``transformers`` when the speech stack is installed;
    otherwise falls back to the deterministic stub (CI-safe). Pass
    ``backend="stub"`` explicitly for offline tests. ``backend="transformers"``
    requires ``buildml[speech]`` and may download Whisper-class weights.
    Integration path only: not FM training from scratch.
    Delegates to :func:`buildml.dl.speech.transcribe_from_dataset`.

    Parameters
    ----------
    session:
        Active Session with an ingested dataset.
    audio_column:
        Audio feature column to transcribe.
    backend:
        ASR backend (``stub``, ``transformers``, ``auto``, or ``None`` for
        environment-aware default).
    model_id:
        Optional Hugging Face model id for transformers backend.
    sample_rate:
        Target audio sample rate for decoding.
    max_samples:
        Maximum audio samples per row.
    source_sample_rate:
        Optional source sample rate before resampling.
    partition:
        Dataset partition to transcribe (``all`` by default).

    Returns
    -------
    SpeechTranscribeResult
        Transcripts, model metadata, and row counts. Stub use is disclosed.

    Raises
    ------
    ValidationError
        When no dataset is attached to the Session.
    """
    from buildml.dl.speech import resolve_asr_backend, transcribe_from_dataset

    if session.dataset is None:
        raise ValidationError("transcribe_speech requires an ingested dataset")
    resolved = resolve_asr_backend(backend)
    result = transcribe_from_dataset(
        session.dataset,
        audio_column=audio_column,
        backend=resolved,
        model_id=model_id,
        sample_rate=sample_rate,
        max_samples=max_samples,
        source_sample_rate=source_sample_rate,
        partition=partition,
        split_plan=session._split_plan,
    )
    # Preserve requested vs resolved for history honesty.
    result.meta["requested_backend"] = backend
    session._dl_speech_result = result
    session._record(
        "transcribe_speech",
        {
            "audio_column": audio_column,
            "backend": result.backend,
            "requested_backend": backend,
            "model_id": result.model_id,
            "partition": partition,
            "n_rows": result.n_rows,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def serve_bundle(
    session,
    path: str | Path | None = None,
    *,
    kind: Literal["pipeline", "torchscript"] = "pipeline",
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "BuildML Serve",
    blocking: bool = False,
    api_keys: str | list[str] | tuple[str, ...] | None = None,
    basic_auth: str
    | tuple[str, str]
    | list[tuple[str, str]]
    | dict[str, str]
    | None = None,
    docs_enabled: bool | None = None,
    allow_insecure_public_bind: bool = False,
    ssl_certfile: str | Path | None = None,
    ssl_keyfile: str | Path | None = None,
    trusted: bool = False,
    config: Any | None = None,
) -> Any:
    """Launch BuildML managed serving for a pipeline or TorchScript artifact.

    Requires ``buildml[serve]``. Defaults to localhost bind. Optional
    ``api_keys`` (Bearer / ``X-API-Key``) and/or ``basic_auth`` enable shared-
    secret middleware (still not a managed IAM / cloud auth product). Either
    mechanism may authorize. When auth is enabled, OpenAPI docs default to
    closed unless ``docs_enabled=True``. Non-loopback binds without auth raise
    unless ``allow_insecure_public_bind=True``. Optional ``ssl_certfile`` /
    ``ssl_keyfile`` enable local uvicorn HTTPS (library-owned; not managed
    certs). Prefer TLS at a reverse proxy for non-local exposure. When ``path``
    is omitted and ``kind="pipeline"``, uses the last saved pipeline path
    recorded on the Session if available. Optional ``config`` accepts a
    :class:`~buildml.serving.config.ServeConfig`.

    Not registered as an AI tool: CLI / Session-primary by design.
    Delegates to :func:`buildml.serving.launch.serve_bundle`.

    Parameters
    ----------
    session:
        Active Session that may hold the last saved pipeline path.
    path:
        Optional artifact path; inferred for pipelines when omitted.
    kind:
        Artifact kind (``pipeline`` or ``torchscript``).
    host:
        Bind host address.
    port:
        Bind port number.
    title:
        Service title shown in OpenAPI metadata.
    blocking:
        When True, block until the server stops.
    api_keys:
        Optional API keys enabling Bearer / header auth middleware.
    basic_auth:
        Optional HTTP Basic credentials (``user:pass``, pair, or mapping).
    docs_enabled:
        OpenAPI/docs toggle; ``None`` auto-closes docs when auth is on.
    allow_insecure_public_bind:
        When True, permit non-loopback binds without auth.
    ssl_certfile:
        Optional TLS certificate file for local HTTPS.
    ssl_keyfile:
        Optional TLS private key file for local HTTPS.
    trusted:
        Must be ``True`` to deserialize the served artifact.
    config:
        Optional :class:`~buildml.serving.config.ServeConfig` base.

    Returns
    -------
    ServeHandle
        Running server handle with URL and lifecycle controls.

    Raises
    ------
    ValidationError
        When no resolvable artifact path is available.
    """
    from buildml.serving.launch import serve_bundle as _serve

    resolved = path
    if resolved is None and config is None:
        resolved = getattr(session, "_last_pipeline_path", None)
    if resolved is None and config is None:
        raise ValidationError(
            "serve_bundle requires path= to a pipeline bundle or TorchScript file "
            "(or a prior save_pipeline on this Session)."
        )
    handle = _serve(
        resolved,
        kind=kind,
        host=host,
        port=port,
        title=title,
        blocking=blocking,
        api_keys=api_keys,
        basic_auth=basic_auth,
        docs_enabled=docs_enabled,
        allow_insecure_public_bind=allow_insecure_public_bind,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        trusted=trusted,
        config=config,
    )
    session._serve_handle = handle
    auth_on = api_keys is not None or basic_auth is not None
    if config is not None and getattr(config, "auth_enabled", False):
        auth_on = True
    tls_on = ssl_certfile is not None or ssl_keyfile is not None
    session._record(
        "serve_bundle",
        {
            "path": str(resolved) if resolved is not None else None,
            "kind": kind,
            "host": host,
            "port": port,
            "auth": auth_on,
            "tls": tls_on,
            "docs_enabled": docs_enabled,
        },
        result_summary={"url": handle.url, "kind": kind, "auth": auth_on, "tls": tls_on},
        warnings=(
            (
                "Shared-secret auth (API-key and/or Basic) enabled; still not managed "
                "IAM. Docs default closed when auth is on. Terminate TLS at a reverse "
                "proxy for non-local exposure."
            )
            if auth_on
            else (
                "No authentication; localhost-oriented. Pass api_keys= / basic_auth= "
                "or use a reverse proxy for exposure."
            ),
        ),
    )
    return handle


def load_pretrained_backbone(
    session,
    modality: Literal["vision", "audio", "speech"],
    architecture: str | None = None,
    *,
    weights: Literal["none", "mock", "pretrained"] = "mock",
    freeze: bool = True,
    seed: int = 0,
    model_id: str | None = None,
) -> Any:
    """Load a curated pretrained vision/audio/speech backbone (integration hook).

    Delegates to :func:`buildml.dl.zoo.load_pretrained_backbone` and stores the
    backbone on the Session for downstream head attachment.

    Parameters
    ----------
    session:
        Active Session to attach the loaded backbone to.
    modality:
        Backbone modality (``vision``, ``audio``, or ``speech``).
    architecture:
        Optional architecture identifier within the curated zoo.
    weights:
        Weight source (``none``, ``mock``, or ``pretrained``).
    freeze:
        When True, freeze backbone parameters by default.
    seed:
        Seed for mock-weight initialization.
    model_id:
        Optional Hugging Face or zoo model identifier.

    Returns
    -------
    PretrainedBackbone
        Loaded backbone metadata and module shell.
    """
    from buildml.dl.zoo import load_pretrained_backbone as _load

    backbone = _load(
        modality,
        architecture,
        weights=weights,
        freeze=freeze,
        seed=seed,
        model_id=model_id,
    )
    session._dl_backbone = backbone
    session._record(
        "load_pretrained_backbone",
        {
            "modality": modality,
            "architecture": architecture,
            "weights": weights,
            "freeze": freeze,
            "model_id": model_id,
        },
        result_summary=backbone.to_dict(),
        warnings=tuple(backbone.warnings),
    )
    return backbone


def attach_backbone_head(
    session,
    n_classes: int,
    *,
    freeze_backbone: bool | None = None,
) -> Any:
    """Attach a classification head to the Session pretrained backbone.

    Delegates to :func:`buildml.dl.zoo.attach_backbone_head` using the backbone
    stored by :func:`load_pretrained_backbone`.

    Parameters
    ----------
    session:
        Active Session with a loaded pretrained backbone.
    n_classes:
        Number of output classes for the attached head.
    freeze_backbone:
        Optional override for whether the backbone stays frozen.

    Returns
    -------
    BackboneHeadBundle
        Combined backbone+head module metadata.

    Raises
    ------
    ValidationError
        When no backbone is loaded or ``n_classes`` is invalid.
    """
    from buildml.dl.zoo import attach_backbone_head as _attach

    backbone = getattr(session, "_dl_backbone", None)
    if backbone is None:
        raise ValidationError(
            "attach_backbone_head requires a prior load_pretrained_backbone(...)."
        )
    if int(n_classes) < 2:
        raise ValidationError("n_classes must be >= 2")
    head = _attach(
        backbone,
        n_classes=int(n_classes),
        freeze_backbone=freeze_backbone,
    )
    session._dl_backbone_head = head
    session._record(
        "attach_backbone_head",
        {
            "n_classes": int(n_classes),
            "freeze_backbone": freeze_backbone,
            "modality": getattr(backbone, "modality", None),
            "architecture": getattr(backbone, "architecture", None),
        },
        result_summary=head.to_dict() if hasattr(head, "to_dict") else {"n_classes": int(n_classes)},
        warnings=tuple(getattr(head, "warnings", ()) or ()),
    )
    return head


def evaluate_asr(
    session,
    *,
    hypotheses: list[str] | None = None,
    references: list[str],
    lowercase: bool = True,
) -> Any:
    """Score ASR hypotheses vs references (WER/CER); reuse last transcription texts.

    Delegates to :func:`buildml.dl.speech.evaluate_asr`. When ``hypotheses`` is
    omitted, reuses texts from the prior :func:`transcribe_speech` result.

    Parameters
    ----------
    session:
        Active Session that may hold a prior speech transcription result.
    hypotheses:
        Optional hypothesis strings; inferred from Session when omitted.
    references:
        Reference transcript strings aligned with hypotheses.
    lowercase:
        When True, lowercase strings before WER/CER scoring.

    Returns
    -------
    AsrEvalResult
        WER/CER metrics and scoring metadata.

    Raises
    ------
    ValidationError
        When hypotheses are missing and no transcription result exists.
    """
    from buildml.dl.speech import evaluate_asr as _eval

    hyps = hypotheses
    if hyps is None:
        speech = getattr(session, "_dl_speech_result", None)
        texts = None if speech is None else getattr(speech, "texts", None)
        if not texts:
            raise ValidationError(
                "evaluate_asr requires hypotheses= or a prior "
                "transcribe_speech result with texts."
            )
        hyps = list(texts)
    result = _eval(hypotheses=hyps, references=list(references), lowercase=lowercase)
    session._dl_asr_eval = result
    session._record(
        "evaluate_asr",
        {
            "n_hypotheses": len(hyps),
            "n_references": len(references),
            "lowercase": lowercase,
            "from_speech_result": hypotheses is None,
        },
        result_summary=(
            result.to_dict()
            if hasattr(result, "to_dict")
            else cast(dict[str, Any], getattr(result, "__dict__", {}))
        ),
        warnings=tuple(getattr(result, "warnings", ()) or ()),
    )
    return result


def pack_torchserve(
    session,
    output_dir: str | Path,
    *,
    torchscript_path: str | Path | None = None,
    model_name: str = "buildml_model",
) -> Any:
    """Pack a TorchScript artifact into a TorchServe-ready directory layout.

    Delegates to :func:`buildml.dl.packaging.pack_torchserve_model`. Uses the
    last TorchScript export on the Session when ``torchscript_path`` is omitted.

    Parameters
    ----------
    session:
        Active Session that may hold a prior TorchScript export result.
    output_dir:
        Destination directory for the TorchServe model store layout.
    torchscript_path:
        Optional explicit TorchScript artifact path.
    model_name:
        Model name used in the TorchServe manifest.

    Returns
    -------
    PackagingResult
        Output paths and packaging disclosures.

    Raises
    ------
    ValidationError
        When no TorchScript path is available.
    """
    from buildml.dl.packaging import pack_torchserve_model

    src = torchscript_path
    if src is None:
        export = getattr(session, "_dl_export_result", None)
        if export is not None and getattr(export, "format", None) == "torchscript":
            src = export.path
    if src is None:
        raise ValidationError(
            "pack_torchserve requires torchscript_path= or a prior "
            "export_torch(..., format='torchscript')."
        )
    result = pack_torchserve_model(src, output_dir, model_name=model_name)
    session._dl_packaging_result = result
    session._record(
        "pack_torchserve",
        {"output_dir": str(output_dir), "torchscript_path": str(src), "model_name": model_name},
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def prepare_tensorrt_export(
    session,
    output_dir: str | Path,
    *,
    onnx_path: str | Path | None = None,
    engine_name: str = "model.engine",
    fp16: bool = True,
) -> Any:
    """Write a TensorRT ``trtexec`` plan next to a validated ONNX artifact.

    Delegates to :func:`buildml.dl.packaging.prepare_tensorrt_export_plan`.
    Uses the last ONNX export on the Session when ``onnx_path`` is omitted.

    Parameters
    ----------
    session:
        Active Session that may hold a prior ONNX export result.
    output_dir:
        Destination directory for the TensorRT export plan.
    onnx_path:
        Optional explicit ONNX artifact path.
    engine_name:
        Output TensorRT engine filename.
    fp16:
        When True, request FP16 optimization in the export plan.

    Returns
    -------
    PackagingResult
        Export plan paths and limitation disclosures.

    Raises
    ------
    ValidationError
        When no ONNX path is available.
    """
    from buildml.dl.packaging import prepare_tensorrt_export_plan

    src = onnx_path
    if src is None:
        export = getattr(session, "_dl_export_result", None)
        if export is not None and getattr(export, "format", None) == "onnx":
            src = export.path
    if src is None:
        raise ValidationError(
            "prepare_tensorrt_export requires onnx_path= or a prior "
            "export_torch(..., format='onnx')."
        )
    result = prepare_tensorrt_export_plan(
        src, output_dir, engine_name=engine_name, fp16=fp16
    )
    session._dl_packaging_result = result
    session._record(
        "prepare_tensorrt_export",
        {
            "output_dir": str(output_dir),
            "onnx_path": str(src),
            "engine_name": engine_name,
            "fp16": fp16,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.warnings),
    )
    return result


def emit_k8s_ddp_job(
    session,
    path: str | Path,
    *,
    job_name: str = "buildml-torchrun-ddp",
    namespace: str = "default",
    image: str = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    nnodes: int = 2,
    nproc_per_node: int = 2,
    script_path: str = "/workspace/train.py",
    cpu_request: str = "2",
    memory_request: str = "4Gi",
    gpu_limit: int = 1,
    gpu_request: int | None = None,
    service_account: str | None = None,
    include_configmap: bool = True,
) -> Any:
    """Emit a Kubernetes Job YAML for torchrun multi-node DDP (template only).

    Delegates to :func:`buildml.dl.k8s.write_torchrun_ddp_job`. This writes a
    starter manifest: not a managed cluster orchestrator.

    Parameters
    ----------
    session:
        Active Session recording the emitted manifest metadata.
    path:
        Destination YAML file path.
    job_name:
        Kubernetes Job name.
    namespace:
        Kubernetes namespace.
    image:
        Container image for torchrun workers.
    nnodes:
        Number of nodes in the torchrun job.
    nproc_per_node:
        Processes launched per node.
    script_path:
        Training script path inside the container.
    cpu_request:
        CPU resource request per worker.
    memory_request:
        Memory resource request per worker.
    gpu_limit:
        GPU limit per worker.
    gpu_request:
        Optional GPU request per worker.
    service_account:
        Optional Kubernetes service account name.
    include_configmap:
        When True, include a starter ConfigMap manifest.

    Returns
    -------
    K8sManifestResult
        Written manifest paths and template limitations.
    """
    from buildml.dl.k8s import write_torchrun_ddp_job

    result = write_torchrun_ddp_job(
        path,
        job_name=job_name,
        namespace=namespace,
        image=image,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        script_path=script_path,
        cpu_request=cpu_request,
        memory_request=memory_request,
        gpu_limit=gpu_limit,
        gpu_request=gpu_request,
        service_account=service_account,
        include_configmap=include_configmap,
    )
    session._dl_k8s_result = result
    session._record(
        "emit_k8s_ddp_job",
        {
            "path": str(path),
            "job_name": job_name,
            "namespace": namespace,
            "nnodes": nnodes,
            "nproc_per_node": nproc_per_node,
            "cpu_request": cpu_request,
            "memory_request": memory_request,
            "gpu_limit": gpu_limit,
            "gpu_request": gpu_request,
            "service_account": service_account,
            "include_configmap": include_configmap,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.limitations),
    )
    return result


def emit_k8s_serve_deployment(
    session,
    path: str | Path,
    *,
    name: str = "buildml-serve",
    namespace: str = "default",
    image: str = "buildml-serve:local",
    replicas: int = 1,
    port: int = 8080,
    cpu_request: str = "1",
    memory_request: str = "2Gi",
    gpu_limit: int | None = None,
    service_account: str | None = None,
    bundle_path: str = "/models/bundle",
    kind: str = "pipeline",
    api_key_secret_name: str = "buildml-serve-secrets",
    api_key_secret_key: str = "api-key",
    include_secret: bool = True,
    trusted: bool = True,
) -> Any:
    """Emit a Kubernetes Deployment+Service YAML for managed serve (template only).

    Delegates to :func:`buildml.dl.k8s.write_serve_deployment`. Default image
    ``buildml-serve:local`` matches ``deploy/serve/Dockerfile``. Manifest uses
    a Secret for the API key and does not emit
    ``--allow-insecure-public-bind``. Template only: not a managed cluster
    orchestrator.

    Parameters
    ----------
    session:
        Active Session recording the emitted manifest metadata.
    path:
        Destination YAML file path.
    name:
        Deployment and Service name.
    namespace:
        Kubernetes namespace.
    image:
        Container image for the serve deployment.
    replicas:
        Desired replica count.
    port:
        Service/container port for managed serve.
    cpu_request:
        CPU resource request per replica.
    memory_request:
        Memory resource request per replica.
    gpu_limit:
        Optional GPU limit per replica.
    service_account:
        Optional Kubernetes service account name.
    bundle_path:
        In-container bundle path.
    kind:
        ``pipeline`` or ``torchscript``.
    api_key_secret_name:
        Secret name for ``BUILDML_API_KEY``.
    api_key_secret_key:
        Key inside the Secret.
    include_secret:
        Emit a placeholder Secret document when True.
    trusted:
        Pass ``--trusted`` in the rendered command when True.

    Returns
    -------
    K8sManifestResult
        Written manifest paths and template limitations.
    """
    from buildml.dl.k8s import write_serve_deployment

    result = write_serve_deployment(
        path,
        name=name,
        namespace=namespace,
        image=image,
        replicas=replicas,
        port=port,
        cpu_request=cpu_request,
        memory_request=memory_request,
        gpu_limit=gpu_limit,
        service_account=service_account,
        bundle_path=bundle_path,
        kind=kind,
        api_key_secret_name=api_key_secret_name,
        api_key_secret_key=api_key_secret_key,
        include_secret=include_secret,
        trusted=trusted,
    )
    session._dl_k8s_result = result
    session._record(
        "emit_k8s_serve_deployment",
        {
            "path": str(path),
            "name": name,
            "namespace": namespace,
            "image": image,
            "replicas": replicas,
            "port": port,
            "cpu_request": cpu_request,
            "memory_request": memory_request,
            "gpu_limit": gpu_limit,
            "service_account": service_account,
            "bundle_path": bundle_path,
            "kind": kind,
            "api_key_secret_name": api_key_secret_name,
            "include_secret": include_secret,
            "trusted": trusted,
        },
        result_summary=result.to_dict(),
        warnings=tuple(result.limitations),
    )
    return result


def domain_adapt_speech_torch(
    session,
    *,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    freeze_encoder: bool = True,
    audio_column: str | None = None,
    batch_size: int = 8,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
    normalize_audio: bool = True,
    encoder_dim: int = 64,
    seed: int = 0,
) -> Any:
    """Domain-adapt / finetune-lite speech classify (not FM continued pretrain).

    Alias of :func:`fit_speech_torch` with stronger domain-adapt disclosures
    recorded under the ``domain_adapt_speech_torch`` operation name.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    device:
        Compute device (``cpu``, ``cuda``, ``mps``, or ``auto``).
    freeze_encoder:
        When True, freeze the speech encoder during adaptation.
    audio_column:
        Optional audio column when loaders must be built.
    batch_size:
        Minibatch size when loaders must be built.
    sample_rate:
        Target sample rate when loaders must be built.
    max_samples:
        Maximum samples per example when loaders must be built.
    source_sample_rate:
        Optional source sample rate when loaders must be built.
    normalize_audio:
        When True, normalize audio amplitude on train only.
    encoder_dim:
        Encoder embedding dimension for the speech classifier.
    seed:
        Seed for shuffling and training.

    Returns
    -------
    Session
        ``session`` with ``dl_train_result`` attached for chaining.
    """
    from buildml.dl.speech import domain_adapt_speech_disclosures

    result = fit_speech_torch(
        session,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        freeze_encoder=freeze_encoder,
        audio_column=audio_column,
        batch_size=batch_size,
        sample_rate=sample_rate,
        max_samples=max_samples,
        source_sample_rate=source_sample_rate,
        normalize_audio=normalize_audio,
        encoder_dim=encoder_dim,
        seed=seed,
    )
    # Re-record under the domain-adapt name with stronger disclosures.
    disclosures = domain_adapt_speech_disclosures()
    session._record(
        "domain_adapt_speech_torch",
        {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "device": device,
            "freeze_encoder": freeze_encoder,
            "alias_of": "fit_speech_torch",
        },
        result_summary={
            "disclosures": list(disclosures),
            "train": None
            if session.dl_train_result is None
            else {"n_epochs_ran": session.dl_train_result.n_epochs_ran},
        },
        warnings=disclosures,
    )
    return result


def refuse_speech_foundation_pretrain(session) -> None:
    """Explicit refuse path for FM-from-scratch / large continued-pretrain asks.

    Records the refusal on the Session and delegates to
    :func:`buildml.dl.speech.refuse_foundation_model_pretrain` so callers get a
    clear, honest boundary instead of a silent no-op.

    Parameters
    ----------
    session:
        Active Session recording the refusal audit entry.
    """
    from buildml.dl.speech import refuse_foundation_model_pretrain

    session._record(
        "refuse_speech_foundation_pretrain",
        {"requested": "foundation_model_pretrain"},
        result_summary={"refused": True},
    )
    refuse_foundation_model_pretrain(requested="foundation_model_pretrain")
