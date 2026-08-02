"""Thin Session facades over buildml.dl."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


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
    """Return 'multimodal' / 'text' when auto tabular rebuild would be wrong."""
    modality = getattr(module, "modality", None) or ""
    layout = getattr(module, "input_layout", None)
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

    Classical preprocess: Session ``impute`` / ``encode`` / ``scale`` already
    mutate the attached frame with train-fitted plans. Attached plans are
    disclosed on the loader report. Pass ``apply_plans=True`` to explicitly
    re-apply fitted plans via :meth:`apply_preprocess_plans` before building
    loaders (score-time replay; does not refit).

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
) -> Session:
    """Train an ``nn.Module`` on the train Torch loader.

    Requires ``pip install 'buildml[torch]'``. When ``module`` is omitted, builds
    a tabular MLP, text classifier, or multimodal fusion module from the active
    loader contract so the happy path does not require a hand-rolled network.

    Does not replace classical :meth:`fit` / :attr:`fit_result`.
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
            from buildml.dl.multimodal import build_multimodal_fusion

            text_vocab = getattr(session._torch_loaders, "text_vocab", None)
            multimodal = getattr(session._torch_loaders, "multimodal_contract", None)
            contract = session._torch_loaders.contract
            modality = getattr(session._torch_loaders, "modality", None) or ""
            is_multimodal = multimodal is not None or str(modality).endswith("_fusion")
            if is_multimodal:
                mm = multimodal
                if mm is None:
                    raise ValidationError(
                        "Multimodal loaders are missing multimodal_contract for fit_torch"
                    )
                n_classes = max(2, len(contract.class_labels) or 2)
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
                n_classes = max(2, len(contract.class_labels) or 2)
                module = build_text_classifier(
                    text_vocab.vocab_size,
                    n_classes=n_classes,
                    dropout=dropout,
                )
            else:
                in_features = len(contract.feature_columns)
                n_classes = max(2, len(contract.class_labels) or 2)
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
    return session


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
    a limitation unless you supply a custom factory path — this helper does not
    silently refit Session-global plans inside each fold.
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
    """
    from buildml.dl.checkpoint import save_torch_bundle

    if session._dl_train_result is None:
        raise ValidationError("No Torch trainer. Call fit_torch(...) first.")
    destination = save_torch_bundle(path, session._dl_train_result)
    session._record("save_torch_bundle", {"path": str(destination)})
    return destination


def load_torch_bundle(
    session, path: str | Path, module: Any, *, map_location: str | None = None
) -> Session:
    """Load a Torch trainer bundle into this Session.

    Parameters
    ----------
    path:
        Bundle directory with ``meta.json`` and ``trainer.pt``.
    module:
        Compatible ``nn.Module`` shell that receives ``load_state_dict``.
    map_location:
        Optional device for ``torch.load`` (default CPU).
    """
    from buildml.dl.checkpoint import load_torch_bundle

    session._dl_train_result = load_torch_bundle(path, module, map_location=map_location)
    session._record(
        "load_torch_bundle",
        {"path": str(path), "module": type(module).__name__, "map_location": map_location},
        result_summary=session._dl_train_result.to_dict(),
    )
    return session


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
) -> Any:
    """Build fused multimodal DataLoaders (tabular/text/image/audio mixes).

    Requires ``buildml[torch]``. Fit stats (vocab, numeric mean/std, image
    channel mean/std, audio amplitude mean/std) use the train partition only.
    Batches follow ``(numeric?, tokens?, image?, audio?, y)`` for present
    modalities. Audio fusion is a small 1D-CNN branch — not a speech foundation
    model.
    """
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
    Torch alone. Uses a small 1D-CNN fusion branch — not a speech foundation
    model.
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
    Alpha-quality escape hatch — see export result limitations.
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
    config: Any | None = None,
) -> Any:
    """Single-node DDP training via a fresh ``module_factory`` per process.

    Requires ``torch.cuda.device_count() >= 2`` unless ``allow_cpu_ddp=True``
    (gloo smoke only). Multi-node cluster launch is out of scope.
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
        ddp_config=DDPConfig(world_size=world_size, allow_cpu_ddp=allow_cpu_ddp),
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
        },
        result_summary=ddp_result.to_dict(),
        warnings=tuple(ddp_result.warnings),
    )
    return ddp_result
