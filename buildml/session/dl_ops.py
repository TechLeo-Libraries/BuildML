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
) -> Session:
    """Train an ``nn.Module`` on the train Torch loader.

    Requires ``pip install 'buildml[torch]'``. When ``module`` is omitted, builds
    a tabular MLP (or text classifier when the last loaders were text) from the
    loader contract so the happy path does not require a hand-rolled network.

    Does not replace classical :meth:`fit` / :attr:`fit_result`.
    """
    from buildml.dl.models import build_tabular_mlp, build_text_classifier
    from buildml.dl.train import train_supervised_module
    from buildml.dl.types import TrainConfig

    session.assert_can_fit("train")
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
            text_vocab = getattr(session._torch_loaders, "text_vocab", None)
            contract = session._torch_loaders.contract
            if text_vocab is not None:
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
