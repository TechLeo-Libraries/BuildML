"""Thin Session facades over buildml.dl (no new DL depth)."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


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
    task: Literal['classification', 'regression', 'auto'] = "auto",
) -> Any:
    """Build Torch DataLoaders from current roles and split partitions.

    Requires ``pip install 'buildml[torch]'`` (or ``buildml[dl]``). Shuffle
    applies to the train loader only. When ``normalize`` is True, mean/std
    are fit on train and frozen for validation/test. Classical preprocess
    plans are not auto-applied; call them first if needed.

    Returns
    -------
    TorchLoaderBundle
        Loaders keyed by partition plus the feature contract.
    """
    from buildml.dl.loaders import make_loaders
    from buildml.dl.types import LoaderConfig

    session.assert_can_fit("train")
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
        },
        result_summary=bundle.report.to_dict(),
        warnings=tuple(bundle.report.warnings),
    )
    return bundle


def fit_torch(
    session,
    module: Any,
    *,
    loss_fn: Any | None = None,
    optimizer_factory: Any | None = None,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: Literal['cpu', 'cuda', 'mps', 'auto'] = "auto",
    grad_clip_norm: float | None = None,
    log_every: int = 1,
    early_stopping_patience: int | None = None,
    early_stopping_monitor: str = "val_loss",
    scheduler: Literal['none', 'step', 'plateau', 'cosine'] = "none",
    resume: bool = False,
    config: Any | None = None,
) -> Session:
    """Train a caller-supplied ``nn.Module`` on the train Torch loader.

    Requires ``pip install 'buildml[torch]'``. Delegates to
    :func:`buildml.dl.train.train_supervised_module`. Does not replace
    classical :meth:`fit` / :attr:`fit_result`.

    Parameters
    ----------
    module:
        Unfitted (or warm) ``torch.nn.Module``. When ``resume=True``, weights
        are restored from :attr:`dl_train_result` before continuing.
    loss_fn:
        Optional ``(module, xb, yb) -> loss``. Defaults to CrossEntropy
        (classification) or MSE (regression).
    optimizer_factory:
        Optional ``callable(params) -> optimizer``. Defaults to Adam.
    epochs / learning_rate / device / grad_clip_norm / log_every:
        Train-loop knobs used when ``config`` is omitted. With ``resume=True``,
        ``epochs`` are **additional** epochs.
    early_stopping_patience / early_stopping_monitor / scheduler:
        M2 knobs when ``config`` is omitted. Patience requires a validation
        loader. Scheduler defaults to ``none`` (see :class:`~buildml.dl.types.TrainConfig`).
    resume:
        When True, continue from :attr:`dl_train_result` (e.g. after
        :meth:`load_torch_bundle`), restoring optimizer/scheduler state.
    config:
        Optional :class:`~buildml.dl.types.TrainConfig` overriding the
        scalar knobs above.
    """
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
    partition: Literal['train', 'validation', 'test'] = "test",
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
