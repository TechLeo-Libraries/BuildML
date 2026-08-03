"""Run the epoch loop that turns an untrained module into a trained one.

The loop itself is the standard one: for each epoch, pass over the training
data computing loss and stepping the optimiser, then optionally pass over the
validation data computing loss without stepping. What surrounds it is where the
care is.

**Validation is measured, never learned from.** The validation pass runs with no
optimiser, so no gradient reaches the weights. Early stopping reads that loss
and it alone: stopping on training loss would only detect that the model
stopped memorising, which tells you nothing about generalisation, and it is the
reason a validation partition is required rather than optional.

**Stopping early keeps the best epoch, not the last.** An early-stopped run
ends, by construction, on a run of epochs that were not improvements. Returning
those weights would hand back a model worse than one already seen, so the best
monitored epoch is snapshotted and restored.

**Resuming is checked, not assumed.** Continuing a run with different feature
columns, a different target, or a different task would produce a model that
trains without complaint and means nothing. All three are compared against the
saved contract and refused on mismatch. Optimiser and scheduler state are
restored too: Adam's momentum estimates are part of where training had reached,
and discarding them makes a resumed run stumble for several epochs.

Anything that changes silently is recorded as a warning rather than left
implicit: a device fallback, mixed precision disabled on non-CUDA hardware, a
scheduler that could not be restored.

The module is yours. This loop trains what you pass it and does not care how it
was built.

See Also
--------
buildml.dl.types.TrainConfig : Every setting the loop honours.
buildml.dl.results.TrainResult : What it hands back.
buildml.dl.zoo : Prebuilt modules to train.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

from buildml.core.errors import ValidationError
from buildml.dl.curves import build_training_curve
from buildml.dl.extras import require_torch
from buildml.dl.metrics import resolve_device
from buildml.dl.results import EarlyStopInfo, TorchLoaderBundle, TrainResult
from buildml.dl.types import TrainConfig

OptimizerFactory = Callable[[Iterable[Any]], Any]
LossFn = Callable[[Any, Any, Any], Any]


def _default_loss(task: str) -> LossFn:
    torch = require_torch(feature="Torch training")
    if task == "classification":
        criterion = torch.nn.CrossEntropyLoss()

        def loss_fn(module: Any, xb: Any, yb: Any) -> Any:
            return criterion(module(xb), yb)

        return loss_fn

    criterion = torch.nn.MSELoss()

    def loss_fn(module: Any, xb: Any, yb: Any) -> Any:
        return criterion(module(xb), yb)

    return loss_fn


def _default_optimizer_factory(learning_rate: float) -> OptimizerFactory:
    torch = require_torch(feature="Torch training")

    def factory(params: Iterable[Any]) -> Any:
        return torch.optim.Adam(params, lr=learning_rate)

    return factory


def _split_batch(batch: Any) -> tuple[Any, Any]:
    """Split a loader batch into (inputs, targets).

    Supports ``(xb, yb)`` and multimodal ``(x_tab, token_ids, yb)``.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValidationError("Loader batch must be (inputs..., y)")
    if len(batch) == 2:
        return batch[0], batch[1]
    return tuple(batch[:-1]), batch[-1]


def _batch_size(inputs: Any) -> int:
    if hasattr(inputs, "shape"):
        return int(inputs.shape[0])
    if isinstance(inputs, (tuple, list)) and inputs:
        return int(inputs[0].shape[0])
    raise ValidationError("Could not infer batch size from inputs")


def _to_device(obj: Any, device: Any) -> Any:
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return tuple(_to_device(x, device) for x in obj)
    return obj


def _epoch_loss(
    module: Any,
    loader: Any,
    loss_fn: LossFn,
    *,
    device: str,
    optimizer: Any | None = None,
    grad_clip_norm: float | None = None,
    scaler: Any | None = None,
    use_amp: bool = False,
) -> float:
    from contextlib import nullcontext

    torch = require_torch(feature="Torch training")
    train_mode = optimizer is not None
    module.train(train_mode)
    total = 0.0
    n = 0
    dev = torch.device(device)
    autocast_ctx = torch.cuda.amp.autocast(enabled=True) if use_amp else nullcontext()
    for batch in loader:
        xb, yb = _split_batch(batch)
        xb = _to_device(xb, dev)
        yb = _to_device(yb, dev)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss = loss_fn(module, xb, yb)
        if train_mode:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip_norm)
                optimizer.step()
        batch_n = _batch_size(xb)
        total += float(loss.detach().float().cpu()) * batch_n
        n += batch_n
    return total / max(n, 1)


def _current_lr(optimizer: Any) -> float:
    groups = getattr(optimizer, "param_groups", None) or []
    if not groups:
        return float("nan")
    return float(groups[0].get("lr", float("nan")))


def _build_scheduler(optimizer: Any, cfg: TrainConfig, *, epochs_this_call: int) -> Any | None:
    torch = require_torch(feature="Torch training")
    name = (cfg.scheduler or "none").lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(cfg.scheduler_step_size)),
            gamma=float(cfg.scheduler_gamma),
        )
    if name == "cosine":
        if cfg.scheduler_t_max is not None:
            t_max = int(cfg.scheduler_t_max)
        else:
            t_max = max(1, epochs_this_call)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if name == "plateau":
        mode = cfg.early_stopping_mode if cfg.early_stopping_patience else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(cfg.scheduler_factor),
            patience=max(0, int(cfg.scheduler_patience)),
            threshold=float(cfg.scheduler_threshold),
        )
    raise ValidationError(
        f"Unknown scheduler {cfg.scheduler!r}. Use none, step, plateau, or cosine."
    )


def _is_improvement(
    value: float,
    best: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < (best - min_delta)
    if mode == "max":
        return value > (best + min_delta)
    raise ValidationError(f"Unknown early_stopping_mode {mode!r}; use min or max.")


def _validate_resume(prior: TrainResult, loader_bundle: TorchLoaderBundle) -> None:
    if prior.contract.feature_columns != loader_bundle.contract.feature_columns:
        raise ValidationError(
            "Resume refused: feature columns differ from the saved trainer contract."
        )
    if prior.contract.target_column != loader_bundle.contract.target_column:
        raise ValidationError(
            "Resume refused: target column differs from the saved trainer contract."
        )
    if prior.task != loader_bundle.contract.task:
        raise ValidationError(
            f"Resume refused: task mismatch ({prior.task} vs {loader_bundle.contract.task})."
        )


def train_supervised_module(
    module: Any,
    loader_bundle: TorchLoaderBundle,
    *,
    config: TrainConfig | None = None,
    loss_fn: LossFn | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    resume_from: TrainResult | None = None,
) -> TrainResult:
    """Train a module for a number of epochs, watching validation as it goes.

    Moves the module to the resolved device, builds an optimiser and any
    scheduler, then runs the epoch loop: training pass, optional validation
    pass, early-stopping check, scheduler step: until the epochs are exhausted
    or patience runs out.

    Parameters
    ----------
    module:
        Any ``torch.nn.Module``. Built by :mod:`buildml.dl.zoo` or by you; this
        loop does not care which.
    loader_bundle:
        The data loaders plus the feature contract that describes them. Must
        contain a ``'train'`` loader; a ``'validation'`` loader enables
        validation loss and early stopping.
    config:
        Training settings. Defaults to a plain :class:`~buildml.dl.types.TrainConfig`.
    loss_fn:
        A callable ``(module, inputs, targets) -> loss``. Defaults to cross
        entropy for classification and mean squared error for regression. Pass
        your own for a custom objective: the signature takes the module rather
        than its output so that multi-output or auxiliary-loss models are
        expressible.
    optimizer_factory:
        A callable taking parameters and returning an optimiser. Defaults to
        Adam at ``config.learning_rate``.
    resume_from:
        A prior result to continue from, typically loaded from a bundle. Weights,
        optimiser state, and scheduler state are restored, and history is
        appended rather than replaced. ``config.epochs`` then means *additional*
        epochs.

    Returns
    -------
    TrainResult
        The trained module together with its history, device, contract,
        optimiser and scheduler state, early-stopping record, and any warnings.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If the bundle has no train loader, if ``epochs`` is below 1, if early
        stopping is requested without a validation loader or with patience
        below 1, if the monitored metric is not among the recorded ones, if the
        scheduler name is unknown, or if a resume disagrees with the saved
        contract.

    Notes
    -----
    **Early stopping requires a validation partition, and that is not a
    technicality.** Training loss almost always keeps falling; stopping on it
    would fire only once the model stopped memorising. Validation loss is what
    turns upward when generalisation starts to degrade, which is the moment
    early stopping exists to catch.

    **Resuming compares contracts before it restores anything.** Different
    feature columns, a different target, or a different task all mean the saved
    weights no longer describe the problem, and continuing would train happily
    while meaning nothing. Each is refused with a message naming the mismatch.

    **``mixed_precision=True`` off CUDA is a no-op with a warning.** It is not
    an error: the same configuration should run on a laptop and a GPU box: but
    the speedup will not be there, and silence about that would be misleading.

    Examples
    --------
    >>> result = train_supervised_module(  # doctest: +SKIP
    ...     module,
    ...     bundle,
    ...     config=TrainConfig(epochs=20, early_stopping_patience=3),
    ... )
    >>> result.early_stop.triggered, result.early_stop.best_epoch  # doctest: +SKIP
    (True, 12)
    >>> more = train_supervised_module(  # doctest: +SKIP
    ...     result.module, bundle, config=TrainConfig(epochs=5), resume_from=result
    ... )

    See Also
    --------
    buildml.dl.types.TrainConfig : Every setting, and when to change it.
    buildml.dl.checkpoint : Saving a result so it can be resumed later.
    """
    torch = require_torch(feature="Torch training")
    if "train" not in loader_bundle.loaders:
        raise ValidationError("TorchLoaderBundle has no train loader")
    cfg = config or TrainConfig()
    if cfg.epochs < 1:
        raise ValidationError("epochs must be >= 1")

    task = loader_bundle.contract.task
    device_spec = resolve_device(cfg.device)
    warnings = list(loader_bundle.report.warnings)
    if device_spec.fallback_warning:
        warnings.append(device_spec.fallback_warning)

    prior = resume_from
    if prior is not None:
        _validate_resume(prior, loader_bundle)
        module.load_state_dict(prior.module.state_dict())
        if prior.device.resolved != device_spec.resolved:
            warnings.append(
                f"Resume device {device_spec.resolved!r} differs from prior "
                f"{prior.device.resolved!r}; optimizer state was remapped via load."
            )

    module = module.to(torch.device(device_spec.resolved))
    resolved_loss = loss_fn or _default_loss(task)
    factory = optimizer_factory or _default_optimizer_factory(cfg.learning_rate)
    optimizer = factory(module.parameters())
    if prior is not None and prior.optimizer_state is not None:
        try:
            optimizer.load_state_dict(prior.optimizer_state)
        except Exception as exc:  # noqa: BLE001: surface as ValidationError
            raise ValidationError(
                f"Could not restore optimizer state for resume: {exc}"
            ) from exc

    use_amp = bool(cfg.mixed_precision) and str(device_spec.resolved).startswith("cuda")
    scaler = None
    if cfg.mixed_precision and not use_amp:
        warnings.append(
            "mixed_precision=True but device is not CUDA; AMP disabled (CPU/MPS no-op)."
        )
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        warnings.append(f"CUDA AMP enabled on device {device_spec.resolved}.")

    scheduler = _build_scheduler(optimizer, cfg, epochs_this_call=cfg.epochs)
    if prior is not None and prior.scheduler_state is not None and scheduler is not None:
        if (prior.scheduler_name or "none") == (cfg.scheduler or "none"):
            try:
                scheduler.load_state_dict(prior.scheduler_state)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Could not restore scheduler state; continuing fresh: {exc}")
        else:
            warnings.append(
                f"Scheduler changed from {prior.scheduler_name!r} to {cfg.scheduler!r}; "
                "starting a new scheduler."
            )

    history: list[dict[str, float]] = list(prior.history) if prior is not None else []
    start_epoch = int(prior.n_epochs_ran) if prior is not None else 0
    train_loader = loader_bundle.loaders["train"]
    val_loader = loader_bundle.loaders.get("validation")

    patience = cfg.early_stopping_patience
    early_enabled = patience is not None
    if early_enabled and val_loader is None:
        raise ValidationError(
            "early_stopping_patience requires a validation DataLoader. "
            "Create a validation partition (split/group_split/time_split) before fit_torch."
        )
    if early_enabled and patience is not None and patience < 1:
        raise ValidationError("early_stopping_patience must be >= 1 when enabled")

    monitor = cfg.early_stopping_monitor
    mode = cfg.early_stopping_mode
    min_delta = float(cfg.early_stopping_min_delta)
    best_value: float | None = None
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    wait = 0
    triggered = False
    stop_reason = f"completed_epochs:{cfg.epochs}"
    stopped_epoch = start_epoch + cfg.epochs

    # Seed best from prior early-stop bookkeeping when resuming.
    if prior is not None and prior.early_stop is not None and prior.early_stop.enabled:
        best_value = prior.early_stop.best_value
        best_epoch = prior.early_stop.best_epoch
        if cfg.restore_best_weights:
            best_state = deepcopy(prior.module.state_dict())

    for offset in range(1, cfg.epochs + 1):
        epoch = start_epoch + offset
        train_loss = _epoch_loss(
            module,
            train_loader,
            resolved_loss,
            device=device_spec.resolved,
            optimizer=optimizer,
            grad_clip_norm=cfg.grad_clip_norm,
            scaler=scaler,
            use_amp=use_amp,
        )
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "lr": _current_lr(optimizer),
        }
        if val_loader is not None:
            row["val_loss"] = _epoch_loss(
                module,
                val_loader,
                resolved_loss,
                device=device_spec.resolved,
                optimizer=None,
                use_amp=use_amp,
            )

        if early_enabled:
            if monitor not in row:
                raise ValidationError(
                    f"early_stopping_monitor {monitor!r} missing from epoch metrics "
                    f"(available: {sorted(k for k in row if k != 'epoch')}). "
                    "Use val_loss when a validation loader exists."
                )
            metric_value = float(row[monitor])
            if _is_improvement(metric_value, best_value, mode=mode, min_delta=min_delta):
                best_value = metric_value
                best_epoch = epoch
                wait = 0
                if cfg.restore_best_weights:
                    best_state = deepcopy(module.state_dict())
            else:
                wait += 1

        if scheduler is not None:
            if cfg.scheduler == "plateau":
                if monitor in row:
                    plateau_metric = row[monitor]
                else:
                    plateau_metric = row.get("val_loss", train_loss)
                scheduler.step(float(plateau_metric))
            else:
                scheduler.step()

        if cfg.log_every <= 1 or epoch % cfg.log_every == 0 or offset == cfg.epochs:
            history.append(row)

        if early_enabled and wait >= int(patience or 0):
            triggered = True
            stopped_epoch = epoch
            stop_reason = (
                f"early_stopping: no improvement in {monitor} for {patience} epoch(s) "
                f"(best={best_value} at epoch {best_epoch} on validation)"
            )
            break
    else:
        stopped_epoch = start_epoch + cfg.epochs
        if early_enabled:
            stop_reason = (
                f"completed_epochs:{cfg.epochs} without early-stop trigger "
                f"(best {monitor}={best_value} at epoch {best_epoch})"
            )

    if early_enabled and cfg.restore_best_weights and best_state is not None:
        module.load_state_dict(best_state)

    early_info = EarlyStopInfo(
        enabled=bool(early_enabled),
        triggered=triggered,
        monitor=monitor,
        mode=mode,
        patience=patience,
        best_epoch=best_epoch,
        best_value=best_value,
        stopped_epoch=stopped_epoch,
        restore_best_weights=bool(cfg.restore_best_weights) if early_enabled else False,
        partition="validation",
        reason=stop_reason,
    )

    mm_contract = getattr(loader_bundle, "multimodal_contract", None)
    mm_preprocess = None if mm_contract is None else dict(mm_contract.to_dict())
    result = TrainResult(
        module=module,
        task=task,
        config=cfg,
        device=device_spec,
        contract=loader_bundle.contract,
        optimizer_state=optimizer.state_dict(),
        history=history,
        n_train_rows=loader_bundle.report.n_train,
        n_epochs_ran=stopped_epoch,
        warnings=warnings,
        early_stop=early_info,
        scheduler_name=str(cfg.scheduler or "none"),
        scheduler_state=None if scheduler is None else scheduler.state_dict(),
        resumed_from_epochs=start_epoch,
        multimodal_preprocess=mm_preprocess,
    )
    result.training_curve = build_training_curve(result)
    return result
