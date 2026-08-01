"""CPU-first supervised train loop for caller-supplied modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch
from buildml.dl.metrics import resolve_device
from buildml.dl.results import TorchLoaderBundle, TrainResult
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


def _epoch_loss(
    module: Any,
    loader: Any,
    loss_fn: LossFn,
    *,
    device: str,
    optimizer: Any | None = None,
    grad_clip_norm: float | None = None,
) -> float:
    torch = require_torch(feature="Torch training")
    train_mode = optimizer is not None
    module.train(train_mode)
    total = 0.0
    n = 0
    dev = torch.device(device)
    for xb, yb in loader:
        xb = xb.to(dev)
        yb = yb.to(dev)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(module, xb, yb)
        if train_mode:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip_norm)
            optimizer.step()
        batch_n = int(xb.shape[0])
        total += float(loss.detach().cpu()) * batch_n
        n += batch_n
    return total / max(n, 1)


def train_supervised_module(
    module: Any,
    loader_bundle: TorchLoaderBundle,
    *,
    config: TrainConfig | None = None,
    loss_fn: LossFn | None = None,
    optimizer_factory: OptimizerFactory | None = None,
) -> TrainResult:
    """Run a supervised epoch loop on the train loader; optional val loss each epoch."""
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

    module = module.to(torch.device(device_spec.resolved))
    resolved_loss = loss_fn or _default_loss(task)
    factory = optimizer_factory or _default_optimizer_factory(cfg.learning_rate)
    optimizer = factory(module.parameters())

    history: list[dict[str, float]] = []
    train_loader = loader_bundle.loaders["train"]
    val_loader = loader_bundle.loaders.get("validation")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _epoch_loss(
            module,
            train_loader,
            resolved_loss,
            device=device_spec.resolved,
            optimizer=optimizer,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        row: dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}
        if val_loader is not None:
            row["val_loss"] = _epoch_loss(
                module,
                val_loader,
                resolved_loss,
                device=device_spec.resolved,
                optimizer=None,
            )
        if cfg.log_every <= 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            history.append(row)

    return TrainResult(
        module=module,
        task=task,
        config=cfg,
        device=device_spec,
        contract=loader_bundle.contract,
        optimizer_state=optimizer.state_dict(),
        history=history,
        n_train_rows=loader_bundle.report.n_train,
        n_epochs_ran=cfg.epochs,
        warnings=warnings,
    )
