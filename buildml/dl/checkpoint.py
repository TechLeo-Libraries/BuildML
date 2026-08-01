"""Torch trainer bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.dl.curves import build_training_curve
from buildml.dl.extras import require_torch
from buildml.dl.results import EarlyStopInfo, TrainResult
from buildml.dl.types import DeviceSpec, FeatureContract, TrainConfig

BUNDLE_FORMAT = "buildml.torch_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Torch trainer bundles and Session checkpoints are complementary, not interchangeable. "
    "A trainer bundle stores module weights, optimizer state, TrainConfig, epoch history, "
    "and the feature/label contract (buildml.torch_bundle.v1). "
    "A Session checkpoint stores data, roles, splits, history, and optional classical plans; "
    "it does not embed Torch weights. Reload data via checkpoint_load; reload weights via "
    "load_torch_bundle. Resume training with fit_torch(..., resume=True) after load_torch_bundle."
)


@dataclass(slots=True)
class TorchBundle:
    """Loaded trainer artifact (module must be supplied by the caller on load)."""

    train_result: TrainResult
    meta: dict[str, Any]


def _meta_from_result(result: TrainResult) -> dict[str, Any]:
    return {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "module": type(result.module).__name__,
        "task": result.task,
        "config": result.config.to_dict(),
        "device": result.device.to_dict(),
        "contract": result.contract.to_dict(),
        "n_train_rows": result.n_train_rows,
        "n_epochs_ran": result.n_epochs_ran,
        "history": list(result.history),
        "warnings": list(result.warnings),
        "early_stop": None if result.early_stop is None else result.early_stop.to_dict(),
        "scheduler_name": result.scheduler_name,
        "resumed_from_epochs": result.resumed_from_epochs,
        "compatibility": CHECKPOINT_BOUNDARY,
    }


def save_torch_bundle(path: str | Path, train_result: TrainResult) -> Path:
    """Write a trainer bundle directory.

    Layout
    ------
    ``meta.json`` (``buildml.torch_bundle.v1``) and ``trainer.pt`` (torch.save dict).
    """
    torch = require_torch(feature="Torch trainer bundle save")
    if train_result is None:
        raise ValidationError("No TrainResult to save")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "module_state": train_result.module.state_dict(),
        "optimizer_state": train_result.optimizer_state,
        "scheduler_state": train_result.scheduler_state,
        "scheduler_name": train_result.scheduler_name,
        "config": train_result.config.to_dict(),
        "device": train_result.device.to_dict(),
        "contract": train_result.contract.to_dict(),
        "history": list(train_result.history),
        "task": train_result.task,
        "n_train_rows": train_result.n_train_rows,
        "n_epochs_ran": train_result.n_epochs_ran,
        "warnings": list(train_result.warnings),
        "early_stop": None
        if train_result.early_stop is None
        else train_result.early_stop.to_dict(),
        "resumed_from_epochs": train_result.resumed_from_epochs,
        "module_class": type(train_result.module).__name__,
    }
    torch.save(payload, destination / "trainer.pt")
    meta = _meta_from_result(train_result)
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_torch_bundle(
    path: str | Path,
    module: Any,
    *,
    map_location: str | None = None,
) -> TrainResult:
    """Load weights/config into a caller-supplied module shell.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``trainer.pt``.
    module:
        Uninitialized or compatible ``nn.Module`` instance that will receive
        ``load_state_dict``.
    map_location:
        Optional torch device string for deserialization (defaults to CPU).
    """
    torch = require_torch(feature="Torch trainer bundle load")
    root = Path(path)
    meta_path = root / "meta.json"
    trainer_path = root / "trainer.pt"
    if not meta_path.is_file() or not trainer_path.is_file():
        raise ValidationError(
            f"Incomplete Torch trainer bundle at {root}. "
            "Expected meta.json and trainer.pt (buildml.torch_bundle.v1)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported trainer bundle format {fmt!r}; expected {BUNDLE_FORMAT}. "
            "This is not a Session checkpoint or classical pipeline bundle."
        )

    location = map_location or "cpu"
    payload = torch.load(trainer_path, map_location=location, weights_only=False)
    if payload.get("format") != BUNDLE_FORMAT:
        raise ValidationError(
            f"trainer.pt format mismatch: {payload.get('format')!r} (expected {BUNDLE_FORMAT})"
        )

    module.load_state_dict(payload["module_state"])
    cfg = TrainConfig(**payload["config"])
    device_payload = payload["device"]
    device = DeviceSpec(
        requested=device_payload["requested"],
        resolved=device_payload["resolved"],
        fallback_warning=device_payload.get("fallback_warning"),
    )
    contract = FeatureContract.from_dict(payload["contract"])
    module = module.to(torch.device(location))
    early_payload = payload.get("early_stop")
    early = None if early_payload is None else EarlyStopInfo.from_dict(early_payload)
    result = TrainResult(
        module=module,
        task=payload["task"],
        config=cfg,
        device=device,
        contract=contract,
        optimizer_state=payload.get("optimizer_state"),
        history=list(payload.get("history") or []),
        n_train_rows=int(payload.get("n_train_rows") or 0),
        n_epochs_ran=int(payload.get("n_epochs_ran") or 0),
        warnings=list(payload.get("warnings") or []),
        early_stop=early,
        scheduler_name=str(payload.get("scheduler_name") or cfg.scheduler or "none"),
        scheduler_state=payload.get("scheduler_state"),
        resumed_from_epochs=int(payload.get("resumed_from_epochs") or 0),
    )
    result.training_curve = build_training_curve(result)
    return result
