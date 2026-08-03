"""Save and reload a Torch training run, weights and all.

BuildML has two kinds of persistence and confusing them wastes an afternoon. A
**Session checkpoint** stores data, roles, splits, history, and classical
preprocessing plans: the state of your analysis. A **trainer bundle**, which is
what this module writes, stores module weights, optimiser state, the training
configuration, the epoch history, and the feature contract: the state of your
model. Neither contains the other, and a full restore usually needs both.

A bundle is a directory with two files. ``meta.json`` is human-readable and
holds everything that serialises to JSON, so you can inspect what a bundle
contains without loading Torch at all. ``trainer.pt`` holds the tensors.

Loading requires you to supply the module instance. The bundle stores weights,
not architecture: reconstructing a class from a file would mean executing code
from that file, and a model bundle is exactly the sort of artifact that gets
passed around. You build the module; the bundle fills it in.

Optimiser and scheduler state are saved because resuming without them is worse
than it sounds. Adam's momentum estimates take many steps to rebuild, so a
resume that drops them stumbles for several epochs before recovering.

See Also
--------
buildml.dl.train : Producing the result this saves.
buildml.dl.export : Deployment artifacts, a different job.
"""

from __future__ import annotations

from buildml.core.serialization import require_trusted_deserialize

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
    "the feature/label contract, and optional multimodal_preprocess meta "
    "(image/audio stats, sample rates, layout) under buildml.torch_bundle.v1. "
    "Load restores that meta for inspection but does not rebuild DataLoaders or "
    "auto-apply media preprocess: remake multimodal/text loaders explicitly. "
    "A Session checkpoint stores data, roles, splits, history, and optional classical plans; "
    "it does not embed Torch weights. Reload data via checkpoint_load; reload weights via "
    "load_torch_bundle. Resume training with fit_torch(..., resume=True) after load_torch_bundle."
)

_MULTIMODAL_LOAD_WARNING = (
    "Loaded multimodal_preprocess meta from the trainer bundle for honesty "
    "(frozen image/audio/text stats and layout). DataLoaders were not rebuilt; "
    "call make_multimodal_torch_loaders / make_image_multimodal_torch_loaders / "
    "make_audio_multimodal_torch_loaders again before fit/evaluate/export. "
    "Remaking loaders re-fits train-only stats from the current Session frame."
)


@dataclass(slots=True)
class TorchBundle:
    """A loaded trainer bundle: the restored run plus its metadata.

    Attributes
    ----------
    train_result:
        The reconstructed run, with weights already loaded into the module you
        supplied.
    meta:
        The parsed ``meta.json``: format version, BuildML version, module
        class name, and the JSON-safe view of everything the bundle holds.

    Notes
    -----
    ``meta`` is worth reading before trusting a restore. The recorded module
    class name tells you whether the shell you supplied matches what was saved,
    and the BuildML version tells you which release wrote it.

    See Also
    --------
    load_torch_bundle : Produces this.
    """

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
        "multimodal_preprocess": result.multimodal_preprocess,
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
    """Write a training run to disk so it can be resumed or reloaded.

    Creates the directory if needed and writes two files: ``trainer.pt`` with
    the tensors, and ``meta.json`` with everything readable without Torch.

    Parameters
    ----------
    path:
        Directory to write to. Created if absent; existing files are
        overwritten.
    train_result:
        The run to save.

    Returns
    -------
    pathlib.Path
        The directory written.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If no result was supplied.

    Notes
    -----
    **The architecture is not saved: only its weights.** Reloading needs the
    same module class, constructed the same way. Keep the code that builds it
    alongside the bundle, or the weights are a directory of numbers with no
    shape to fit.

    **``meta.json`` is designed to be read directly.** Checking which columns a
    saved model expects, or how many epochs it ran, does not require loading
    Torch or the weights.

    Examples
    --------
    Save after training, then inspect without Torch::

        save_torch_bundle("artifacts/run-01", train_result)

        import json
        meta = json.loads(open("artifacts/run-01/meta.json").read())
        meta["contract"]["feature_columns"]

    See Also
    --------
    load_torch_bundle : The other half.
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
        "multimodal_preprocess": train_result.multimodal_preprocess,
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
    trusted: bool = False,
) -> TrainResult:
    """Restore a saved run into a module you construct.

    Reads the bundle, loads the weights into your module, and rebuilds the
    surrounding :class:`~buildml.dl.results.TrainResult`: configuration,
    device record, contract, history, early-stop record, and optimiser and
    scheduler state, so training can resume where it left off.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``trainer.pt``.
    module:
        A freshly constructed module of the same architecture. Its weights are
        replaced by the saved ones, so however it was initialised does not
        matter: but its shape must match.
    map_location:
        Where to deserialise tensors. Defaults to CPU, which loads correctly
        regardless of what the run trained on; move the module afterwards if
        you want it elsewhere.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    TrainResult
        The restored run, with the training curve rebuilt from the history.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If either file is missing, or if the format marker is not
        ``buildml.torch_bundle.v1``: which normally means the path points at a
        Session checkpoint or a classical pipeline bundle instead.

    Notes
    -----
    **A shape mismatch surfaces as a Torch error from ``load_state_dict``**,
    naming the layers that disagree. That is usually enough to identify the
    constructor argument that changed since the bundle was written.

    **Multimodal preprocessing metadata is restored for inspection only.**
    Frozen image and audio statistics, sample rates, and layout come back on the
    result, but no DataLoaders are rebuilt and no media transforms are applied.
    Rebuild loaders explicitly before fitting, evaluating, or exporting: and
    note that doing so re-fits train-only statistics from the current frame,
    which is why the difference is disclosed in ``warnings`` rather than left
    implicit.

    **Defaulting to CPU is deliberate.** A bundle saved from CUDA loads fine on
    a CPU-only machine this way; mapping straight to a device that is not there
    would fail instead.

    Examples
    --------
    Reconstruct and resume::

        module = MyNetwork(n_features=12, n_classes=3)
        result = load_torch_bundle("artifacts/run-01", module)
        result.n_epochs_ran

    See Also
    --------
    save_torch_bundle : The other half.
    buildml.dl.train.train_supervised_module : Resuming with ``resume=True``.
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
    require_trusted_deserialize(trusted=trusted, artifact='torch payload', path=path)

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
    mm_preprocess = payload.get("multimodal_preprocess")
    if mm_preprocess is None:
        mm_preprocess = meta.get("multimodal_preprocess")
    if mm_preprocess is not None:
        mm_preprocess = dict(mm_preprocess)
    warnings = list(payload.get("warnings") or [])
    if mm_preprocess is not None and _MULTIMODAL_LOAD_WARNING not in warnings:
        warnings.append(_MULTIMODAL_LOAD_WARNING)
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
        warnings=warnings,
        early_stop=early,
        scheduler_name=str(payload.get("scheduler_name") or cfg.scheduler or "none"),
        scheduler_state=payload.get("scheduler_state"),
        resumed_from_epochs=int(payload.get("resumed_from_epochs") or 0),
        multimodal_preprocess=mm_preprocess,
    )
    result.training_curve = build_training_curve(result)
    return result
