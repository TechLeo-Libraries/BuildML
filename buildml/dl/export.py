"""TorchScript / ONNX export helpers for fitted BuildML Torch modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import require_torch
from buildml.dl.results import TrainResult

ExportFormat = Literal["torchscript", "onnx"]


@dataclass(slots=True)
class ExportResult:
    """Outcome of exporting a Torch module to TorchScript or ONNX."""

    path: Path
    format: ExportFormat
    opset: int | None
    example_input_shapes: tuple[tuple[int, ...], ...]
    dynamic_axes: dict[str, dict[int, str]]
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "opset": self.opset,
            "example_input_shapes": [list(s) for s in self.example_input_shapes],
            "dynamic_axes": {k: dict(v) for k, v in self.dynamic_axes.items()},
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }


def _example_from_loader(loader: Any) -> Any:
    batch = next(iter(loader))
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValidationError("Loader batch must be (inputs..., y)")
    inputs = batch[:-1]
    if len(inputs) == 1:
        return inputs[0]
    return tuple(inputs)


def _shapes_of(example: Any) -> tuple[tuple[int, ...], ...]:
    if hasattr(example, "shape"):
        return (tuple(int(x) for x in example.shape),)
    if isinstance(example, (tuple, list)):
        return tuple(tuple(int(x) for x in t.shape) for t in example)
    raise ValidationError(f"Unsupported example input type: {type(example).__name__}")


def resolve_example_input(
    *,
    module: Any | None = None,
    train_result: TrainResult | None = None,
    loader_bundle: Any | None = None,
    example_input: Any | None = None,
) -> Any:
    """Resolve a batch of example inputs for tracing / ONNX export."""
    if example_input is not None:
        return example_input
    if loader_bundle is not None and getattr(loader_bundle, "loaders", None):
        train_loader = loader_bundle.loaders.get("train")
        if train_loader is not None:
            return _example_from_loader(train_loader)
    raise ValidationError(
        "Export requires example_input or a loader_bundle with a train loader "
        "(pass Session torch loaders or an explicit tensor / tuple)."
    )


def export_torchscript(
    module: Any,
    path: str | Path,
    *,
    example_input: Any,
    method: Literal["trace", "script"] = "trace",
) -> ExportResult:
    """Export an ``nn.Module`` to TorchScript (``.pt``)."""
    torch = require_torch(feature="TorchScript export")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval()
    warnings: list[str] = []
    if method == "script":
        try:
            scripted = torch.jit.script(module)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"torch.jit.script failed: {exc}") from exc
    else:
        try:
            with torch.no_grad():
                scripted = torch.jit.trace(module, example_input, strict=False)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"torch.jit.trace failed: {exc}") from exc
        warnings.append("TorchScript used trace; control-flow outside the example path is frozen.")
    scripted.save(str(destination))
    shapes = _shapes_of(example_input)
    return ExportResult(
        path=destination,
        format="torchscript",
        opset=None,
        example_input_shapes=shapes,
        dynamic_axes={},
        disclosures=(
            f"TorchScript export via jit.{method} to {destination.name}.",
            "Reload with torch.jit.load in a matching Torch major version.",
        ),
        limitations=(
            "TorchScript is an alpha escape hatch — not a full serving product.",
            "Dynamic Python control flow and data-dependent shapes may not transfer.",
        ),
        warnings=tuple(warnings),
        meta={"method": method, "module": type(module).__name__},
    )


def export_onnx(
    module: Any,
    path: str | Path,
    *,
    example_input: Any,
    opset: int = 17,
    dynamic_batch: bool = True,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> ExportResult:
    """Export an ``nn.Module`` to ONNX.

    Requires ``buildml[onnx]`` (the ``onnx`` package). On Torch ≥2.9 the default
    dynamo exporter also wants ``onnxscript``; BuildML uses the legacy
    TorchScript-based exporter (``dynamo=False``) so multimodal dual-arg modules
    export without that extra dependency.
    """
    torch = require_torch(feature="ONNX export")
    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MissingExtraError(
            "onnx",
            "ONNX export (pip install 'buildml[onnx]' or pip install onnx)",
        ) from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval()

    if isinstance(example_input, (tuple, list)):
        args = tuple(example_input)
        default_names = [f"input_{i}" for i in range(len(args))]
    else:
        args = (example_input,)
        default_names = ["input"]
    names = input_names or default_names
    if len(names) != len(args):
        raise ValidationError("input_names length must match example_input arity")
    outs = output_names or ["output"]

    dynamic_axes: dict[str, dict[int, str]] = {}
    if dynamic_batch:
        for name in names:
            dynamic_axes[name] = {0: "batch"}
        dynamic_axes[outs[0]] = {0: "batch"}

    warnings: list[str] = []
    export_kwargs: dict[str, Any] = {
        "input_names": names,
        "output_names": outs,
        "dynamic_axes": dynamic_axes or None,
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    # Torch 2.9+ defaults dynamo=True (needs onnxscript). Prefer legacy path.
    try:
        import inspect

        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
            warnings.append(
                "ONNX export used dynamo=False (legacy TorchScript exporter) "
                "for BuildML alpha compatibility."
            )
    except (TypeError, ValueError):  # pragma: no cover - extremely defensive
        pass

    try:
        with torch.no_grad():
            if "dynamo" in export_kwargs:
                torch.onnx.export(module, args, str(destination), **export_kwargs)
            else:
                torch.onnx.export(
                    module,
                    args if len(args) > 1 else args[0],
                    str(destination),
                    **export_kwargs,
                )
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"torch.onnx.export failed: {exc}") from exc

    try:
        model = onnx.load(str(destination))
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"onnx.checker reported issues: {exc}")

    shapes = _shapes_of(example_input if not isinstance(example_input, tuple) else example_input)
    if isinstance(example_input, (tuple, list)):
        shapes = tuple(tuple(int(x) for x in t.shape) for t in example_input)

    return ExportResult(
        path=destination,
        format="onnx",
        opset=int(opset),
        example_input_shapes=shapes,
        dynamic_axes=dynamic_axes,
        disclosures=(
            f"ONNX export at opset={opset} to {destination.name}.",
            "dynamic_axes batch dim enabled." if dynamic_batch else "Static batch axes.",
        ),
        limitations=(
            "ONNX export is alpha-quality: opset coverage varies by operator and Torch version.",
            "Not a managed inference server; validate with your runtime "
            "(onnxruntime, TensorRT, …).",
            "Multimodal tuple inputs export as multiple ONNX graph inputs.",
        ),
        warnings=tuple(warnings),
        meta={
            "module": type(module).__name__,
            "input_names": names,
            "output_names": outs,
        },
    )


def export_module(
    module: Any,
    path: str | Path,
    *,
    format: ExportFormat = "torchscript",
    example_input: Any,
    opset: int = 17,
    dynamic_batch: bool = True,
    torchscript_method: Literal["trace", "script"] = "trace",
) -> ExportResult:
    """Export ``module`` to TorchScript or ONNX."""
    if format == "torchscript":
        return export_torchscript(
            module, path, example_input=example_input, method=torchscript_method
        )
    if format == "onnx":
        return export_onnx(
            module,
            path,
            example_input=example_input,
            opset=opset,
            dynamic_batch=dynamic_batch,
        )
    raise ValidationError(f"Unknown export format {format!r}; use torchscript or onnx")


def export_train_result(
    train_result: TrainResult,
    path: str | Path,
    *,
    format: ExportFormat = "torchscript",
    loader_bundle: Any | None = None,
    example_input: Any | None = None,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> ExportResult:
    """Export a :class:`TrainResult` module with example inputs from loaders."""
    example = resolve_example_input(
        train_result=train_result,
        loader_bundle=loader_bundle,
        example_input=example_input,
    )
    # Move example to CPU for portable artifacts.
    torch = require_torch(feature="Torch export")
    device = torch.device("cpu")
    module = train_result.module.to(device).eval()

    def _cpu(obj: Any) -> Any:
        if hasattr(obj, "to"):
            return obj.to(device)
        if isinstance(obj, (tuple, list)):
            return tuple(_cpu(x) for x in obj)
        return obj

    example = _cpu(example)
    result = export_module(
        module,
        path,
        format=format,
        example_input=example,
        opset=opset,
        dynamic_batch=dynamic_batch,
    )
    result.meta["task"] = train_result.task
    result.meta["contract"] = train_result.contract.to_dict()
    return result


def load_torchscript(path: str | Path, *, map_location: str = "cpu") -> Any:
    """Load a TorchScript module saved by :func:`export_torchscript`."""
    torch = require_torch(feature="TorchScript load")
    return torch.jit.load(str(path), map_location=map_location)


def smoke_load_onnx(path: str | Path) -> dict[str, Any]:
    """Optional ONNX file smoke check (requires ``onnx`` package)."""
    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MissingExtraError(
            "onnx",
            "ONNX smoke check requires the onnx package. "
            "Install with: pip install 'buildml[onnx]' or pip install onnx",
        ) from exc
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return {
        "ir_version": int(model.ir_version),
        "opset": [int(o.version) for o in model.opset_import],
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
    }
