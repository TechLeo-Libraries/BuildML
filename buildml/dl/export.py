"""Convert a trained module into an artifact something else can run.

A Torch model in memory needs Python, PyTorch, and your model class to make a
prediction. Serving usually cannot assume any of those. Export produces a
self-contained artifact that carries the computation graph along with the
weights.

Two formats, with different reach. **TorchScript** stays inside the Torch
ecosystem and is the higher-fidelity option — it handles the widest range of
model code and reloads with ``torch.jit.load``. **ONNX** is the interchange
format, readable by onnxruntime, TensorRT, and other engines, at the cost of
narrower operator coverage: not every Torch operation has an ONNX equivalent at
every opset.

Both export by tracing, which is where the important caveat lives. Tracing runs
the model once on an example input and records the operations that executed. Any
branch not taken during that run is not in the artifact. A model whose behaviour
depends on its input values will export cleanly and then be wrong for inputs that
would have taken a different path — no error, just a different answer. Use
``method="script"`` for TorchScript when control flow matters.

Export produces a file. It does not serve it, and validating the artifact against
your actual runtime is a step nothing here can do for you.

See Also
--------
buildml.dl.packaging : Wrapping these artifacts for a serving stack.
buildml.dl.checkpoint : Saving for resumption rather than deployment.
"""

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
    """What an export produced, and what the artifact cannot be assumed to do.

    Attributes
    ----------
    path:
        The file written.
    format:
        ``'torchscript'`` or ``'onnx'``.
    opset:
        The ONNX operator set version. ``None`` for TorchScript.
    example_input_shapes:
        The shapes traced with. Anything not marked dynamic is baked in.
    dynamic_axes:
        Which dimensions may vary at inference, by input name. Normally just
        the batch dimension.
    disclosures:
        What was exported and how.
    limitations:
        What the artifact does not guarantee.
    warnings:
        Notably that tracing froze the control flow, or that the ONNX checker
        found something.
    meta:
        Module class name, input and output names, and the task and contract
        when exported from a training result.

    Notes
    -----
    **``example_input_shapes`` tells you what the artifact expects.** Every
    dimension outside ``dynamic_axes`` is fixed at the traced value, so a model
    traced on 32-length sequences will not accept 64-length ones.

    See Also
    --------
    export_torchscript : Produces this for TorchScript.
    export_onnx : Produces this for ONNX.
    """

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
        """Return the export outcome as JSON-safe values.

        Includes the traced shapes and dynamic axes, which are what a consumer
        of the artifact needs to know before feeding it anything.

        Returns
        -------
        dict
            Path as a string, format, opset, input shapes as lists, dynamic
            axes, the three prose lists, and metadata.
        """
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
    """Find a representative input batch to trace the model with.

    Tracing needs real tensors of the right shape and dtype. This uses an
    explicit example when given one, and otherwise pulls the first batch from
    the training loader.

    Parameters
    ----------
    module:
        Accepted for signature symmetry; not currently used.
    train_result:
        Accepted for signature symmetry; not currently used.
    loader_bundle:
        Loaders to take the first training batch from.
    example_input:
        An explicit tensor or tuple of tensors, which takes priority.

    Returns
    -------
    Any
        A single tensor for single-input models, or a tuple for multimodal
        ones. Targets are stripped from the loader batch.

    Raises
    ------
    ValidationError
        If neither an example nor a usable loader bundle is available, or if
        the loader's batches are not shaped as ``(inputs..., y)``.

    Notes
    -----
    **The example's shape defines the artifact's expectations.** A batch of 16
    traces to an artifact that assumes 16 unless the batch axis is marked
    dynamic — which the ONNX path does by default.

    See Also
    --------
    export_train_result : The usual caller.
    """
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
    """Save a module as TorchScript, runnable without your model class.

    Produces a ``.pt`` file containing both the computation graph and the
    weights. Reload it with ``torch.jit.load`` anywhere Torch is available — no
    Python source, no class definition.

    Parameters
    ----------
    module:
        The module to export. Switched to evaluation mode first, which disables
        dropout and freezes batch-norm statistics.
    path:
        Where to write. Parent directories are created.
    example_input:
        A representative batch. Required for tracing; unused when scripting.
    method:
        ``'trace'`` runs the model once and records what happened.
        ``'script'`` compiles the source, preserving control flow.

    Returns
    -------
    ExportResult
        The path, the traced shapes, and the limitations.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If tracing or scripting fails, with the underlying Torch error
        included.

    Notes
    -----
    **Tracing records one execution path and discards the rest.** If your
    forward pass branches on input values, the artifact contains only the branch
    the example took. It will run on any input and quietly give the wrong answer
    for those that should have gone elsewhere. Scripting preserves branches, at
    the cost of only supporting a subset of Python.

    **Evaluation mode is applied and baked in.** Dropout is off and batch-norm
    uses its running statistics, which is what inference should do — and is not
    reversible in the exported artifact.

    **Reload with a compatible Torch major version.** TorchScript is not
    guaranteed to load across major releases.

    Examples
    --------
    Trace, then confirm it reloads::

        result = export_torchscript(module, "artifacts/model.pt", example_input=batch)
        loaded = load_torchscript(result.path, trusted=True)

    See Also
    --------
    export_onnx : The cross-framework alternative.
    load_torchscript : Reading the result back.
    """
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
    """Save a module as ONNX, runnable outside PyTorch entirely.

    Produces a graph in the open ONNX format, which onnxruntime, TensorRT, and
    other engines can execute — often faster than PyTorch, and without the
    PyTorch dependency. The exported model is validated with the ONNX checker
    when possible.

    Parameters
    ----------
    module:
        The module to export. Switched to evaluation mode first.
    path:
        Where to write. Parent directories are created.
    example_input:
        A representative batch. A tuple becomes multiple graph inputs.
    opset:
        ONNX operator set version. Higher supports more operations; your runtime
        has to support the version you pick.
    dynamic_batch:
        Mark the batch dimension as variable, so the artifact accepts any batch
        size rather than only the traced one.
    input_names / output_names:
        Names for the graph's inputs and outputs. Defaulted when omitted.

    Returns
    -------
    ExportResult
        The path, opset, traced shapes, dynamic axes, and any checker
        complaints in ``warnings``.

    Raises
    ------
    MissingExtraError
        If PyTorch or the ``onnx`` package is not installed. Install with
        ``pip install buildml[onnx]``.
    ValidationError
        If the export fails, or if ``input_names`` does not match the number of
        inputs.

    Notes
    -----
    **Enable ``dynamic_batch`` unless you have a reason not to.** Without it the
    artifact accepts exactly the traced batch size, which makes it useless for
    single-row requests.

    **This uses the legacy TorchScript-based exporter.** Torch 2.9 and later
    default to a dynamo-based exporter that requires ``onnxscript``; BuildML
    passes ``dynamo=False`` so multimodal modules with multiple inputs export
    without that additional dependency. A warning records the choice.

    **Operator coverage is genuinely uneven.** An operation with no ONNX
    equivalent at your chosen opset fails the export outright, which is at least
    a loud failure. Raising the opset often resolves it, if your runtime keeps
    up.

    **A checker complaint is a warning, not an error.** Some models the strict
    checker objects to run correctly in practice, so the export completes and
    the objection is recorded — but verify against your actual runtime.

    Examples
    --------
    Export with a variable batch dimension::

        result = export_onnx(
            module, "artifacts/model.onnx", example_input=batch, opset=17,
        )
        result.warnings  # empty when the checker was satisfied

    See Also
    --------
    export_torchscript : Higher fidelity, Torch only.
    smoke_load_onnx : Inspecting the result.
    buildml.dl.packaging.prepare_tensorrt_export_plan : The next step for TensorRT.
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
    input_names: list[str] | None = None,
) -> ExportResult:
    """Export to either format, chosen by argument.

    A dispatcher over :func:`export_torchscript` and :func:`export_onnx`, useful
    when the format is configuration rather than a decision made in code.

    Parameters
    ----------
    module:
        The module to export.
    path:
        Where to write.
    format:
        ``'torchscript'`` or ``'onnx'``.
    example_input:
        A representative batch.
    opset:
        ONNX opset version. Ignored for TorchScript.
    dynamic_batch:
        Mark the batch dimension variable. ONNX only.
    torchscript_method:
        ``'trace'`` or ``'script'``. TorchScript only.
    input_names:
        Graph input names. ONNX only.

    Returns
    -------
    ExportResult
        Whatever the chosen exporter returned.

    Raises
    ------
    MissingExtraError
        If a required package is not installed.
    ValidationError
        If the format is unrecognised, or propagated from the exporter.

    See Also
    --------
    export_torchscript : The TorchScript path, and its caveats.
    export_onnx : The ONNX path, and its caveats.
    """
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
            input_names=input_names,
        )
    raise ValidationError(f"Unknown export format {format!r}; use torchscript or onnx")


def _layout_input_names(module: Any, example: Any) -> list[str] | None:
    """Prefer multimodal ``input_layout`` names for ONNX graph inputs."""
    layout = getattr(module, "input_layout", None)
    if not layout:
        return None
    names = [str(x) for x in layout]
    if isinstance(example, (tuple, list)):
        if len(names) != len(example):
            return None
        return names
    if len(names) == 1:
        return names
    return None


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
    """Export a trained model, taking the example batch from its loaders.

    The convenient entry point. Pulls an example from the loaders, moves both
    the module and the example to CPU so the artifact is portable, names
    multimodal inputs after their modalities, and attaches the task and feature
    contract to the result's metadata.

    Parameters
    ----------
    train_result:
        The training outcome to export.
    path:
        Where to write.
    format:
        ``'torchscript'`` or ``'onnx'``.
    loader_bundle:
        Loaders to take the example batch from.
    example_input:
        An explicit example, overriding the loaders.
    opset:
        ONNX opset version.
    dynamic_batch:
        Mark the batch dimension variable. ONNX only.

    Returns
    -------
    ExportResult
        With ``meta`` carrying the task, the feature contract, and the input
        layout for multimodal models.

    Raises
    ------
    MissingExtraError
        If a required package is not installed.
    ValidationError
        If no example can be resolved, or propagated from the exporter.

    Notes
    -----
    **Everything is moved to CPU first, and this mutates the training result's
    module.** A CUDA-traced artifact is tied to a GPU being present; a
    CPU-traced one loads anywhere and can be moved afterwards. The side effect
    is worth knowing about if you plan to keep training.

    **Multimodal inputs are named after the layout** — ``numeric``, ``tokens``,
    ``image``, ``audio`` — rather than ``input_0`` and friends, which makes the
    resulting graph considerably easier to wire up correctly.

    **The contract travels in ``meta``, and it is the thing that makes the
    artifact usable.** It records the column order and the normalisation
    constants, without which a caller cannot construct a valid input.

    Examples
    --------
    Export the trained model with its loaders::

        result = export_train_result(
            train_result, "artifacts/model.onnx",
            format="onnx", loader_bundle=bundle,
        )
        result.meta["contract"]["feature_columns"]

    See Also
    --------
    export_module : The lower-level dispatcher.
    buildml.dl.packaging : Wrapping the artifact for serving.
    """
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
    layout_names = _layout_input_names(module, example)
    # Prefer bundle layout when module attribute is missing/mismatched.
    if layout_names is None and loader_bundle is not None:
        bundle_layout = getattr(loader_bundle, "input_layout", None)
        if bundle_layout and isinstance(example, (tuple, list)):
            names = [str(x) for x in bundle_layout]
            if len(names) == len(example):
                layout_names = names
    result = export_module(
        module,
        path,
        format=format,
        example_input=example,
        opset=opset,
        dynamic_batch=dynamic_batch,
        input_names=layout_names,
    )
    result.meta["task"] = train_result.task
    result.meta["contract"] = train_result.contract.to_dict()
    if layout_names is not None:
        result.meta["input_layout"] = list(layout_names)
    return result


def load_torchscript(
    path: str | Path,
    *,
    map_location: str = "cpu",
    trusted: bool = False,
) -> Any:
    """Load a TorchScript artifact back into a runnable module.

    Reads a file written by :func:`export_torchscript`. No model class needed —
    the graph is in the file. Requires ``trusted=True`` because TorchScript
    load can execute code depending on Torch version / settings.

    Parameters
    ----------
    path:
        The TorchScript file.
    map_location:
        Where to place the weights. Defaults to CPU, which works regardless of
        what the artifact was exported from.
    trusted:
        Must be ``True`` to deserialize. Pass only for artifacts you created
        or fully trust. Defaults to ``False``.

    Returns
    -------
    torch.jit.ScriptModule
        Callable like the original module, already in evaluation mode.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        When ``trusted`` is false or ``path`` is not a local filesystem path.

    Notes
    -----
    **The loaded module has no feature contract.** It will accept any tensor of
    the right shape and produce numbers, including for inputs that were never
    preprocessed the way training data was. Keep the contract alongside the
    artifact.

    See Also
    --------
    export_torchscript : Writing the file.
    """
    from buildml.core.serialization import (
        assert_local_load_path,
        require_trusted_deserialize,
    )

    target = assert_local_load_path(path, artifact="TorchScript module")
    require_trusted_deserialize(
        trusted=trusted, artifact="TorchScript module", path=target
    )
    torch = require_torch(feature="TorchScript load")
    return torch.jit.load(str(target), map_location=map_location)


def smoke_load_onnx(path: str | Path) -> dict[str, Any]:
    """Check that an ONNX file is valid and report its interface.

    Loads the model, runs the structural checker, and returns the details a
    caller needs to wire it up: which opsets it uses, and the names of its
    inputs and outputs.

    Parameters
    ----------
    path:
        The ONNX file.

    Returns
    -------
    dict
        IR version, opset versions, input names, and output names.

    Raises
    ------
    MissingExtraError
        If the ``onnx`` package is not installed. Install with
        ``pip install buildml[onnx]``.
    onnx.checker.ValidationError
        If the model is structurally invalid.

    Notes
    -----
    **A structural check, not a functional one.** It confirms the graph is
    well-formed; it does not run the model or verify that its outputs match
    what PyTorch produced. Compare predictions between the original and the
    exported model on a handful of rows before trusting the artifact.

    Examples
    --------
    Confirm the interface before writing a client::

        info = smoke_load_onnx("artifacts/model.onnx")
        info["inputs"], info["outputs"]

    See Also
    --------
    export_onnx : Writing the file.
    """
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
