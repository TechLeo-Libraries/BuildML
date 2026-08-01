"""Deep-learning domain (Torch). Lazy imports — core BuildML never requires Torch."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "DLEvaluateResult",
    "DeviceSpec",
    "EarlyStopInfo",
    "FeatureContract",
    "LoaderConfig",
    "LoaderReport",
    "TorchLoaderBundle",
    "TrainConfig",
    "TrainResult",
    "TrainingCurveReport",
    "build_training_curve",
    "evaluate_module",
    "load_torch_bundle",
    "make_loaders",
    "require_torch",
    "save_torch_bundle",
    "torch_available",
    "torch_training_status",
    "train_supervised_module",
]


def __getattr__(name: str) -> Any:
    if name in {"require_torch", "torch_available"}:
        from buildml.dl import extras

        return getattr(extras, name)
    if name in {
        "DeviceSpec",
        "FeatureContract",
        "LoaderConfig",
        "TrainConfig",
    }:
        from buildml.dl import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "DLEvaluateResult",
        "EarlyStopInfo",
        "LoaderReport",
        "TorchLoaderBundle",
        "TrainResult",
        "TrainingCurveReport",
    }:
        from buildml.dl import results

        return getattr(results, name)
    if name in {"build_training_curve", "torch_training_status"}:
        from buildml.dl import curves

        return getattr(curves, name)
    if name == "make_loaders":
        from buildml.dl.loaders import make_loaders

        return make_loaders
    if name == "train_supervised_module":
        from buildml.dl.train import train_supervised_module

        return train_supervised_module
    if name == "evaluate_module":
        from buildml.dl.metrics import evaluate_module

        return evaluate_module
    if name in {"BUNDLE_FORMAT", "CHECKPOINT_BOUNDARY", "save_torch_bundle", "load_torch_bundle"}:
        from buildml.dl import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    raise AttributeError(f"module 'buildml.dl' has no attribute {name!r}")
