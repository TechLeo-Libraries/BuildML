"""Deep-learning domain (Torch). Lazy imports — core BuildML never requires Torch."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "DDPConfig",
    "DDPTrainResult",
    "DLEvaluateResult",
    "DeviceSpec",
    "EarlyStopInfo",
    "ExportResult",
    "FeatureContract",
    "LoaderConfig",
    "LoaderReport",
    "MultimodalContract",
    "MultimodalLoaderConfig",
    "TabularMLP",
    "TextLoaderConfig",
    "TextVocab",
    "TorchCVResult",
    "TorchLoaderBundle",
    "TorchNestedCVResult",
    "TorchSearchResult",
    "TrainConfig",
    "TrainResult",
    "TrainingCurveReport",
    "apply_audio_waveform_stats",
    "apply_image_channel_stats",
    "build_multimodal_fusion",
    "build_tabular_mlp",
    "build_text_classifier",
    "build_training_curve",
    "cross_validate_torch",
    "ddp_cuda_device_count",
    "decode_audio_cell",
    "decode_image_cell",
    "evaluate_module",
    "export_module",
    "export_onnx",
    "export_torchscript",
    "export_train_result",
    "fit_audio_waveform_stats",
    "fit_image_channel_stats",
    "load_torch_bundle",
    "load_torchscript",
    "make_loaders",
    "make_multimodal_loaders",
    "make_text_loaders",
    "nested_cv_torch",
    "require_pillow",
    "require_soundfile",
    "require_torch",
    "save_torch_bundle",
    "search_torch",
    "smoke_load_onnx",
    "stack_audio_column",
    "stack_image_column",
    "torch_available",
    "torch_training_status",
    "train_supervised_module",
    "train_supervised_module_ddp",
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
    if name in {"TabularMLP", "build_tabular_mlp", "build_text_classifier"}:
        from buildml.dl import models as models_mod

        return getattr(models_mod, name)
    if name in {"TextLoaderConfig", "TextVocab", "make_text_loaders"}:
        from buildml.dl import text as text_mod

        return getattr(text_mod, name)
    if name in {"TorchCVResult", "cross_validate_torch"}:
        from buildml.dl import cv as cv_mod

        return getattr(cv_mod, name)
    if name in {
        "TorchNestedCVResult",
        "TorchSearchResult",
        "nested_cv_torch",
        "search_torch",
    }:
        from buildml.dl import search as search_mod

        return getattr(search_mod, name)
    if name in {
        "MultimodalContract",
        "MultimodalLoaderConfig",
        "build_multimodal_fusion",
        "make_multimodal_loaders",
    }:
        from buildml.dl import multimodal as mm_mod

        return getattr(mm_mod, name)
    if name in {
        "apply_image_channel_stats",
        "decode_image_cell",
        "fit_image_channel_stats",
        "require_pillow",
        "stack_image_column",
    }:
        from buildml.dl import image as image_mod

        return getattr(image_mod, name)
    if name in {
        "apply_audio_waveform_stats",
        "decode_audio_cell",
        "fit_audio_waveform_stats",
        "require_soundfile",
        "stack_audio_column",
    }:
        from buildml.dl import audio as audio_mod

        return getattr(audio_mod, name)
    if name in {
        "ExportResult",
        "export_module",
        "export_onnx",
        "export_torchscript",
        "export_train_result",
        "load_torchscript",
        "smoke_load_onnx",
    }:
        from buildml.dl import export as export_mod

        return getattr(export_mod, name)
    if name in {
        "DDPConfig",
        "DDPTrainResult",
        "ddp_cuda_device_count",
        "train_supervised_module_ddp",
    }:
        from buildml.dl import ddp as ddp_mod

        return getattr(ddp_mod, name)
    raise AttributeError(f"module 'buildml.dl' has no attribute {name!r}")
