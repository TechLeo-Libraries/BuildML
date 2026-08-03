"""Deep-learning domain (Torch). Lazy imports: core BuildML never requires Torch."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "DDPConfig",
    "DDPTrainResult",
    "DLEvaluateResult",
    "DeviceSpec",
    "DistributedEnv",
    "EarlyStopInfo",
    "ExportResult",
    "FeatureContract",
    "LoaderConfig",
    "LoaderReport",
    "MultimodalContract",
    "MultimodalLoaderConfig",
    "SpeechContract",
    "SpeechLoaderConfig",
    "SpeechTranscribeResult",
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
    "build_speech_classifier",
    "build_tabular_mlp",
    "build_text_classifier",
    "build_tiny_speech_encoder",
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
    "AsrEvalResult",
    "attach_backbone_head",
    "BackboneHeadResult",
    "evaluate_asr",
    "freeze_module",
    "list_pretrained_backbones",
    "load_audio_backbone",
    "load_pretrained_backbone",
    "load_speech_backbone",
    "load_torch_bundle",
    "load_torchscript",
    "load_vision_backbone",
    "make_loaders",
    "make_multimodal_loaders",
    "make_speech_loaders",
    "make_text_loaders",
    "nested_cv_torch",
    "pack_torchserve_model",
    "parse_torchrun_env",
    "prepare_tensorrt_export_plan",
    "refuse_foundation_model_pretrain",
    "render_serve_deployment",
    "render_torchrun_ddp_job",
    "require_pillow",
    "require_soundfile",
    "require_speech_stack",
    "require_torch",
    "save_torch_bundle",
    "search_torch",
    "smoke_load_onnx",
    "speech_stack_available",
    "stack_audio_column",
    "stack_image_column",
    "torch_available",
    "torch_training_status",
    "train_supervised_module",
    "train_supervised_module_ddp",
    "transcribe_audio_values",
    "transcribe_from_dataset",
    "write_serve_deployment",
    "write_torchrun_ddp_job",
    "PackagingResult",
    "PretrainedBackbone",
    "K8sJobRenderResult",
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
        "DistributedEnv",
        "ddp_cuda_device_count",
        "parse_torchrun_env",
        "train_supervised_module_ddp",
    }:
        from buildml.dl import ddp as ddp_mod

        return getattr(ddp_mod, name)
    if name in {
        "AsrEvalResult",
        "SpeechContract",
        "SpeechLoaderConfig",
        "SpeechTranscribeResult",
        "build_speech_classifier",
        "build_tiny_speech_encoder",
        "evaluate_asr",
        "make_speech_loaders",
        "refuse_foundation_model_pretrain",
        "require_speech_stack",
        "speech_stack_available",
        "transcribe_audio_values",
        "transcribe_from_dataset",
    }:
        from buildml.dl import speech as speech_mod

        return getattr(speech_mod, name)
    if name in {
        "BackboneHeadResult",
        "PretrainedBackbone",
        "attach_backbone_head",
        "freeze_module",
        "list_pretrained_backbones",
        "load_audio_backbone",
        "load_pretrained_backbone",
        "load_speech_backbone",
        "load_vision_backbone",
    }:
        from buildml.dl import zoo as zoo_mod

        return getattr(zoo_mod, name)
    if name in {
        "PackagingResult",
        "TORCHSERVE_COMPOSE_EXAMPLE",
        "pack_torchserve_model",
        "prepare_tensorrt_export_plan",
    }:
        from buildml.dl import packaging as packaging_mod

        return getattr(packaging_mod, name)
    if name in {
        "K8sJobRenderResult",
        "render_serve_deployment",
        "render_torchrun_ddp_job",
        "write_serve_deployment",
        "write_torchrun_ddp_job",
    }:
        from buildml.dl import k8s as k8s_mod

        return getattr(k8s_mod, name)
    raise AttributeError(f"module 'buildml.dl' has no attribute {name!r}")
