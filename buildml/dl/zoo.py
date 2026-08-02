"""Pretrained vision / audio / speech backbone hooks (integration, not a full zoo).

Loads ResNet/ViT-class (torchvision), Wav2Vec-class (transformers), and
Whisper-encoder-class (transformers) backbones with freeze/finetune helpers.

Weight modes
------------
* ``none`` — architecture only (random init; no download).
* ``mock`` — architecture + tiny random compatible tensors (CI-safe).
* ``pretrained`` — real published weights (may download; needs network/extra).

This is **not** a Hugging Face / TorchVision catalog product and does **not**
train foundation models from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import require_torch

WeightMode = Literal["none", "mock", "pretrained"]
VisionArch = Literal["resnet18", "vit_b_16"]
AudioArch = Literal["wav2vec2_base"]
SpeechArch = Literal["whisper_tiny_encoder"]


@dataclass(slots=True)
class PretrainedBackbone:
    """Loaded backbone module plus honest disclosures."""

    module: Any
    modality: Literal["vision", "audio", "speech"]
    architecture: str
    weight_mode: WeightMode
    frozen: bool
    feature_dim: int
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "architecture": self.architecture,
            "weight_mode": self.weight_mode,
            "frozen": self.frozen,
            "feature_dim": self.feature_dim,
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
            "module": type(self.module).__name__,
        }


def freeze_module(module: Any, *, freeze: bool = True) -> Any:
    """Freeze or unfreeze all parameters on ``module`` (in-place)."""
    for param in module.parameters():
        param.requires_grad = not freeze
    return module


def _apply_mock_weights(module: Any, *, seed: int = 0) -> None:
    torch = require_torch(feature="pretrained zoo mock weights")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim >= 2:
                torch.nn.init.xavier_uniform_(param, generator=generator)
            else:
                torch.nn.init.zeros_(param)


def _require_torchvision() -> Any:
    try:
        import torchvision  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MissingExtraError(
            "torch",
            "Vision backbones need torchvision (pip install torchvision alongside buildml[torch])",
        ) from exc
    return torchvision


def _require_transformers() -> Any:
    try:
        import transformers  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MissingExtraError(
            "speech",
            "Audio/speech transformer backbones need transformers (pip install 'buildml[speech]')",
        ) from exc
    return transformers


def load_vision_backbone(
    architecture: VisionArch = "resnet18",
    *,
    weights: WeightMode = "mock",
    freeze: bool = True,
    seed: int = 0,
) -> PretrainedBackbone:
    """Load a torchvision vision backbone as a feature extractor.

    Parameters
    ----------
    architecture:
        ``resnet18`` or ``vit_b_16``.
    weights:
        ``none`` / ``mock`` (default, CI-safe) / ``pretrained`` (may download).
    freeze:
        When True, all backbone parameters have ``requires_grad=False``.
    """
    torch = require_torch(feature="vision backbone")
    tv = _require_torchvision()
    warnings: list[str] = []
    if architecture == "resnet18":
        weights_enum = None
        if weights == "pretrained":
            weights_enum = tv.models.ResNet18_Weights.DEFAULT
        model = tv.models.resnet18(weights=weights_enum)
        feature_dim = int(model.fc.in_features)
        model.fc = torch.nn.Identity()
    elif architecture == "vit_b_16":
        weights_enum = None
        if weights == "pretrained":
            weights_enum = tv.models.ViT_B_16_Weights.DEFAULT
        model = tv.models.vit_b_16(weights=weights_enum)
        feature_dim = int(model.heads.head.in_features)
        model.heads.head = torch.nn.Identity()
    else:
        raise ValidationError(
            f"Unsupported vision architecture {architecture!r}; expected 'resnet18' or 'vit_b_16'."
        )

    if weights == "mock":
        _apply_mock_weights(model, seed=seed)
        warnings.append("mock weights: random init for CI — not ImageNet quality.")
    elif weights == "none":
        warnings.append("weights=none: random architecture init; no pretrained tensors.")
    freeze_module(model, freeze=freeze)
    return PretrainedBackbone(
        module=model,
        modality="vision",
        architecture=architecture,
        weight_mode=weights,
        frozen=freeze,
        feature_dim=feature_dim,
        disclosures=(
            f"Vision backbone {architecture} via torchvision (weights={weights}).",
            "fc/head replaced with Identity for feature extraction.",
            "freeze=True keeps backbone fixed for linear-probe / finetune-head workflows."
            if freeze
            else "Backbone parameters are trainable (finetune).",
        ),
        limitations=(
            "Not a full pretrained zoo product — curated ResNet/ViT hooks only.",
            "Caller supplies the classification/regression head and data pipeline.",
            "pretrained mode may download large weights; prefer mock/none in CI.",
        ),
        warnings=tuple(warnings),
        meta={"provider": "torchvision", "seed": seed},
    )


def load_audio_backbone(
    architecture: AudioArch = "wav2vec2_base",
    *,
    weights: WeightMode = "mock",
    freeze: bool = True,
    seed: int = 0,
    model_id: str | None = None,
) -> PretrainedBackbone:
    """Load a transformers Wav2Vec2-class audio encoder backbone."""
    require_torch(feature="audio backbone")
    transformers = _require_transformers()
    warnings: list[str] = []
    if architecture != "wav2vec2_base":
        raise ValidationError(
            f"Unsupported audio architecture {architecture!r}; expected 'wav2vec2_base'."
        )
    resolved_id = model_id or "facebook/wav2vec2-base"
    if weights == "pretrained":
        model = transformers.Wav2Vec2Model.from_pretrained(resolved_id)
    else:
        # Config-only construct — no weight download.
        try:
            config = transformers.Wav2Vec2Config.from_pretrained(resolved_id)
        except Exception:  # noqa: BLE001
            config = transformers.Wav2Vec2Config()
            warnings.append("Could not fetch Wav2Vec2Config remotely; used local default config.")
        model = transformers.Wav2Vec2Model(config)
        if weights == "mock":
            _apply_mock_weights(model, seed=seed)
            warnings.append("mock weights: random init for CI — not wav2vec2 quality.")
        else:
            warnings.append("weights=none: random architecture init.")
    freeze_module(model, freeze=freeze)
    feature_dim = int(getattr(model.config, "hidden_size", 768))
    return PretrainedBackbone(
        module=model,
        modality="audio",
        architecture=architecture,
        weight_mode=weights,
        frozen=freeze,
        feature_dim=feature_dim,
        disclosures=(
            f"Audio backbone {architecture} via transformers (weights={weights}).",
            f"model_id={resolved_id}",
        ),
        limitations=(
            "Integration hook — not a full audio FM zoo or pretraining stack.",
            "pretrained mode may download large weights; prefer mock/none in CI.",
        ),
        warnings=tuple(warnings),
        meta={"provider": "transformers", "model_id": resolved_id, "seed": seed},
    )


def load_speech_backbone(
    architecture: SpeechArch = "whisper_tiny_encoder",
    *,
    weights: WeightMode = "mock",
    freeze: bool = True,
    seed: int = 0,
    model_id: str | None = None,
) -> PretrainedBackbone:
    """Load a Whisper-class encoder backbone (finetune/feature extract — not FM pretrain)."""
    require_torch(feature="speech backbone")
    transformers = _require_transformers()
    warnings: list[str] = []
    if architecture != "whisper_tiny_encoder":
        raise ValidationError(
            f"Unsupported speech architecture {architecture!r}; expected 'whisper_tiny_encoder'."
        )
    # Default to HF tiny-random for mock/none; openai whisper-tiny for pretrained.
    if weights == "pretrained":
        resolved_id = model_id or "openai/whisper-tiny"
        model = transformers.WhisperModel.from_pretrained(resolved_id)
    else:
        resolved_id = model_id or (
            "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
        )
        try:
            config = transformers.WhisperConfig.from_pretrained(resolved_id)
        except Exception:  # noqa: BLE001
            config = transformers.WhisperConfig()
            warnings.append("Could not fetch WhisperConfig remotely; used local default config.")
        model = transformers.WhisperModel(config)
        if weights == "mock":
            _apply_mock_weights(model, seed=seed)
            warnings.append("mock weights: random init for CI — not Whisper quality.")
        else:
            warnings.append("weights=none: random architecture init.")
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
    freeze_module(encoder, freeze=freeze)
    feature_dim = int(getattr(encoder.config, "d_model", getattr(model.config, "d_model", 384)))
    return PretrainedBackbone(
        module=encoder,
        modality="speech",
        architecture=architecture,
        weight_mode=weights,
        frozen=freeze,
        feature_dim=feature_dim,
        disclosures=(
            f"Speech encoder {architecture} via transformers (weights={weights}).",
            f"model_id={resolved_id}",
            "This loads an encoder for finetune/feature extract — "
            "not Whisper-scale foundation-model training from scratch.",
        ),
        limitations=(
            "BuildML does not train Whisper-scale FMs from scratch "
            "(needs massive data/compute outside a pip library).",
            "pretrained mode may download weights; prefer mock/none in CI.",
        ),
        warnings=tuple(warnings),
        meta={"provider": "transformers", "model_id": resolved_id, "seed": seed},
    )


def _validate_weight_mode(weights: str) -> WeightMode:
    if weights not in {"none", "mock", "pretrained"}:
        raise ValidationError(
            f"Unsupported weights mode {weights!r}; expected 'none', 'mock', or 'pretrained'."
        )
    return weights  # type: ignore[return-value]


def load_pretrained_backbone(
    modality: Literal["vision", "audio", "speech"],
    architecture: str | None = None,
    *,
    weights: WeightMode = "mock",
    freeze: bool = True,
    seed: int = 0,
    model_id: str | None = None,
) -> PretrainedBackbone:
    """Dispatch helper for vision / audio / speech backbone loads."""
    weights = _validate_weight_mode(weights)
    if modality == "vision":
        arch: VisionArch = "resnet18"  # type: ignore[assignment]
        if architecture is not None:
            if architecture not in {"resnet18", "vit_b_16"}:
                raise ValidationError(f"Unknown vision architecture {architecture!r}")
            arch = architecture  # type: ignore[assignment]
        return load_vision_backbone(arch, weights=weights, freeze=freeze, seed=seed)
    if modality == "audio":
        if architecture is not None and architecture != "wav2vec2_base":
            raise ValidationError(
                f"Unknown audio architecture {architecture!r}; expected 'wav2vec2_base'."
            )
        return load_audio_backbone(
            "wav2vec2_base",
            weights=weights,
            freeze=freeze,
            seed=seed,
            model_id=model_id,
        )
    if modality == "speech":
        if architecture is not None and architecture != "whisper_tiny_encoder":
            raise ValidationError(
                f"Unknown speech architecture {architecture!r}; expected 'whisper_tiny_encoder'."
            )
        return load_speech_backbone(
            "whisper_tiny_encoder",
            weights=weights,
            freeze=freeze,
            seed=seed,
            model_id=model_id,
        )
    raise ValidationError(f"Unknown modality {modality!r}")
