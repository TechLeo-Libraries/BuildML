"""Pretrained vision / audio / speech backbone hooks (integration, not a full zoo).

Loads ResNet/ViT-class (torchvision), Wav2Vec/HuBERT-class (transformers), and
Whisper-encoder-class (transformers) backbones with freeze/finetune helpers.

Weight modes
------------
* ``none`` — architecture only (random init; no download).
* ``mock`` — architecture + tiny random compatible tensors (CI-safe).
* ``pretrained`` — real published weights (may download; needs network/extra).

Shipped paths are real library loaders. Limits describe product scope
(not a full HF/TorchVision zoo SaaS; not FM-from-scratch), not stubs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import require_torch

WeightMode = Literal["none", "mock", "pretrained"]
VisionArch = Literal["resnet18", "resnet34", "resnet50", "vit_b_16", "vit_b_32"]
AudioArch = Literal["wav2vec2_base", "hubert_base"]
SpeechArch = Literal["whisper_tiny_encoder", "whisper_base_encoder"]

_VISION_ARCHS = ("resnet18", "resnet34", "resnet50", "vit_b_16", "vit_b_32")
_AUDIO_ARCHS = ("wav2vec2_base", "hubert_base")
_SPEECH_ARCHS = ("whisper_tiny_encoder", "whisper_base_encoder")

_DEFAULT_AUDIO_IDS = {
    "wav2vec2_base": "facebook/wav2vec2-base",
    "hubert_base": "facebook/hubert-base-ls960",
}
_DEFAULT_SPEECH_PRETRAINED = {
    "whisper_tiny_encoder": "openai/whisper-tiny",
    "whisper_base_encoder": "openai/whisper-base",
}
_DEFAULT_SPEECH_MOCK = {
    "whisper_tiny_encoder": "hf-internal-testing/tiny-random-WhisperForConditionalGeneration",
    "whisper_base_encoder": "hf-internal-testing/tiny-random-WhisperForConditionalGeneration",
}


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


@dataclass(slots=True)
class BackboneHeadResult:
    """Backbone + linear head for linear-probe / finetune-head workflows."""

    module: Any
    backbone: PretrainedBackbone
    n_classes: int
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "backbone": self.backbone.to_dict(),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "module": type(self.module).__name__,
        }


def list_pretrained_backbones() -> tuple[dict[str, str], ...]:
    """Return the curated architecture catalog (honesty: not a full zoo product)."""
    rows: list[dict[str, str]] = []
    for arch in _VISION_ARCHS:
        rows.append({"modality": "vision", "architecture": arch, "provider": "torchvision"})
    for arch in _AUDIO_ARCHS:
        rows.append({"modality": "audio", "architecture": arch, "provider": "transformers"})
    for arch in _SPEECH_ARCHS:
        rows.append({"modality": "speech", "architecture": arch, "provider": "transformers"})
    return tuple(rows)


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


def attach_backbone_head(
    backbone: PretrainedBackbone,
    *,
    n_classes: int,
    freeze_backbone: bool | None = None,
) -> BackboneHeadResult:
    """Attach a linear classification head to a loaded backbone."""
    torch = require_torch(feature="attach backbone head")
    if n_classes < 2:
        raise ValidationError("n_classes must be >= 2")
    should_freeze = backbone.frozen if freeze_backbone is None else bool(freeze_backbone)
    freeze_module(backbone.module, freeze=should_freeze)
    feature_dim = int(backbone.feature_dim)
    head = torch.nn.Linear(feature_dim, int(n_classes))
    encoder = backbone.module
    modality_name = f"{backbone.modality}_backbone_classify"

    class _BackboneClassifier(torch.nn.Module):
        modality = modality_name

        def __init__(self) -> None:
            super().__init__()
            self.backbone = encoder
            self.head = head
            self.n_classes = int(n_classes)
            self.feature_dim = feature_dim

        def _pool(self, out: Any) -> Any:
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state.mean(dim=1)
            if isinstance(out, (tuple, list)):
                hidden = out[0]
                return hidden.mean(dim=1) if getattr(hidden, "dim", lambda: 0)() == 3 else hidden
            if hasattr(out, "dim") and out.dim() == 3:
                return out.mean(dim=1)
            if hasattr(out, "dim") and out.dim() > 2:
                return out.flatten(1)
            return out

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            feats = self.backbone(*args, **kwargs)
            return self.head(self._pool(feats))

    module = _BackboneClassifier()
    updated = PretrainedBackbone(
        module=backbone.module,
        modality=backbone.modality,
        architecture=backbone.architecture,
        weight_mode=backbone.weight_mode,
        frozen=should_freeze,
        feature_dim=backbone.feature_dim,
        disclosures=backbone.disclosures,
        limitations=backbone.limitations,
        warnings=backbone.warnings,
        meta=dict(backbone.meta),
    )
    return BackboneHeadResult(
        module=module,
        backbone=updated,
        n_classes=int(n_classes),
        disclosures=(
            f"Attached Linear({feature_dim} → {n_classes}) on {backbone.architecture}.",
            "Linear-probe / finetune-head helper — not foundation-model pretrain.",
        ),
        limitations=(
            "Caller still owns DataLoaders / TrainConfig / fit_torch wiring.",
            "Not a full zoo product or auto-training platform.",
        ),
        warnings=(),
    )


def load_vision_backbone(
    architecture: VisionArch = "resnet18",
    *,
    weights: WeightMode = "mock",
    freeze: bool = True,
    seed: int = 0,
) -> PretrainedBackbone:
    """Load a torchvision vision backbone as a feature extractor."""
    torch = require_torch(feature="vision backbone")
    tv = _require_torchvision()
    warnings: list[str] = []
    if architecture not in _VISION_ARCHS:
        raise ValidationError(
            f"Unsupported vision architecture {architecture!r}; expected one of {_VISION_ARCHS}."
        )
    if architecture.startswith("resnet"):
        ctor = getattr(tv.models, architecture)
        weights_enum = None
        if weights == "pretrained":
            weights_cls = {
                "resnet18": tv.models.ResNet18_Weights,
                "resnet34": tv.models.ResNet34_Weights,
                "resnet50": tv.models.ResNet50_Weights,
            }[architecture]
            weights_enum = weights_cls.DEFAULT
        model = ctor(weights=weights_enum)
        feature_dim = int(model.fc.in_features)
        model.fc = torch.nn.Identity()
    else:
        ctor = getattr(tv.models, architecture)
        weights_enum = None
        if weights == "pretrained":
            weights_cls = {
                "vit_b_16": tv.models.ViT_B_16_Weights,
                "vit_b_32": tv.models.ViT_B_32_Weights,
            }[architecture]
            weights_enum = weights_cls.DEFAULT
        model = ctor(weights=weights_enum)
        feature_dim = int(model.heads.head.in_features)
        model.heads.head = torch.nn.Identity()

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
        ),
        limitations=(
            "Not a full pretrained zoo product — curated ResNet/ViT hooks only.",
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
    """Load a transformers Wav2Vec2 / HuBERT-class audio encoder backbone."""
    require_torch(feature="audio backbone")
    transformers = _require_transformers()
    warnings: list[str] = []
    if architecture not in _AUDIO_ARCHS:
        raise ValidationError(
            f"Unsupported audio architecture {architecture!r}; expected one of {_AUDIO_ARCHS}."
        )
    resolved_id = model_id or _DEFAULT_AUDIO_IDS[architecture]
    if architecture == "wav2vec2_base":
        model_cls, config_cls = transformers.Wav2Vec2Model, transformers.Wav2Vec2Config
    else:
        model_cls, config_cls = transformers.HubertModel, transformers.HubertConfig
    if weights == "pretrained":
        model = model_cls.from_pretrained(resolved_id)
    else:
        try:
            config = config_cls.from_pretrained(resolved_id)
        except Exception:  # noqa: BLE001
            config = config_cls()
            warnings.append(f"Could not fetch {architecture} config remotely; used local default.")
        model = model_cls(config)
        if weights == "mock":
            _apply_mock_weights(model, seed=seed)
            warnings.append(f"mock weights: random init for CI — not {architecture} quality.")
        else:
            warnings.append("weights=none: random architecture init.")
    freeze_module(model, freeze=freeze)
    return PretrainedBackbone(
        module=model,
        modality="audio",
        architecture=architecture,
        weight_mode=weights,
        frozen=freeze,
        feature_dim=int(getattr(model.config, "hidden_size", 768)),
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
    if architecture not in _SPEECH_ARCHS:
        raise ValidationError(
            f"Unsupported speech architecture {architecture!r}; expected one of {_SPEECH_ARCHS}."
        )
    if weights == "pretrained":
        resolved_id = model_id or _DEFAULT_SPEECH_PRETRAINED[architecture]
        model = transformers.WhisperModel.from_pretrained(resolved_id)
    else:
        resolved_id = model_id or _DEFAULT_SPEECH_MOCK[architecture]
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
            "Encoder for finetune/feature extract — not Whisper-scale FM training from scratch.",
        ),
        limitations=(
            "BuildML does not train Whisper-scale FMs from scratch.",
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
        arch = architecture or "resnet18"
        if arch not in _VISION_ARCHS:
            raise ValidationError(f"Unknown vision architecture {arch!r}")
        return load_vision_backbone(arch, weights=weights, freeze=freeze, seed=seed)  # type: ignore[arg-type]
    if modality == "audio":
        arch = architecture or "wav2vec2_base"
        if arch == "hubert":
            arch = "hubert_base"
        if arch not in _AUDIO_ARCHS:
            raise ValidationError(f"Unknown audio architecture {arch!r}")
        return load_audio_backbone(
            arch, weights=weights, freeze=freeze, seed=seed, model_id=model_id  # type: ignore[arg-type]
        )
    if modality == "speech":
        arch = architecture or "whisper_tiny_encoder"
        if arch not in _SPEECH_ARCHS:
            raise ValidationError(f"Unknown speech architecture {arch!r}")
        return load_speech_backbone(
            arch, weights=weights, freeze=freeze, seed=seed, model_id=model_id  # type: ignore[arg-type]
        )
    raise ValidationError(f"Unknown modality {modality!r}")
