"""Start from a model that already learned something, instead of from scratch.

Training a vision or audio model from random initialisation needs an enormous
amount of labelled data. Transfer learning avoids that: take a network already
trained on a very large corpus, keep the representations it learned, and train
only a small head on your labels. It routinely works with a few hundred examples
where from-scratch training would need hundreds of thousands.

This module loads such backbones. Vision architectures come from torchvision
(ResNet and ViT), audio and speech from Hugging Face transformers (Wav2Vec2,
HuBERT, and the Whisper encoder). Each is returned as a feature extractor with
its classification head removed, ready for :func:`attach_backbone_head`.

Three weight modes, and picking the right one matters. ``pretrained`` downloads
the real published weights and is what you want for actual work. ``mock``
constructs the architecture with random weights, which is fast, offline, and
useless for accuracy: it exists so tests can exercise the plumbing without
pulling gigabytes. ``none`` is the same thing without even the random
initialisation pass.

This is an integration layer over torchvision and transformers, not a model zoo
of its own. It curates a handful of architectures that work well; it does not
mirror the full Hugging Face hub, and it does not train foundation models from
scratch.

See Also
--------
buildml.dl.train : Training the head.
buildml.dl.image : Preparing image tensors.
buildml.dl.speech : Transcription, rather than feature extraction.
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
    """A loaded feature extractor, with a clear record of what it actually is.

    Attributes
    ----------
    module:
        The backbone, with its classification head replaced by identity so it
        emits features rather than class scores.
    modality:
        ``'vision'``, ``'audio'``, or ``'speech'``.
    architecture:
        Which one was loaded.
    weight_mode:
        ``'pretrained'``, ``'mock'``, or ``'none'``. **Check this before
        drawing conclusions from any result**: mock weights are random.
    frozen:
        Whether the parameters are excluded from gradient updates.
    feature_dim:
        Output width, needed to size a head.
    disclosures:
        What was loaded and from where.
    limitations:
        Scope boundaries.
    warnings:
        Notably that weights are random, or that a config could not be fetched.
    meta:
        Provider, model identifier, and seed.

    Notes
    -----
    **``weight_mode`` is the field to look at first.** A pipeline that works
    end to end with ``mock`` weights is a pipeline that works; it is not a model
    that predicts anything.

    See Also
    --------
    load_pretrained_backbone : Produces this.
    attach_backbone_head : Making it a classifier.
    """

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
        """Return the backbone's provenance as JSON-safe values.

        Records what was loaded and under which weight mode: the thing that
        determines whether a downstream result means anything. The module
        itself is reported by class name.

        Returns
        -------
        dict
            Modality, architecture, weight mode, frozen flag, feature
            dimension, the three prose lists, metadata, and the module class
            name.
        """
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
    """A backbone with a classification head attached, ready to train.

    Attributes
    ----------
    module:
        The combined model: backbone, pooling, and linear head. Train this.
    backbone:
        The underlying backbone record, with its frozen flag updated.
    n_classes:
        Output width.
    disclosures:
        What was attached.
    limitations:
        What remains yours: loaders, training configuration, the fit call.
    warnings:
        Anything notable.

    See Also
    --------
    attach_backbone_head : Produces this.
    """

    module: Any
    backbone: PretrainedBackbone
    n_classes: int
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the assembled model's description as JSON-safe values.

        The backbone's own record nests inside, so the weight mode stays
        visible in the log.

        Returns
        -------
        dict
            Class count, the nested backbone description, the three prose
            lists, and the module class name.
        """
        return {
            "n_classes": self.n_classes,
            "backbone": self.backbone.to_dict(),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "module": type(self.module).__name__,
        }


def list_pretrained_backbones() -> tuple[dict[str, str], ...]:
    """List the architectures this module knows how to load.

    A curated set, not a mirror of any hub. Each entry names a modality, an
    architecture, and which library provides it.

    Returns
    -------
    tuple of dict
        One entry per architecture, with ``modality``, ``architecture``, and
        ``provider`` keys.

    Notes
    -----
    **Availability here does not mean the provider is installed.** Vision needs
    torchvision, audio and speech need transformers, and both are optional. The
    loaders raise a clear error when the provider is missing.

    See Also
    --------
    load_pretrained_backbone : Loading one of these.
    """
    rows: list[dict[str, str]] = []
    for arch in _VISION_ARCHS:
        rows.append({"modality": "vision", "architecture": arch, "provider": "torchvision"})
    for arch in _AUDIO_ARCHS:
        rows.append({"modality": "audio", "architecture": arch, "provider": "transformers"})
    for arch in _SPEECH_ARCHS:
        rows.append({"modality": "speech", "architecture": arch, "provider": "transformers"})
    return tuple(rows)


def freeze_module(module: Any, *, freeze: bool = True) -> Any:
    """Stop or resume gradient updates for every parameter in a module.

    Freezing sets ``requires_grad = False`` throughout, so the optimiser leaves
    those weights alone. This is what makes transfer learning cheap: the
    backbone's learned representations are kept as-is, and only the head is
    trained.

    Parameters
    ----------
    module:
        The module to modify. Changed in place.
    freeze:
        ``True`` to freeze, ``False`` to unfreeze.

    Returns
    -------
    torch.nn.Module
        The same module, for chaining.

    Notes
    -----
    **Frozen makes training much faster and much less flexible.** No gradients
    are computed for those layers, so each step is cheaper and uses less memory
   : but the representations cannot adapt to your data at all.

    **Unfreezing after the head has trained is the usual refinement.** Training
    a random head against an unfrozen backbone lets large early gradients damage
    good pretrained weights; training the head first, then unfreezing with a
    small learning rate, avoids that.

    See Also
    --------
    attach_backbone_head : Where the frozen flag is usually set.
    """
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
    """Turn a feature extractor into a classifier for your labels.

    Wraps the backbone with pooling and a single linear layer sized to your
    class count. The result is a module that takes the backbone's input and
    emits class logits, trainable with the ordinary training loop.

    Parameters
    ----------
    backbone:
        A loaded backbone.
    n_classes:
        How many classes to predict.
    freeze_backbone:
        Override the backbone's frozen state. ``None`` keeps it as loaded.

    Returns
    -------
    BackboneHeadResult
        The combined module, the updated backbone record, and the limitations.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If ``n_classes`` is below 2.

    Notes
    -----
    **Pooling adapts to whatever the backbone emits.** Transformer outputs with
    a sequence dimension are mean-pooled across it; already-flat outputs pass
    through. This is what lets one head work across vision, audio, and speech
    backbones that have quite different output shapes.

    **A frozen backbone with a linear head is a linear probe**, and it is a
    genuinely useful measurement: it tells you how much of your task is already
    captured by the pretrained representations, before you spend anything on
    fine-tuning.

    **Loaders and training are still yours.** This returns a module, not a
    trained model.

    Examples
    --------
    Linear probe on a pretrained ResNet::

        backbone = load_vision_backbone("resnet18", weights="pretrained")
        head = attach_backbone_head(backbone, n_classes=5)
        head.module  # pass to train_supervised_module

    See Also
    --------
    freeze_module : Changing the frozen state later.
    buildml.dl.train.train_supervised_module : Training the result.
    """
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
            "Linear-probe / finetune-head helper: not foundation-model pretrain.",
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
    """Load a ResNet or Vision Transformer as an image feature extractor.

    Constructs the architecture, optionally with ImageNet weights, and replaces
    the final classification layer with identity so the module emits features
    instead of ImageNet class scores.

    Parameters
    ----------
    architecture:
        ``'resnet18'``, ``'resnet34'``, ``'resnet50'``, ``'vit_b_16'``, or
        ``'vit_b_32'``.
    weights:
        ``'pretrained'`` for real ImageNet weights, ``'mock'`` for random
        weights suitable only for testing, ``'none'`` for a bare architecture.
    freeze:
        Freeze the parameters. On by default, which is the linear-probe setup.
    seed:
        Controls mock initialisation, so tests are reproducible.

    Returns
    -------
    PretrainedBackbone
        The feature extractor and its provenance.

    Raises
    ------
    MissingExtraError
        If PyTorch or torchvision is not installed.
    ValidationError
        If the architecture is not one of the supported names.

    Notes
    -----
    **ResNets are the safer default; ViTs need more data.** Vision transformers
    reach higher ceilings but are more sensitive to fine-tuning data volume, and
    ResNet-18 is a better first attempt on a small labelled set.

    **Pretrained weights expect ImageNet-style input**: three channels,
    224x224, normalised with ImageNet statistics. Feeding differently prepared
    images works mechanically and degrades the representations, sometimes
    substantially.

    **``pretrained`` downloads on first use**, tens to hundreds of megabytes, to
    the torch hub cache. Use ``mock`` in CI.

    Examples
    --------
    Pretrained extractor, frozen for a linear probe::

        backbone = load_vision_backbone("resnet18", weights="pretrained")
        backbone.feature_dim  # 512

    See Also
    --------
    attach_backbone_head : Making it a classifier.
    buildml.dl.image : Preparing the image tensors.
    """
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
        warnings.append("mock weights: random init for CI: not ImageNet quality.")
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
            "Not a full pretrained zoo product: curated ResNet/ViT hooks only.",
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
    """Load a Wav2Vec2 or HuBERT encoder as an audio feature extractor.

    Both were trained on very large quantities of unlabelled speech using
    self-supervised objectives, and both produce representations that transfer
    well to downstream audio tasks with modest labelled data.

    Parameters
    ----------
    architecture:
        ``'wav2vec2_base'`` or ``'hubert_base'``.
    weights:
        ``'pretrained'`` for real weights, ``'mock'`` for random, ``'none'``
        for a bare architecture.
    freeze:
        Freeze the parameters.
    seed:
        Controls mock initialisation.
    model_id:
        A specific Hugging Face model identifier, overriding the default.

    Returns
    -------
    PretrainedBackbone
        The feature extractor and its provenance.

    Raises
    ------
    MissingExtraError
        If PyTorch or transformers is not installed. Install with
        ``pip install buildml[speech]``.
    ValidationError
        If the architecture is not one of the supported names.

    Notes
    -----
    **These expect raw 16 kHz waveforms**, not spectrograms and not audio at
    other sample rates. :mod:`buildml.dl.audio` resamples to 16 kHz by default
    for exactly this reason.

    **The output has a time dimension.** These emit one vector per frame rather
    than one per clip, so a head has to pool across time :
    :func:`attach_backbone_head` mean-pools automatically.

    **Wav2Vec2 and HuBERT are close in practice.** They differ in training
    objective; for a downstream classification task the choice rarely matters
    much, and trying both is cheap.

    Examples
    --------
    Frozen speech representations::

        backbone = load_audio_backbone("wav2vec2_base", weights="pretrained")
        backbone.feature_dim  # 768

    See Also
    --------
    buildml.dl.audio : Preparing waveforms at the right rate.
    load_speech_backbone : Whisper, for transcription-oriented features.
    """
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
            warnings.append(f"mock weights: random init for CI: not {architecture} quality.")
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
            "Integration hook: not a full audio FM zoo or pretraining stack.",
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
    """Load the encoder half of a Whisper model as a speech feature extractor.

    Whisper was trained on a very large multilingual transcription corpus. Its
    encoder learned representations that capture linguistic content well, and
    those transfer usefully to speech classification tasks. The decoder: the
    part that generates text: is discarded here.

    Parameters
    ----------
    architecture:
        ``'whisper_tiny_encoder'`` or ``'whisper_base_encoder'``.
    weights:
        ``'pretrained'`` for real weights, ``'mock'`` for random, ``'none'``
        for a bare architecture.
    freeze:
        Freeze the parameters.
    seed:
        Controls mock initialisation.
    model_id:
        A specific Hugging Face model identifier, overriding the default.

    Returns
    -------
    PretrainedBackbone
        The encoder and its provenance.

    Raises
    ------
    MissingExtraError
        If PyTorch or transformers is not installed. Install with
        ``pip install buildml[speech]``.
    ValidationError
        If the architecture is not one of the supported names.

    Notes
    -----
    **Whisper expects log-mel spectrograms, not raw waveforms**: a different
    input format from Wav2Vec2 and HuBERT. Use the Hugging Face Whisper
    feature extractor to prepare audio for it.

    **Encoder features skew toward linguistic content.** Whisper was trained to
    transcribe, so its representations emphasise what was said. For tasks about
    who is speaking or how, Wav2Vec2 or HuBERT may suit better.

    **For transcription, this is the wrong entry point.** See
    :mod:`buildml.dl.speech`, which uses the whole model.

    Examples
    --------
    Feature extraction from the tiny encoder::

        backbone = load_speech_backbone("whisper_tiny_encoder", weights="pretrained")
        backbone.feature_dim  # 384 for tiny

    See Also
    --------
    buildml.dl.speech : Transcription with the full model.
    load_audio_backbone : Waveform-input alternatives.
    """
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
            warnings.append("mock weights: random init for CI: not Whisper quality.")
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
            "Encoder for finetune/feature extract: not Whisper-scale FM training from scratch.",
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
    """Load a backbone for any modality through one entry point.

    Dispatches to the modality-specific loader, applying a sensible default
    architecture when none is named. Convenient when the modality is
    configuration rather than a decision made in code.

    Parameters
    ----------
    modality:
        ``'vision'``, ``'audio'``, or ``'speech'``.
    architecture:
        Which architecture. Defaults to ``'resnet18'``, ``'wav2vec2_base'``, or
        ``'whisper_tiny_encoder'`` respectively.
    weights:
        ``'pretrained'``, ``'mock'``, or ``'none'``.
    freeze:
        Freeze the parameters.
    seed:
        Controls mock initialisation.
    model_id:
        A specific Hugging Face identifier. Audio and speech only; vision
        weights come from torchvision.

    Returns
    -------
    PretrainedBackbone
        The feature extractor and its provenance.

    Raises
    ------
    MissingExtraError
        If a required provider is not installed.
    ValidationError
        If the modality, architecture, or weight mode is unrecognised.

    Notes
    -----
    ``'hubert'`` is accepted as an alias for ``'hubert_base'``, since the short
    form is the natural thing to type.

    Examples
    --------
    Load whatever the configuration asked for::

        backbone = load_pretrained_backbone("vision", weights="pretrained")

    See Also
    --------
    list_pretrained_backbones : What can be loaded.
    load_vision_backbone : The vision path, with its caveats.
    load_audio_backbone : The audio path, with its caveats.
    load_speech_backbone : The speech path, with its caveats.
    """
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
