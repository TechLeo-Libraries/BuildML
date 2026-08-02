"""Speech foundation-model integration path (ASR inference + finetune-lite).

Honest alpha scope
------------------
* **ASR transcription** via a stub backend (CI-safe) or optional Hugging Face
  ``transformers`` Whisper-class pipelines behind ``buildml[speech]``.
* **Classification fine-tune** on frozen/tiny speech encoder embeddings fused
  with optional tabular labels — leakage-safe splits, train-only amp stats.

This is an **integration / finetune-lite** path. It does **not** train a
Whisper-scale foundation model from scratch.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.dl.audio import (
    apply_audio_waveform_stats,
    decode_audio_cell,
    fit_audio_waveform_stats,
    stack_audio_column,
)
from buildml.dl.extras import require_torch
from buildml.dl.labels import encode_class_targets, fit_class_labels
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.types import FeatureContract

SpeechBackend = Literal["stub", "transformers"]
SpeechMode = Literal["classify", "asr"]


@dataclass(slots=True)
class SpeechLoaderConfig:
    """Knobs for speech classification DataLoader construction."""

    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    seed: int = 0
    sample_rate: int = 16_000
    max_samples: int = 16_000
    source_sample_rate: int | None = None
    normalize_audio: bool = True
    encoder_dim: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeechContract:
    """Schema carried with speech Torch loaders / trainers."""

    audio_column: str
    target_column: str
    task: Literal["classification"] = "classification"
    class_labels: tuple[Any, ...] = ()
    sample_rate: int = 16_000
    max_samples: int = 16_000
    source_sample_rate: int | None = None
    audio_mean: float | None = None
    audio_std: float | None = None
    encoder_dim: int = 64
    modality: str = "speech_classify"
    disclosures: tuple[str, ...] = ()

    def to_feature_contract(self) -> FeatureContract:
        return FeatureContract(
            feature_columns=(self.audio_column,),
            target_column=self.target_column,
            task="classification",
            class_labels=self.class_labels,
            normalize_mean=None,
            normalize_std=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_column": self.audio_column,
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": list(self.class_labels),
            "sample_rate": self.sample_rate,
            "max_samples": self.max_samples,
            "source_sample_rate": self.source_sample_rate,
            "audio_mean": self.audio_mean,
            "audio_std": self.audio_std,
            "encoder_dim": self.encoder_dim,
            "modality": self.modality,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class SpeechTranscribeResult:
    """ASR transcription outcomes for one or more audio cells."""

    texts: list[str]
    backend: str
    model_id: str | None
    n_rows: int
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "texts": list(self.texts),
            "backend": self.backend,
            "model_id": self.model_id,
            "n_rows": self.n_rows,
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }


def require_speech_stack(*, feature: str = "Speech transformers backend") -> Any:
    """Import and return ``transformers``, or raise :class:`MissingExtraError`."""
    try:
        import transformers
    except ImportError as exc:
        raise MissingExtraError("speech", feature) from exc
    return transformers


def speech_stack_available() -> bool:
    """Return True when ``transformers`` is importable."""
    return importlib.util.find_spec("transformers") is not None


def _resolve_audio_target(
    dataset: Dataset,
    audio_column: str | None,
) -> tuple[str, str]:
    roles = dataset.roles
    target_cols = [c for c, r in roles.items() if r == ColumnRole.TARGET]
    if len(target_cols) != 1:
        raise ValidationError(
            "Speech classify path requires exactly one target role column."
        )
    target = target_cols[0]
    if audio_column is not None:
        if audio_column not in dataset.columns:
            raise ValidationError(f"audio_column {audio_column!r} not in dataset")
        return audio_column, target
    feature_cols = [c for c, r in roles.items() if r == ColumnRole.FEATURE]
    if len(feature_cols) != 1:
        raise ValidationError(
            "Pass audio_column= explicitly when multiple feature columns exist."
        )
    return feature_cols[0], target


def build_tiny_speech_encoder(
    *,
    in_channels: int = 1,
    embed_dim: int = 64,
    sample_rate: int = 16_000,
) -> Any:
    """Build a tiny 1D-CNN speech encoder (Torch-only; no HF weights).

    Suitable for CI and finetune-lite classification. Not a Whisper-scale FM.
    """
    torch = require_torch(feature="Tiny speech encoder")

    class TinySpeechEncoder(torch.nn.Module):
        modality = "speech_encoder"

        def __init__(self) -> None:
            super().__init__()
            self.sample_rate = int(sample_rate)
            self.embed_dim = int(embed_dim)
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, 16, kernel_size=9, stride=2, padding=4),
                torch.nn.ReLU(),
                torch.nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
                torch.nn.ReLU(),
                torch.nn.Conv1d(32, embed_dim, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
            )

        def forward(self, x: Any) -> Any:
            # x: (B, 1, T) or (B, T)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h = self.net(x)
            return h.squeeze(-1)

    return TinySpeechEncoder()


def build_speech_classifier(
    *,
    n_classes: int,
    embed_dim: int = 64,
    sample_rate: int = 16_000,
    freeze_encoder: bool = False,
) -> Any:
    """Build encoder + linear head for speech classification fine-tune."""
    torch = require_torch(feature="Speech classifier")
    if n_classes < 2:
        raise ValidationError("n_classes must be >= 2 for speech classification")
    encoder = build_tiny_speech_encoder(embed_dim=embed_dim, sample_rate=sample_rate)
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    class SpeechClassifier(torch.nn.Module):
        modality = "speech_classify"

        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Linear(embed_dim, n_classes)
            self.n_classes = int(n_classes)
            self.embed_dim = int(embed_dim)
            self.freeze_encoder = bool(freeze_encoder)

        def forward(self, x: Any) -> Any:
            return self.head(self.encoder(x))

    return SpeechClassifier()


def make_speech_loaders(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    audio_column: str | None = None,
    config: SpeechLoaderConfig | None = None,
) -> TorchLoaderBundle:
    """Build waveform DataLoaders for speech classification (finetune-lite).

    Amplitude mean/std are fit on the **train** partition only.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    require_torch(feature="Speech Torch DataLoaders")
    cfg = config or SpeechLoaderConfig()
    if cfg.batch_size < 1:
        raise ValidationError("batch_size must be >= 1")
    if cfg.max_samples < 1:
        raise ValidationError("max_samples must be >= 1")

    audio_col, target = _resolve_audio_target(dataset, audio_column)
    frame = dataset._ensure_pandas()
    train_idx = list(split_plan.indices_for("train"))
    if not train_idx:
        raise ValidationError("Train partition is empty; cannot build speech loaders")

    train_audio, train_lengths = stack_audio_column(
        frame.iloc[train_idx][audio_col].tolist(),
        sample_rate=cfg.sample_rate,
        max_samples=cfg.max_samples,
        source_sample_rate=cfg.source_sample_rate,
        return_lengths=True,
    )
    audio_mean: float | None = None
    audio_std: float | None = None
    if cfg.normalize_audio:
        mean_arr, std_arr = fit_audio_waveform_stats(train_audio, lengths=train_lengths)
        audio_mean = float(mean_arr.reshape(-1)[0])
        audio_std = float(std_arr.reshape(-1)[0])

    y_train = frame.iloc[train_idx][target]
    if not pd.api.types.is_numeric_dtype(y_train):
        raise ValidationError(
            f"Target '{target}' must be numeric class ids for the speech classify path "
            "(encode labels to integers first)."
        )
    class_labels = fit_class_labels(y_train)
    disclosures = (
        "Speech classify finetune-lite: tiny 1D-CNN encoder + linear head.",
        "Not training a Whisper-scale foundation model from scratch.",
        "Amplitude mean/std fit on train only when normalize_audio=True.",
        "Class labels remapped to contiguous 0..K-1; class_labels stores original ids.",
    )
    contract = SpeechContract(
        audio_column=audio_col,
        target_column=target,
        class_labels=class_labels,
        sample_rate=cfg.sample_rate,
        max_samples=cfg.max_samples,
        source_sample_rate=cfg.source_sample_rate,
        audio_mean=audio_mean,
        audio_std=audio_std,
        encoder_dim=cfg.encoder_dim,
        disclosures=disclosures,
    )

    torch = require_torch(feature="Speech Torch DataLoaders")
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loaders: dict[str, Any] = {}
    warnings: list[str] = list(disclosures)
    n_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    for name in ("train", "validation", "test"):
        idx = list(split_plan.indices_for(name))  # type: ignore[arg-type]
        if not idx:
            if name == "train":
                raise ValidationError("Train partition is empty")
            warnings.append(f"Partition '{name}' is empty; no DataLoader created.")
            continue
        part = frame.iloc[idx]
        waves = stack_audio_column(
            part[audio_col].tolist(),
            sample_rate=cfg.sample_rate,
            max_samples=cfg.max_samples,
            source_sample_rate=cfg.source_sample_rate,
        )
        if audio_mean is not None and audio_std is not None:
            waves = apply_audio_waveform_stats(
                waves,
                np.asarray([audio_mean], dtype=np.float32),
                np.asarray([audio_std], dtype=np.float32),
            )
        y = encode_class_targets(part[target], class_labels)
        x_t = torch.as_tensor(waves, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        dataset_t = torch.utils.data.TensorDataset(x_t, y_t)
        shuffle = bool(cfg.shuffle_train and name == "train")
        loaders[name] = torch.utils.data.DataLoader(
            dataset_t,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and name == "train",
            drop_last=cfg.drop_last and name == "train",
            generator=generator if shuffle else None,
        )
        n_counts[name] = len(idx)

    feature_contract = contract.to_feature_contract()
    report = LoaderReport(
        batch_size=cfg.batch_size,
        shuffle_train=cfg.shuffle_train,
        normalize=bool(cfg.normalize_audio),
        feature_columns=feature_contract.feature_columns,
        target_column=feature_contract.target_column,
        task="classification",
        n_train=n_counts["train"],
        n_validation=n_counts["validation"],
        n_test=n_counts["test"],
        class_labels=class_labels,
        warnings=warnings,
        split_kind=split_plan.kind,
        group_column=None,
        time_column=None,
        groups_disjoint=None,
        time_order_ok=None,
    )
    return TorchLoaderBundle(
        loaders=loaders,
        contract=feature_contract,
        report=report,
        modality="speech_classify",
        input_layout=("audio",),
        speech_contract=contract,
    )


def _stub_transcribe_one(wave: np.ndarray, *, sample_rate: int) -> str:
    """Deterministic CI-safe pseudo-transcript from waveform energy fingerprint."""
    flat = np.asarray(wave, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return "[silence]"
    energy = float(np.mean(np.square(flat)))
    digest = hashlib.sha1(
        f"{sample_rate}:{energy:.6f}:{flat[:8].tobytes().hex()}".encode()
    ).hexdigest()[:8]
    bucket = int(digest[:2], 16) % 5
    tokens = ("alpha", "bravo", "charlie", "delta", "echo")
    return f"[stub-asr] {tokens[bucket]} {digest}"


def _load_transformers_asr(model_id: str) -> Any:
    transformers = require_speech_stack(feature="Speech ASR (transformers)")
    # Prefer pipeline API — Whisper-class and compatible seq2seq ASR models.
    return transformers.pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=-1,
    )


def transcribe_audio_values(
    values: list[Any],
    *,
    backend: SpeechBackend = "stub",
    model_id: str | None = None,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
) -> SpeechTranscribeResult:
    """Transcribe a list of audio path/waveform cells.

    Parameters
    ----------
    backend:
        ``stub`` (default, CI-safe) or ``transformers`` (requires ``buildml[speech]``).
    model_id:
        Hugging Face model id when ``backend="transformers"``. Defaults to a tiny
        internal-testing Whisper when omitted (still may download once).
    """
    if backend not in {"stub", "transformers"}:
        raise ValidationError("speech backend must be 'stub' or 'transformers'")
    warnings: list[str] = []
    texts: list[str] = []
    resolved_model = model_id
    limitations = (
        "Integration/finetune path — not training a foundation model from scratch.",
        "Stub backend is for tests/smoke only; do not treat stub text as real ASR.",
        "Transformers path may download weights; keep CI on stub or tiny fixtures.",
    )
    disclosures = (
        f"Speech ASR backend={backend}.",
        "Primary product path is ASR transcription; classify uses fit_speech_torch.",
    )

    if backend == "stub":
        resolved_model = resolved_model or "buildml-stub-asr"
        for value in values:
            wave = decode_audio_cell(
                value,
                sample_rate=sample_rate,
                max_samples=max_samples,
                source_sample_rate=source_sample_rate,
            )
            texts.append(_stub_transcribe_one(wave, sample_rate=sample_rate))
        return SpeechTranscribeResult(
            texts=texts,
            backend=backend,
            model_id=resolved_model,
            n_rows=len(texts),
            disclosures=disclosures,
            limitations=limitations,
            warnings=warnings,
            meta={"sample_rate": sample_rate, "max_samples": max_samples},
        )

    # transformers path
    resolved_model = (
        resolved_model
        or "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
    )
    asr = _load_transformers_asr(resolved_model)
    for value in values:
        wave = decode_audio_cell(
            value,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
        )
        mono = np.asarray(wave, dtype=np.float32).reshape(-1)
        try:
            out = asr({"array": mono, "sampling_rate": int(sample_rate)})
            if isinstance(out, dict):
                texts.append(str(out.get("text", "")).strip() or "[empty]")
            else:
                texts.append(str(out).strip() or "[empty]")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"transformers ASR failed for one cell: {exc}")
            texts.append("[asr-error]")
    return SpeechTranscribeResult(
        texts=texts,
        backend=backend,
        model_id=resolved_model,
        n_rows=len(texts),
        disclosures=disclosures + ("Uses Hugging Face transformers ASR pipeline.",),
        limitations=limitations,
        warnings=warnings,
        meta={"sample_rate": sample_rate, "max_samples": max_samples},
    )


def transcribe_from_dataset(
    dataset: Dataset,
    *,
    audio_column: str,
    backend: SpeechBackend = "stub",
    model_id: str | None = None,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
    partition: Literal["train", "validation", "test", "all"] = "all",
    split_plan: SplitPlan | None = None,
) -> SpeechTranscribeResult:
    """Transcribe an audio feature column from a Dataset (optional split slice)."""
    if audio_column not in dataset.columns:
        raise ValidationError(f"audio_column {audio_column!r} not in dataset")
    frame = dataset._ensure_pandas()
    if partition == "all":
        values = frame[audio_column].tolist()
    else:
        if split_plan is None:
            raise ValidationError(
                "partition != 'all' requires a SplitPlan (call session.split first)."
            )
        idx = list(split_plan.indices_for(partition))
        values = frame.iloc[idx][audio_column].tolist()
    result = transcribe_audio_values(
        values,
        backend=backend,
        model_id=model_id,
        sample_rate=sample_rate,
        max_samples=max_samples,
        source_sample_rate=source_sample_rate,
    )
    result.meta["audio_column"] = audio_column
    result.meta["partition"] = partition
    return result


def resolve_audio_paths(values: list[Any]) -> list[str]:
    """Helper: stringify path-like audio cells (for diagnostics)."""
    out: list[str] = []
    for value in values:
        if isinstance(value, (str, Path)):
            out.append(str(value))
        else:
            out.append(f"<waveform shape={getattr(value, 'shape', None)}>")
    return out
