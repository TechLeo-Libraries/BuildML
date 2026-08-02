"""Multimodal fusion path for BuildML DL (tabular ⊕ text ⊕ image ⊕ audio).

Builds leakage-safe loaders (train-only vocab, train-only numeric normalize,
train-only image channel stats, train-only audio amplitude stats) and a
built-in late-fusion module that concatenates available modality embeddings
before a task head.

Supported modality mixes (at least two modalities required):
- tabular + text (Pass G)
- tabular + image / text + image / tabular + text + image (Pass J)
- tabular + audio / text + audio / tabular + text + audio (Pass L)
- image + audio and richer mixes that include audio (Pass L)

Audio uses a small 1D-CNN branch — honest alpha fusion, not a speech
foundation-model product.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.dl.audio import (
    apply_audio_waveform_stats,
    fit_audio_waveform_stats,
    stack_audio_column,
)
from buildml.dl.dataset import infer_task
from buildml.dl.extras import require_torch
from buildml.dl.image import (
    apply_image_channel_stats,
    fit_image_channel_stats,
    stack_image_column,
)
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.text import fit_vocab, texts_to_ids
from buildml.dl.transforms import apply_standardize, fit_standardize, frame_to_numeric_matrix
from buildml.dl.types import FeatureContract

ModalityName = Literal["numeric", "tokens", "image", "audio"]


@dataclass(slots=True)
class MultimodalLoaderConfig:
    """Knobs for multimodal DataLoader construction."""

    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    normalize: bool = True
    normalize_images: bool = True
    normalize_audio: bool = True
    seed: int = 0
    max_len: int = 64
    max_vocab: int = 5000
    min_freq: int = 1
    image_size: tuple[int, int] = (32, 32)
    image_channels: int = 3
    audio_sample_rate: int = 16_000
    audio_max_samples: int = 16_000
    audio_source_sample_rate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_size"] = list(self.image_size)
        return payload


@dataclass(slots=True)
class MultimodalContract:
    """Schema for fused multimodal loaders / trainers."""

    numeric_columns: tuple[str, ...]
    text_column: str | None
    image_column: str | None
    audio_column: str | None
    target_column: str
    task: Literal["classification", "regression"]
    class_labels: tuple[Any, ...] = ()
    vocab: dict[str, Any] = field(default_factory=dict)
    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None
    image_mean: tuple[float, ...] | None = None
    image_std: tuple[float, ...] | None = None
    image_size: tuple[int, int] = (32, 32)
    image_channels: int = 3
    audio_mean: tuple[float, ...] | None = None
    audio_std: tuple[float, ...] | None = None
    audio_sample_rate: int = 16_000
    audio_max_samples: int = 16_000
    audio_source_sample_rate: int | None = None
    input_layout: tuple[str, ...] = ()
    modality: str = "tabular_text_fusion"

    def to_feature_contract(self) -> FeatureContract:
        cols: list[str] = list(self.numeric_columns)
        if self.text_column:
            cols.append(self.text_column)
        if self.image_column:
            cols.append(self.image_column)
        if self.audio_column:
            cols.append(self.audio_column)
        return FeatureContract(
            feature_columns=tuple(cols),
            target_column=self.target_column,
            task=self.task,
            class_labels=self.class_labels,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
        )

    def to_dict(self) -> dict[str, Any]:
        def _jsonable(value: Any) -> Any:
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        return {
            "numeric_columns": list(self.numeric_columns),
            "text_column": self.text_column,
            "image_column": self.image_column,
            "audio_column": self.audio_column,
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": [_jsonable(v) for v in self.class_labels],
            "vocab": dict(self.vocab),
            "normalize_mean": None
            if self.normalize_mean is None
            else [float(v) for v in self.normalize_mean],
            "normalize_std": None
            if self.normalize_std is None
            else [float(v) for v in self.normalize_std],
            "image_mean": None
            if self.image_mean is None
            else [float(v) for v in self.image_mean],
            "image_std": None if self.image_std is None else [float(v) for v in self.image_std],
            "image_size": list(self.image_size),
            "image_channels": self.image_channels,
            "audio_mean": None
            if self.audio_mean is None
            else [float(v) for v in self.audio_mean],
            "audio_std": None if self.audio_std is None else [float(v) for v in self.audio_std],
            "audio_sample_rate": self.audio_sample_rate,
            "audio_max_samples": self.audio_max_samples,
            "audio_source_sample_rate": self.audio_source_sample_rate,
            "input_layout": list(self.input_layout),
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultimodalContract:
        """Rebuild a contract from :meth:`to_dict` (bundle / meta round-trip)."""

        def _float_tuple(key: str) -> tuple[float, ...] | None:
            raw = payload.get(key)
            if raw is None:
                return None
            return tuple(float(v) for v in raw)

        image_size = payload.get("image_size") or (32, 32)
        layout = payload.get("input_layout") or ()
        return cls(
            numeric_columns=tuple(payload.get("numeric_columns") or ()),
            text_column=payload.get("text_column"),
            image_column=payload.get("image_column"),
            audio_column=payload.get("audio_column"),
            target_column=str(payload["target_column"]),
            task=payload["task"],
            class_labels=tuple(payload.get("class_labels") or ()),
            vocab=dict(payload.get("vocab") or {}),
            normalize_mean=_float_tuple("normalize_mean"),
            normalize_std=_float_tuple("normalize_std"),
            image_mean=_float_tuple("image_mean"),
            image_std=_float_tuple("image_std"),
            image_size=(int(image_size[0]), int(image_size[1])),
            image_channels=int(payload.get("image_channels") or 3),
            audio_mean=_float_tuple("audio_mean"),
            audio_std=_float_tuple("audio_std"),
            audio_sample_rate=int(payload.get("audio_sample_rate") or 16_000),
            audio_max_samples=int(payload.get("audio_max_samples") or 16_000),
            audio_source_sample_rate=(
                None
                if payload.get("audio_source_sample_rate") is None
                else int(payload["audio_source_sample_rate"])
            ),
            input_layout=tuple(str(x) for x in layout),
            modality=str(payload.get("modality") or "tabular_text_fusion"),
        )


def _modality_name(
    *,
    has_numeric: bool,
    has_text: bool,
    has_image: bool,
    has_audio: bool,
) -> str:
    parts: list[str] = []
    if has_numeric:
        parts.append("tabular")
    if has_text:
        parts.append("text")
    if has_image:
        parts.append("image")
    if has_audio:
        parts.append("audio")
    if not parts:
        raise ValidationError("Multimodal path requires at least one modality")
    if len(parts) == 1:
        raise ValidationError(
            "Multimodal fusion needs at least two modalities "
            "(tabular/text/image/audio). For a single modality use make_torch_loaders "
            "or make_text_torch_loaders."
        )
    return "_".join(parts) + "_fusion"


def _resolve_multimodal_columns(
    dataset: Dataset,
    *,
    text_column: str | None,
    numeric_columns: list[str] | None,
    image_column: str | None,
    audio_column: str | None,
) -> tuple[list[str], str | None, str | None, str | None, str]:
    target = dataset.require_target()
    frame = dataset._ensure_pandas()
    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    if not feature_cols:
        skip = {
            *dataset.role_columns(ColumnRole.TARGET),
            *dataset.role_columns(ColumnRole.ID),
            *dataset.role_columns(ColumnRole.IGNORE),
            *dataset.role_columns(ColumnRole.GROUP),
            *dataset.role_columns(ColumnRole.TIME),
            *dataset.role_columns(ColumnRole.WEIGHT),
        }
        feature_cols = [c for c in dataset.columns if c not in skip and c != target]

    if image_column is not None and image_column not in dataset.columns:
        raise ValidationError(f"image_column {image_column!r} not in dataset columns")
    if audio_column is not None and audio_column not in dataset.columns:
        raise ValidationError(f"audio_column {audio_column!r} not in dataset columns")
    if (
        image_column is not None
        and audio_column is not None
        and image_column == audio_column
    ):
        raise ValidationError("image_column and audio_column must be distinct")

    media_cols = {c for c in (image_column, audio_column) if c is not None}
    object_like = [
        c
        for c in feature_cols
        if c not in media_cols
        and (frame[c].dtype == object or str(frame[c].dtype).startswith("string"))
    ]
    # Prefer string-like columns for text inference; path/array media columns
    # are excluded when image_column / audio_column are set.
    if text_column is None and not media_cols:
        textish = [c for c in object_like if _looks_like_text_column(frame[c])]
        mediaish = [c for c in object_like if c not in textish]
        if mediaish and not textish:
            raise ValidationError(
                "Object feature column(s) look like media paths/arrays "
                f"({mediaish}); pass audio_column= / image_column= explicitly "
                "(or text_column= if they are truly text). Refusing to tokenize "
                "media paths as tabular⊕text fusion."
            )
        if len(textish) != 1:
            raise ValidationError(
                "Multimodal path needs exactly one text feature column when "
                f"text_column is omitted; found {textish or object_like or 'none'}. "
                "Pass text_column= explicitly, or pass image_column=/audio_column= "
                "for media multimodal."
            )
        text_column = textish[0]
    elif text_column is None and media_cols:
        # Optional text when media is present: infer only if exactly one
        # remaining string-like feature that looks like text (not media paths).
        stringish = [
            c
            for c in object_like
            if c not in media_cols and _looks_like_text_column(frame[c])
        ]
        text_column = stringish[0] if len(stringish) == 1 else None
    elif text_column is not None and text_column not in dataset.columns:
        raise ValidationError(f"text_column {text_column!r} not in dataset columns")

    reserved = {c for c in (text_column, image_column, audio_column) if c is not None}
    if numeric_columns is None:
        numeric_columns = [
            c
            for c in feature_cols
            if c not in reserved and pd.api.types.is_numeric_dtype(frame[c])
        ]
    missing = [c for c in numeric_columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"numeric_columns missing from dataset: {missing}")

    has_numeric = bool(numeric_columns)
    has_text = text_column is not None
    has_image = image_column is not None
    has_audio = audio_column is not None
    if not has_image and not has_audio:
        # Legacy tabular+text path: both required.
        if not has_text:
            raise ValidationError(
                "Multimodal fusion without image_column/audio_column requires a text column."
            )
        if not has_numeric:
            raise ValidationError(
                "Multimodal fusion requires at least one numeric feature column "
                "in addition to the text column (or pass image_column=/audio_column=)."
            )
    else:
        # Media multimodal needs at least one other modality (tabular, text,
        # or the other media column).
        n_modalities = sum([has_numeric, has_text, has_image, has_audio])
        if n_modalities < 2:
            raise ValidationError(
                "Image/audio multimodal requires tabular numeric features and/or a "
                "text column (and/or the other media modality) to fuse with "
                "image_column/audio_column."
            )
    return list(numeric_columns), text_column, image_column, audio_column, target


def _looks_like_media_cell(value: Any) -> bool:
    """True when a cell looks like an image/audio path or array payload."""
    if isinstance(value, (np.ndarray, list, tuple, Path)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return False
    media_ext = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".webp",
        ".tif",
        ".tiff",
        ".wav",
        ".flac",
        ".ogg",
        ".mp3",
        ".aiff",
        ".aif",
    )
    return text.endswith(media_ext) or "/" in text or "\\" in text


def _looks_like_text_column(series: pd.Series) -> bool:
    """Heuristic: string cells that are not filesystem-looking media paths/arrays."""
    sample = series.dropna().head(8).tolist()
    if not sample:
        return False
    media_hits = sum(1 for v in sample if _looks_like_media_cell(v))
    return media_hits < max(1, len(sample) // 2)


def build_multimodal_fusion(
    n_numeric: int = 0,
    vocab_size: int = 0,
    *,
    image_channels: int = 0,
    image_size: tuple[int, int] = (32, 32),
    audio_channels: int = 0,
    audio_samples: int = 16_000,
    task: str = "classification",
    n_classes: int = 2,
    tabular_hidden: tuple[int, ...] = (32,),
    text_embed_dim: int = 32,
    text_hidden: int = 32,
    image_hidden: int = 32,
    audio_hidden: int = 32,
    fusion_hidden: int = 64,
    dropout: float = 0.1,
    padding_idx: int = 0,
) -> Any:
    """Late-fusion module over any mix of tabular / text / image / audio branches."""
    torch = require_torch(feature="MultimodalFusion")
    has_numeric = int(n_numeric) > 0
    has_text = int(vocab_size) >= 2
    has_image = int(image_channels) > 0
    has_audio = int(audio_channels) > 0
    n_mods = sum([has_numeric, has_text, has_image, has_audio])
    if n_mods < 2:
        raise ValidationError(
            "build_multimodal_fusion requires at least two modalities among "
            "tabular, text, image, and audio"
        )
    if not has_image and not has_audio and not (has_numeric and has_text):
        raise ValidationError(
            "build_multimodal_fusion without image/audio requires tabular+text"
        )
    if task not in {"classification", "regression"}:
        raise ValidationError("task must be 'classification' or 'regression'")
    if task == "classification" and n_classes < 2:
        raise ValidationError("n_classes must be >= 2 for classification")

    layout: list[str] = []
    if has_numeric:
        layout.append("numeric")
    if has_text:
        layout.append("tokens")
    if has_image:
        layout.append("image")
    if has_audio:
        layout.append("audio")
    modality = _modality_name(
        has_numeric=has_numeric,
        has_text=has_text,
        has_image=has_image,
        has_audio=has_audio,
    )
    img_h, img_w = int(image_size[0]), int(image_size[1])
    n_audio_samples = int(audio_samples)

    class _MultimodalFusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layout = tuple(layout)
            self.modality = modality
            self.task = task
            self.n_numeric = int(n_numeric) if has_numeric else 0
            self.vocab_size = int(vocab_size) if has_text else 0
            self.image_channels = int(image_channels) if has_image else 0
            self.image_size = (img_h, img_w)
            self.audio_channels = int(audio_channels) if has_audio else 0
            self.audio_samples = n_audio_samples if has_audio else 0
            self.n_classes = int(n_classes) if task == "classification" else 1
            self.padding_idx = int(padding_idx)

            fused_in = 0
            if has_numeric:
                layers: list[Any] = []
                prev = int(n_numeric)
                for width in tabular_hidden:
                    layers.append(torch.nn.Linear(prev, int(width)))
                    layers.append(torch.nn.ReLU())
                    if dropout > 0:
                        layers.append(torch.nn.Dropout(p=float(dropout)))
                    prev = int(width)
                self.tabular = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
                self.tabular_out = prev if layers else int(n_numeric)
                fused_in += self.tabular_out
            else:
                self.tabular = None
                self.tabular_out = 0

            if has_text:
                self.embedding = torch.nn.Embedding(
                    int(vocab_size), int(text_embed_dim), padding_idx=int(padding_idx)
                )
                self.text_proj = torch.nn.Sequential(
                    torch.nn.Linear(int(text_embed_dim), int(text_hidden)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=float(dropout)),
                )
                fused_in += int(text_hidden)
            else:
                self.embedding = None
                self.text_proj = None

            if has_image:
                mid = max(8, int(image_hidden))
                self.image_net = torch.nn.Sequential(
                    torch.nn.Conv2d(int(image_channels), mid, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(mid, mid, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(mid, int(image_hidden)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=float(dropout)),
                )
                fused_in += int(image_hidden)
            else:
                self.image_net = None

            if has_audio:
                mid_a = max(8, int(audio_hidden))
                self.audio_net = torch.nn.Sequential(
                    torch.nn.Conv1d(int(audio_channels), mid_a, kernel_size=9, padding=4),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(mid_a, mid_a, kernel_size=9, padding=4),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool1d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(mid_a, int(audio_hidden)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=float(dropout)),
                )
                fused_in += int(audio_hidden)
            else:
                self.audio_net = None

            out = int(n_classes) if task == "classification" else 1
            self.head = torch.nn.Sequential(
                torch.nn.Linear(fused_in, int(fusion_hidden)),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=float(dropout)),
                torch.nn.Linear(int(fusion_hidden), out),
            )

        def forward(self, *args: Any) -> Any:
            # Dual calling convention:
            # - train/eval: module((t0, t1, ...))  [single packed tuple]
            # - TorchScript/ONNX: module(t0, t1, ...)  [unpacked args]
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                args = tuple(args[0])
            if len(args) != len(self.input_layout):
                raise ValidationError(
                    f"MultimodalFusion expects {len(self.input_layout)} input(s) "
                    f"in order {self.input_layout}; got {len(args)} "
                    f"(types={[type(a).__name__ for a in args]})"
                )
            pieces: list[Any] = []
            for name, tensor in zip(self.input_layout, args, strict=True):
                if name == "numeric":
                    assert self.tabular is not None
                    pieces.append(self.tabular(tensor))
                elif name == "tokens":
                    assert self.embedding is not None and self.text_proj is not None
                    mask = (tensor != self.padding_idx).unsqueeze(-1).float()
                    embedded = self.embedding(tensor) * mask
                    denom = mask.sum(dim=1).clamp(min=1.0)
                    pooled = embedded.sum(dim=1) / denom
                    pieces.append(self.text_proj(pooled))
                elif name == "image":
                    assert self.image_net is not None
                    pieces.append(self.image_net(tensor))
                elif name == "audio":
                    assert self.audio_net is not None
                    pieces.append(self.audio_net(tensor))
                else:
                    raise ValidationError(f"Unknown modality in layout: {name}")
            return self.head(torch.cat(pieces, dim=1))

    return _MultimodalFusion()


def make_multimodal_loaders(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    text_column: str | None = None,
    numeric_columns: list[str] | None = None,
    image_column: str | None = None,
    audio_column: str | None = None,
    config: MultimodalLoaderConfig | None = None,
    task: Literal["classification", "regression", "auto"] = "auto",
) -> TorchLoaderBundle:
    """Build fused multimodal DataLoaders with train-only fit stats.

    Batch layout is ``(*modality_tensors, y)`` in order
    ``numeric`` → ``tokens`` → ``image`` → ``audio`` for whichever modalities
    are present. Pair with :func:`build_multimodal_fusion` (or
    :meth:`Session.fit_torch` auto-build).
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    require_torch(feature="Multimodal Torch DataLoaders")
    cfg = config or MultimodalLoaderConfig()
    if cfg.batch_size < 1:
        raise ValidationError("batch_size must be >= 1")
    if cfg.image_channels not in {1, 3}:
        raise ValidationError("image_channels must be 1 or 3")
    if cfg.audio_sample_rate < 1 or cfg.audio_max_samples < 1:
        raise ValidationError("audio_sample_rate and audio_max_samples must be positive")

    numeric_cols, text_col, image_col, audio_col, target = _resolve_multimodal_columns(
        dataset,
        text_column=text_column,
        numeric_columns=numeric_columns,
        image_column=image_column,
        audio_column=audio_column,
    )
    frame = dataset._ensure_pandas()
    train_idx = list(split_plan.indices_for("train"))
    if not train_idx:
        raise ValidationError("Train partition is empty; cannot build multimodal loaders")

    y_train = frame.iloc[train_idx][target]
    if not pd.api.types.is_numeric_dtype(y_train):
        raise ValidationError(
            f"Target '{target}' must be numeric for the multimodal Torch path "
            "(encode labels to integers first)."
        )
    resolved_task = infer_task(y_train, task)
    class_labels = (
        tuple(sorted(pd.unique(y_train))) if resolved_task == "classification" else ()
    )

    has_numeric = bool(numeric_cols)
    has_text = text_col is not None
    has_image = image_col is not None
    has_audio = audio_col is not None
    modality = _modality_name(
        has_numeric=has_numeric,
        has_text=has_text,
        has_image=has_image,
        has_audio=has_audio,
    )
    layout: list[str] = []
    if has_numeric:
        layout.append("numeric")
    if has_text:
        layout.append("tokens")
    if has_image:
        layout.append("image")
    if has_audio:
        layout.append("audio")

    vocab = None
    if has_text:
        assert text_col is not None
        train_texts = frame.iloc[train_idx][text_col].astype(str).tolist()
        vocab = fit_vocab(
            train_texts,
            max_vocab=cfg.max_vocab,
            min_freq=cfg.min_freq,
            max_len=cfg.max_len,
        )

    mean = std = None
    if has_numeric and cfg.normalize:
        x_train = frame_to_numeric_matrix(frame.iloc[train_idx], numeric_cols)
        mean, std = fit_standardize(x_train)

    img_mean = img_std = None
    if has_image and cfg.normalize_images:
        assert image_col is not None
        train_images = stack_image_column(
            frame.iloc[train_idx][image_col].tolist(),
            size=cfg.image_size,
            channels=cfg.image_channels,
        )
        img_mean, img_std = fit_image_channel_stats(train_images)

    aud_mean = aud_std = None
    if has_audio and cfg.normalize_audio:
        assert audio_col is not None
        train_audio, train_audio_lengths = stack_audio_column(
            frame.iloc[train_idx][audio_col].tolist(),
            sample_rate=cfg.audio_sample_rate,
            max_samples=cfg.audio_max_samples,
            source_sample_rate=cfg.audio_source_sample_rate,
            return_lengths=True,
        )
        aud_mean, aud_std = fit_audio_waveform_stats(
            train_audio, lengths=train_audio_lengths
        )

    torch = require_torch(feature="Multimodal Torch DataLoaders")
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loaders: dict[str, Any] = {}
    warnings: list[str] = [
        f"Multimodal fusion ({modality}): fit stats use train only; "
        f"batch layout is ({', '.join(layout)}, y).",
    ]
    if has_image:
        warnings.append(
            "Image cells accept path strings (Pillow) or array/list tensors."
        )
    if has_audio:
        warnings.append(
            "Audio cells accept path strings (soundfile) or waveform arrays; "
            "short clips are repeat-padded to audio_max_samples "
            "(keeps AdaptiveAvgPool1d informative without a lengths tensor in "
            "forward/export); fusion uses a small 1D-CNN branch "
            "(not a speech foundation model)."
        )
    n_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    for name in ("train", "validation", "test"):
        idx = list(split_plan.indices_for(name))  # type: ignore[arg-type]
        if not idx:
            if name == "train":
                raise ValidationError("Train partition is empty")
            warnings.append(f"Partition '{name}' is empty; no DataLoader created.")
            continue
        part = frame.iloc[idx]
        tensors: list[Any] = []
        if has_numeric:
            x = frame_to_numeric_matrix(part, numeric_cols)
            if cfg.normalize and mean is not None and std is not None:
                x = apply_standardize(x, mean, std)
            tensors.append(torch.as_tensor(x, dtype=torch.float32))
        if has_text:
            assert text_col is not None and vocab is not None
            tokens = texts_to_ids(part[text_col].astype(str).tolist(), vocab)
            tensors.append(torch.as_tensor(tokens, dtype=torch.long))
        if has_image:
            assert image_col is not None
            images = stack_image_column(
                part[image_col].tolist(),
                size=cfg.image_size,
                channels=cfg.image_channels,
            )
            if cfg.normalize_images and img_mean is not None and img_std is not None:
                images = apply_image_channel_stats(images, img_mean, img_std)
            tensors.append(torch.as_tensor(images, dtype=torch.float32))
        if has_audio:
            assert audio_col is not None
            audio = stack_audio_column(
                part[audio_col].tolist(),
                sample_rate=cfg.audio_sample_rate,
                max_samples=cfg.audio_max_samples,
                source_sample_rate=cfg.audio_source_sample_rate,
            )
            if cfg.normalize_audio and aud_mean is not None and aud_std is not None:
                audio = apply_audio_waveform_stats(audio, aud_mean, aud_std)
            tensors.append(torch.as_tensor(audio, dtype=torch.float32))
        y = part[target].to_numpy(dtype=np.float64, copy=True)
        if np.isnan(y).any():
            raise ValidationError("Target contains NaN; clean labels before multimodal loaders")
        if resolved_task == "classification":
            y_t = torch.as_tensor(y, dtype=torch.long)
        else:
            y_t = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)
        dataset_t = torch.utils.data.TensorDataset(*tensors, y_t)
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

    contract = MultimodalContract(
        numeric_columns=tuple(numeric_cols),
        text_column=text_col,
        image_column=image_col,
        audio_column=audio_col,
        target_column=target,
        task=resolved_task,
        class_labels=class_labels,
        vocab={} if vocab is None else vocab.to_dict(),
        normalize_mean=None if mean is None else tuple(float(v) for v in mean),
        normalize_std=None if std is None else tuple(float(v) for v in std),
        image_mean=None if img_mean is None else tuple(float(v) for v in img_mean),
        image_std=None if img_std is None else tuple(float(v) for v in img_std),
        image_size=tuple(int(x) for x in cfg.image_size),  # type: ignore[arg-type]
        image_channels=int(cfg.image_channels),
        audio_mean=None if aud_mean is None else tuple(float(v) for v in aud_mean),
        audio_std=None if aud_std is None else tuple(float(v) for v in aud_std),
        audio_sample_rate=int(cfg.audio_sample_rate),
        audio_max_samples=int(cfg.audio_max_samples),
        audio_source_sample_rate=(
            None
            if cfg.audio_source_sample_rate is None
            else int(cfg.audio_source_sample_rate)
        ),
        input_layout=tuple(layout),
        modality=modality,
    )
    feature_contract = contract.to_feature_contract()
    report = LoaderReport(
        batch_size=cfg.batch_size,
        shuffle_train=cfg.shuffle_train,
        normalize=cfg.normalize,
        feature_columns=feature_contract.feature_columns,
        target_column=feature_contract.target_column,
        task=resolved_task,
        n_train=n_counts["train"],
        n_validation=n_counts["validation"],
        n_test=n_counts["test"],
        class_labels=class_labels,
        warnings=warnings,
        split_kind=split_plan.kind,
    )
    return TorchLoaderBundle(
        loaders=loaders,
        contract=feature_contract,
        report=report,
        multimodal_contract=contract,
        text_vocab=vocab,
        modality=modality,
        input_layout=tuple(layout),
    )


__all__ = [
    "MultimodalContract",
    "MultimodalLoaderConfig",
    "build_multimodal_fusion",
    "make_multimodal_loaders",
]
