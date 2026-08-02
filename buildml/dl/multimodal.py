"""Tabular + text multimodal fusion path for BuildML DL.

Builds leakage-safe loaders (train-only vocab + train-only numeric normalize)
and a built-in late-fusion module that concatenates tabular and text embeddings
before a task head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.dl.dataset import infer_task
from buildml.dl.extras import require_torch
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.text import fit_vocab, texts_to_ids, tokenize
from buildml.dl.transforms import apply_standardize, fit_standardize, frame_to_numeric_matrix
from buildml.dl.types import FeatureContract


@dataclass(slots=True)
class MultimodalLoaderConfig:
    """Knobs for multimodal tabular+text DataLoader construction."""

    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    normalize: bool = True
    seed: int = 0
    max_len: int = 64
    max_vocab: int = 5000
    min_freq: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MultimodalContract:
    """Schema for tabular numeric features + text tokens fused in one model."""

    numeric_columns: tuple[str, ...]
    text_column: str
    target_column: str
    task: Literal["classification", "regression"]
    class_labels: tuple[Any, ...] = ()
    vocab: dict[str, Any] = field(default_factory=dict)
    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None
    modality: str = "tabular_text_fusion"

    def to_feature_contract(self) -> FeatureContract:
        return FeatureContract(
            feature_columns=self.numeric_columns + (self.text_column,),
            target_column=self.target_column,
            task=self.task,
            class_labels=self.class_labels,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_columns": list(self.numeric_columns),
            "text_column": self.text_column,
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": list(self.class_labels),
            "vocab": dict(self.vocab),
            "normalize_mean": None
            if self.normalize_mean is None
            else list(self.normalize_mean),
            "normalize_std": None if self.normalize_std is None else list(self.normalize_std),
            "modality": self.modality,
        }


def _resolve_multimodal_columns(
    dataset: Dataset,
    *,
    text_column: str | None,
    numeric_columns: list[str] | None,
) -> tuple[list[str], str, str]:
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

    object_like = [
        c
        for c in feature_cols
        if frame[c].dtype == object or str(frame[c].dtype).startswith("string")
    ]
    if text_column is None:
        if len(object_like) != 1:
            raise ValidationError(
                "Multimodal path needs exactly one text feature column when "
                f"text_column is omitted; found {object_like or 'none'}. "
                "Pass text_column= explicitly."
            )
        text_column = object_like[0]
    elif text_column not in dataset.columns:
        raise ValidationError(f"text_column {text_column!r} not in dataset columns")

    if numeric_columns is None:
        numeric_columns = [
            c
            for c in feature_cols
            if c != text_column and pd.api.types.is_numeric_dtype(frame[c])
        ]
    if not numeric_columns:
        raise ValidationError(
            "Multimodal fusion requires at least one numeric feature column "
            "in addition to the text column."
        )
    missing = [c for c in numeric_columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"numeric_columns missing from dataset: {missing}")
    return list(numeric_columns), text_column, target


def build_multimodal_fusion(
    n_numeric: int,
    vocab_size: int,
    *,
    task: str = "classification",
    n_classes: int = 2,
    tabular_hidden: tuple[int, ...] = (32,),
    text_embed_dim: int = 32,
    text_hidden: int = 32,
    fusion_hidden: int = 64,
    dropout: float = 0.1,
    padding_idx: int = 0,
) -> Any:
    """Late-fusion module: tabular MLP branch + masked-mean text branch → head."""
    torch = require_torch(feature="MultimodalFusion")
    if n_numeric < 1:
        raise ValidationError("n_numeric must be >= 1")
    if vocab_size < 2:
        raise ValidationError("vocab_size must be >= 2")
    if task not in {"classification", "regression"}:
        raise ValidationError("task must be 'classification' or 'regression'")
    if task == "classification" and n_classes < 2:
        raise ValidationError("n_classes must be >= 2 for classification")

    class _MultimodalFusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
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
            self.embedding = torch.nn.Embedding(
                int(vocab_size), int(text_embed_dim), padding_idx=int(padding_idx)
            )
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(int(text_embed_dim), int(text_hidden)),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=float(dropout)),
            )
            fused_in = self.tabular_out + int(text_hidden)
            out = int(n_classes) if task == "classification" else 1
            self.head = torch.nn.Sequential(
                torch.nn.Linear(fused_in, int(fusion_hidden)),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=float(dropout)),
                torch.nn.Linear(int(fusion_hidden), out),
            )
            self.padding_idx = int(padding_idx)
            self.task = task
            self.n_numeric = int(n_numeric)
            self.vocab_size = int(vocab_size)
            self.n_classes = int(n_classes) if task == "classification" else 1
            self.modality = "tabular_text_fusion"

        def forward(self, inputs: Any) -> Any:
            if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
                x_tab, token_ids = inputs
            else:
                raise ValidationError(
                    "MultimodalFusion expects inputs=(x_numeric, token_ids); "
                    f"got type={type(inputs).__name__}"
                )
            tab = self.tabular(x_tab)
            mask = (token_ids != self.padding_idx).unsqueeze(-1).float()
            embedded = self.embedding(token_ids) * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = embedded.sum(dim=1) / denom
            text = self.text_proj(pooled)
            return self.head(torch.cat([tab, text], dim=1))

    return _MultimodalFusion()


def make_multimodal_loaders(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    text_column: str | None = None,
    numeric_columns: list[str] | None = None,
    config: MultimodalLoaderConfig | None = None,
    task: Literal["classification", "regression", "auto"] = "auto",
) -> TorchLoaderBundle:
    """Build fused tabular+text DataLoaders with train-only vocab and normalize.

    Each batch is ``(x_numeric, token_ids, y)``. Pair with
    :func:`build_multimodal_fusion` (or :meth:`Session.fit_torch` auto-build).
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    require_torch(feature="Multimodal Torch DataLoaders")
    cfg = config or MultimodalLoaderConfig()
    if cfg.batch_size < 1:
        raise ValidationError("batch_size must be >= 1")

    numeric_cols, text_col, target = _resolve_multimodal_columns(
        dataset, text_column=text_column, numeric_columns=numeric_columns
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

    train_texts = frame.iloc[train_idx][text_col].astype(str).tolist()
    vocab = fit_vocab(
        train_texts,
        max_vocab=cfg.max_vocab,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
    )

    x_train = frame_to_numeric_matrix(frame.iloc[train_idx], numeric_cols)
    mean = std = None
    if cfg.normalize:
        mean, std = fit_standardize(x_train)

    torch = require_torch(feature="Multimodal Torch DataLoaders")
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loaders: dict[str, Any] = {}
    warnings: list[str] = [
        "Multimodal fusion: vocabulary and normalize stats fit on train only; "
        "batches are (x_numeric, token_ids, y).",
    ]
    n_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    for name in ("train", "validation", "test"):
        idx = list(split_plan.indices_for(name))  # type: ignore[arg-type]
        if not idx:
            if name == "train":
                raise ValidationError("Train partition is empty")
            warnings.append(f"Partition '{name}' is empty; no DataLoader created.")
            continue
        part = frame.iloc[idx]
        x = frame_to_numeric_matrix(part, numeric_cols)
        if cfg.normalize and mean is not None and std is not None:
            x = apply_standardize(x, mean, std)
        tokens = texts_to_ids(part[text_col].astype(str).tolist(), vocab)
        y = part[target].to_numpy(dtype=np.float64, copy=True)
        if np.isnan(y).any():
            raise ValidationError("Target contains NaN; clean labels before multimodal loaders")
        x_t = torch.as_tensor(x, dtype=torch.float32)
        tok_t = torch.as_tensor(tokens, dtype=torch.long)
        if resolved_task == "classification":
            y_t = torch.as_tensor(y, dtype=torch.long)
        else:
            y_t = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)
        dataset_t = torch.utils.data.TensorDataset(x_t, tok_t, y_t)
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
        target_column=target,
        task=resolved_task,
        class_labels=class_labels,
        vocab=vocab.to_dict(),
        normalize_mean=None if mean is None else tuple(float(v) for v in mean),
        normalize_std=None if std is None else tuple(float(v) for v in std),
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
    bundle = TorchLoaderBundle(loaders=loaders, contract=feature_contract, report=report)
    bundle.multimodal_contract = contract  # type: ignore[attr-defined]
    bundle.text_vocab = vocab  # type: ignore[attr-defined]
    bundle.modality = "tabular_text_fusion"  # type: ignore[attr-defined]
    return bundle


# Keep tokenize import used for discoverability / re-export symmetry with text.py
__all__ = [
    "MultimodalContract",
    "MultimodalLoaderConfig",
    "build_multimodal_fusion",
    "make_multimodal_loaders",
    "tokenize",
]
