"""Text / sequence tensor path for BuildML DL (non-tabular modality).

Tokenizes text with a fold-safe vocabulary fit on train only, builds padded
token-id loaders, and pairs with :func:`buildml.dl.models.build_text_classifier`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.dl.extras import require_torch
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.types import FeatureContract, LoaderConfig

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class TextVocab:
    """Word vocabulary with reserved pad/unk tokens."""

    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]
    pad_id: int = 0
    unk_id: int = 1
    max_len: int = 64

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_to_id": dict(self.token_to_id),
            "id_to_token": list(self.id_to_token),
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
        }


@dataclass(slots=True)
class TextLoaderConfig:
    """Knobs for text DataLoader construction."""

    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    seed: int = 0
    max_len: int = 64
    max_vocab: int = 5000
    min_freq: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TextContract:
    """Schema carried with text Torch loaders / trainers."""

    text_column: str
    target_column: str
    task: Literal["classification"] = "classification"
    class_labels: tuple[Any, ...] = ()
    vocab: dict[str, Any] = field(default_factory=dict)
    modality: str = "text_tokens"

    def to_feature_contract(self) -> FeatureContract:
        return FeatureContract(
            feature_columns=(self.text_column,),
            target_column=self.target_column,
            task="classification",
            class_labels=self.class_labels,
            normalize_mean=None,
            normalize_std=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_column": self.text_column,
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": list(self.class_labels),
            "vocab": dict(self.vocab),
            "modality": self.modality,
        }


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer."""
    return _TOKEN_RE.findall(str(text).lower())


def fit_vocab(
    texts: list[str],
    *,
    max_vocab: int = 5000,
    min_freq: int = 1,
    max_len: int = 64,
) -> TextVocab:
    """Fit a vocabulary on train texts only (pad=0, unk=1)."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    items = [(t, c) for t, c in counts.items() if c >= min_freq]
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    # Reserve 0=pad, 1=unk
    tokens = ["<pad>", "<unk>"] + [t for t, _ in items[: max(0, max_vocab - 2)]]
    token_to_id = {t: i for i, t in enumerate(tokens)}
    return TextVocab(
        token_to_id=token_to_id,
        id_to_token=tuple(tokens),
        pad_id=0,
        unk_id=1,
        max_len=int(max_len),
    )


def texts_to_ids(texts: list[str], vocab: TextVocab) -> np.ndarray:
    """Encode texts to a padded ``int64`` matrix ``[n, max_len]``."""
    matrix = np.full((len(texts), vocab.max_len), vocab.pad_id, dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in tokenize(text)]
        ids = ids[: vocab.max_len]
        if ids:
            matrix[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
    return matrix


def _resolve_text_target(dataset: Dataset, text_column: str | None) -> tuple[str, str]:
    target = dataset.require_target()
    if text_column is not None:
        if text_column not in dataset.columns:
            raise ValidationError(f"text_column {text_column!r} not in dataset columns")
        return text_column, target
    feature_cols = dataset.role_columns(ColumnRole.FEATURE)
    object_like = [
        c
        for c in feature_cols
        if dataset._ensure_pandas()[c].dtype == object
        or str(dataset._ensure_pandas()[c].dtype).startswith("string")
    ]
    if len(object_like) == 1:
        return object_like[0], target
    if len(object_like) > 1:
        raise ValidationError(
            "Multiple text-like feature columns found; pass text_column= explicitly. "
            f"Candidates: {object_like}"
        )
    raise ValidationError(
        "No text feature column found. Set a string feature role or pass text_column=."
    )


def make_text_loaders(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    text_column: str | None = None,
    config: TextLoaderConfig | None = None,
) -> TorchLoaderBundle:
    """Build token-id DataLoaders for text classification.

    Vocabulary is fit on the **train** partition only (leakage-safe).
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    require_torch(feature="Text Torch DataLoaders")
    cfg = config or TextLoaderConfig()
    if cfg.batch_size < 1:
        raise ValidationError("batch_size must be >= 1")

    text_col, target = _resolve_text_target(dataset, text_column)
    frame = dataset._ensure_pandas()
    train_idx = list(split_plan.indices_for("train"))
    if not train_idx:
        raise ValidationError("Train partition is empty; cannot build text loaders")
    train_texts = frame.iloc[train_idx][text_col].astype(str).tolist()
    vocab = fit_vocab(
        train_texts,
        max_vocab=cfg.max_vocab,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
    )

    y_train = frame.iloc[train_idx][target]
    if not pd.api.types.is_numeric_dtype(y_train):
        raise ValidationError(
            f"Target '{target}' must be numeric class ids for the text Torch path "
            "(encode labels to integers first)."
        )
    class_labels = tuple(sorted(pd.unique(y_train)))
    contract = TextContract(
        text_column=text_col,
        target_column=target,
        class_labels=class_labels,
        vocab=vocab.to_dict(),
    )

    torch = require_torch(feature="Text Torch DataLoaders")
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loaders: dict[str, Any] = {}
    warnings: list[str] = [
        "Text modality: vocabulary fit on train only; pad/unk reserved; "
        "this path is sequence/text classification, not tabular numeric tensors.",
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
        x = texts_to_ids(part[text_col].astype(str).tolist(), vocab)
        y = part[target].to_numpy(dtype=np.int64, copy=True)
        if np.isnan(y.astype(np.float64)).any():
            raise ValidationError("Target contains NaN; clean labels before text loaders")
        x_t = torch.as_tensor(x, dtype=torch.long)
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
        normalize=False,
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
        text_contract=contract,
        text_vocab=vocab,
        modality="text_tokens",
    )


def loader_config_from_text(cfg: TextLoaderConfig) -> LoaderConfig:
    """Map text loader knobs onto the shared LoaderConfig shape."""
    return LoaderConfig(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle_train=cfg.shuffle_train,
        drop_last=cfg.drop_last,
        normalize=False,
        seed=cfg.seed,
    )
