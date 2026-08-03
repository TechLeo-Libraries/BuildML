"""Turn text columns into the token sequences a neural network can read.

Networks work on numbers, so text has to become numbers first. The approach here
is the classic one: split each document into words, assign every word an
integer, and represent a document as the sequence of those integers, padded to a
common length so documents can be batched.

The vocabulary is the fitted part, and it is fitted on the training partition
alone. A vocabulary built across all partitions would give the model a slot for
every word in the test set, which both leaks and misleads — at deployment, words
it has never seen will arrive, and a model that never encountered an unknown
token during training has no idea what to do with one. Reserving ``<unk>`` and
training with it present is what makes that case survivable.

Two ids are reserved. Zero is padding, excluded from pooling so document length
does not affect the representation. One is the unknown token, which every
out-of-vocabulary word maps to.

This is the token-id path, distinct from :mod:`buildml.nlp`, which offers TF-IDF
and transformer representations for classical models. Use this when you want to
train a network on text end to end.

See Also
--------
buildml.dl.models.build_text_classifier : The matching model.
buildml.nlp : Classical text representations and tasks.
buildml.dl.multimodal : Text combined with other modalities.
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
from buildml.dl.labels import encode_class_targets, fit_class_labels
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.types import FeatureContract, LoaderConfig

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class TextVocab:
    """The word-to-integer mapping a text model was trained with.

    Attributes
    ----------
    token_to_id:
        Word to integer id. Words absent from this map become ``unk_id``.
    id_to_token:
        The inverse, as a sequence indexed by id. Position 0 is ``<pad>``,
        position 1 is ``<unk>``.
    pad_id:
        The padding id, 0. Masked out of pooling.
    unk_id:
        The unknown-word id, 1.
    max_len:
        Sequence length. Longer documents are truncated, shorter are padded.

    Notes
    -----
    **This must be persisted with the model.** Token ids are arbitrary — they
    depend on the training corpus's word frequencies — so a model paired with a
    different vocabulary is reading a different language.

    **Truncation at ``max_len`` silently discards the tail.** A model with
    ``max_len=64`` sees only the first 64 words of a long document, and nothing
    reports how much was cut.

    See Also
    --------
    fit_vocab : Builds this.
    texts_to_ids : Applies it.
    """

    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]
    pad_id: int = 0
    unk_id: int = 1
    max_len: int = 64

    @property
    def vocab_size(self) -> int:
        """Number of ids, including the two reserved ones.

        This is the width the embedding layer needs.

        Returns
        -------
        int
            Vocabulary size.
        """
        return len(self.id_to_token)

    def to_dict(self) -> dict[str, Any]:
        """Return the vocabulary as JSON-safe values.

        Complete enough to reconstruct — which is how a saved text model gets
        its vocabulary back when loaders are rebuilt.

        Returns
        -------
        dict
            Both directions of the mapping, the reserved ids, the sequence
            length, and the vocabulary size.
        """
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
    """Settings for building text loaders.

    Attributes
    ----------
    batch_size:
        Documents per batch.
    num_workers:
        Background loading processes. Rarely needed here, since token ids are
        already in memory.
    pin_memory:
        Page-lock batches for faster GPU transfer. Train loader only.
    shuffle_train:
        Shuffle the training loader.
    drop_last:
        Discard a final short batch.
    seed:
        Controls shuffling.
    max_len:
        Sequence length. Long enough to capture what matters, short enough to
        avoid padding most documents into mostly-padding.
    max_vocab:
        Cap on vocabulary size, including the two reserved ids. The most
        frequent training words are kept; the rest become ``<unk>``.
    min_freq:
        Minimum training occurrences for a word to earn an id. Raising this
        drops one-off typos and proper nouns, which a model cannot learn
        anything reliable from anyway.

    See Also
    --------
    make_text_loaders : Consumes this.
    """

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
        """Return the loader settings as JSON-safe values.

        Records how a text run was configured, so it can be arranged the same
        way later.

        Returns
        -------
        dict
            Every field.
        """
        return asdict(self)


@dataclass(slots=True)
class TextContract:
    """What a text model needs in order to be fed correctly.

    Attributes
    ----------
    text_column:
        The source column.
    target_column:
        What is being predicted.
    task:
        Always ``'classification'``. Text regression is not on this path.
    class_labels:
        The class vocabulary, indexed by predicted class id.
    vocab:
        The serialised :class:`TextVocab`.
    modality:
        ``'text_tokens'``, distinguishing this from tabular and multimodal
        bundles.

    Notes
    -----
    **The vocabulary is the part that must survive.** Column names can be
    rediscovered; the specific word-to-id mapping cannot, and a text model
    without it is unusable.

    See Also
    --------
    make_text_loaders : Produces this.
    """

    text_column: str
    target_column: str
    task: Literal["classification"] = "classification"
    class_labels: tuple[Any, ...] = ()
    vocab: dict[str, Any] = field(default_factory=dict)
    modality: str = "text_tokens"

    def to_feature_contract(self) -> FeatureContract:
        """Project this down to the tabular-shaped contract shared code expects.

        Loader reports and evaluation take a plain
        :class:`~buildml.dl.types.FeatureContract`. This produces one listing
        the text column as the single feature, with no normalisation statistics
        since text is not scaled.

        Returns
        -------
        FeatureContract
            The flattened view.

        Notes
        -----
        **The vocabulary does not survive the projection.** Keep the
        ``TextContract`` for anything that needs to rebuild loaders.
        """
        return FeatureContract(
            feature_columns=(self.text_column,),
            target_column=self.target_column,
            task="classification",
            class_labels=self.class_labels,
            normalize_mean=None,
            normalize_std=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the text contract as JSON-safe values.

        Includes the full vocabulary, since a persisted contract without it
        cannot rebuild working loaders.

        Returns
        -------
        dict
            Text and target columns, task, class labels, serialised
            vocabulary, and modality.
        """
        return {
            "text_column": self.text_column,
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": list(self.class_labels),
            "vocab": dict(self.vocab),
            "modality": self.modality,
        }


def tokenize(text: str) -> list[str]:
    """Split a document into lowercase word-like tokens.

    Lowercases, then extracts runs of letters, digits, and underscores.
    Punctuation, whitespace, and anything else act as separators and are
    discarded.

    Parameters
    ----------
    text:
        The document. Coerced to ``str``, so numbers and ``NaN`` do not crash.

    Returns
    -------
    list of str
        The tokens, in order.

    Notes
    -----
    **Deliberately simple, with the trade-offs that implies.** Contractions
    split ("don't" becomes "don" and "t"), hyphenated words split, and
    non-Latin scripts without word-boundary punctuation may not segment
    usefully. For a corpus where any of that carries meaning, the richer
    tokenisation in :mod:`buildml.nlp` or a pretrained subword tokeniser is a
    better fit.

    **Lowercasing loses case information.** "Apple" and "apple" become the same
    token, which is usually right and occasionally not.

    Examples
    --------
    >>> tokenize("Hello, World! It's 2024.")
    ['hello', 'world', 'it', 's', '2024']
    """
    return _TOKEN_RE.findall(str(text).lower())


def fit_vocab(
    texts: list[str],
    *,
    max_vocab: int = 5000,
    min_freq: int = 1,
    max_len: int = 64,
) -> TextVocab:
    """Build the word-to-id mapping from training documents.

    Counts every token, keeps those meeting the frequency floor, orders them by
    descending frequency, and truncates to the size cap. Ids 0 and 1 are
    reserved for padding and unknown words before the real words begin.

    Parameters
    ----------
    texts:
        The training documents. **Training only** — including holdout text here
        leaks.
    max_vocab:
        Cap including the two reserved ids, so ``5000`` allows 4998 words.
    min_freq:
        Minimum occurrences to earn an id.
    max_len:
        Sequence length, stored on the vocabulary for later encoding.

    Returns
    -------
    TextVocab
        The mapping, both directions, with the reserved ids and length.

    Notes
    -----
    **Frequency ordering makes truncation sensible.** Cutting the tail removes
    the rarest words, which are the ones the model has least evidence about
    anyway. Ties break alphabetically so the mapping is reproducible.

    **Words that do not make the cut become ``<unk>`` during training, and that
    is useful.** It teaches the model that unknown words exist, so the
    inevitable out-of-vocabulary words at deployment are a case it has seen
    rather than a surprise.

    See Also
    --------
    texts_to_ids : Applying the result.
    """
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
    """Convert documents into a rectangular matrix of token ids.

    Tokenises each document, maps tokens to ids, truncates at the vocabulary's
    length, and pads the remainder. The result is rectangular, which is what
    batching requires.

    Parameters
    ----------
    texts:
        The documents to encode.
    vocab:
        The training vocabulary.

    Returns
    -------
    numpy.ndarray
        An ``(n_documents, max_len)`` int64 matrix.

    Notes
    -----
    **Padding sits at the end, and the model must mask it.** Left padding would
    also work, but the models here assume trailing padding and exclude
    ``pad_id`` positions from pooling.

    **Unknown words map to ``unk_id`` rather than being dropped.** Dropping them
    would shorten the sequence and shift everything after, changing the
    positions the model sees.

    See Also
    --------
    fit_vocab : Building the vocabulary.
    """
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
    frame = dataset._ensure_pandas()
    object_like = [
        c
        for c in feature_cols
        if pd.api.types.is_object_dtype(frame[c]) or pd.api.types.is_string_dtype(frame[c])
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
    """Build loaders that feed token sequences to a text classifier.

    Finds the text column, fits a vocabulary on the training partition, encodes
    every partition with it, and wraps the results as DataLoaders.

    Parameters
    ----------
    dataset:
        The data, with roles and a numeric target.
    split_plan:
        Which rows belong to which partition. A train partition is required.
    text_column:
        The text column. Inferred when exactly one string-like feature exists.
    config:
        Batching, sequence length, and vocabulary settings.

    Returns
    -------
    TorchLoaderBundle
        The loaders, the feature contract, the text contract, the vocabulary,
        and a report.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If no text column can be identified or the named one is absent, if
        several candidates exist and none was named, if the train partition is
        empty, if the target is not numeric class ids, or if ``batch_size`` is
        below 1.

    Notes
    -----
    **The vocabulary comes from training documents only**, so words appearing
    exclusively in validation or test arrive as ``<unk>`` — which is exactly
    what will happen in production, and therefore what the holdout should
    measure.

    **Labels must already be integers.** Encode string class labels before
    calling; the text path does not do it for you.

    **Classification only.** Text regression is not supported here.

    Examples
    --------
    Build loaders and a matching model::

        bundle = make_text_loaders(dataset, split_plan, text_column="review")
        module = build_text_classifier(
            vocab_size=bundle.text_vocab.vocab_size,
            n_classes=len(bundle.contract.class_labels),
        )

    See Also
    --------
    buildml.dl.models.build_text_classifier : The matching model.
    fit_vocab : What gets fitted here.
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
    class_labels = fit_class_labels(y_train)
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
        y = encode_class_targets(part[target], class_labels)
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
    """Translate text loader settings into the shared config shape.

    Some code paths — the training loop, checkpoint metadata — take a generic
    :class:`~buildml.dl.types.LoaderConfig`. This carries the fields the two
    have in common across.

    Parameters
    ----------
    cfg:
        The text loader settings.

    Returns
    -------
    LoaderConfig
        The shared shape, with ``normalize`` forced off.

    Notes
    -----
    **Text-specific settings do not survive.** Sequence length, vocabulary cap,
    and frequency floor have no equivalent field, so keep the
    ``TextLoaderConfig`` if you need to rebuild loaders. ``normalize`` is always
    false — standardising token ids would be meaningless, since they are labels
    rather than quantities.
    """
    return LoaderConfig(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle_train=cfg.shuffle_train,
        drop_last=cfg.drop_last,
        normalize=False,
        seed=cfg.seed,
    )
