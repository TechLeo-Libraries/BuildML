"""Two ready-made networks, so the first Torch run does not start blank.

Writing an ``nn.Module`` from scratch is the usual first obstacle to trying deep
learning on a dataset, and it is a distraction when what you want to know is
whether a neural network helps at all. These builders provide a reasonable
starting architecture for the two commonest cases — numeric tabular features and
short token sequences — so you can get a baseline and then decide whether to
invest in something bespoke.

They return plain ``nn.Module`` instances. Nothing here wraps or subclasses in a
way that constrains what you do next: train them with
:func:`~buildml.dl.train.train_supervised_module`, save them in a trainer
bundle, modify them, or throw them away and write your own.

They are deliberately small. A two-hidden-layer MLP and a mean-pooled embedding
classifier are not competitive with a tuned architecture, and they are not meant
to be — they are the baseline you compare against.

See Also
--------
buildml.dl.zoo : Larger pretrained architectures.
buildml.dl.train : Training whatever you build.
"""

from __future__ import annotations

from typing import Any

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch


def build_tabular_mlp(
    in_features: int,
    *,
    task: str = "classification",
    n_classes: int = 2,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
) -> Any:
    """Build a plain feed-forward network for numeric tabular data.

    A stack of linear layers with ReLU activations and optional dropout, ending
    in a head sized for the task. This is the standard starting point for
    tabular deep learning — worth knowing that gradient-boosted trees usually
    beat it on tabular problems, so treat a neural network here as something to
    justify rather than assume.

    Parameters
    ----------
    in_features:
        Number of input columns. Must match the loader contract's feature count
        exactly, or the first batch fails on a shape mismatch.
    task:
        ``'classification'`` for ``n_classes`` logits, ``'regression'`` for a
        single output.
    n_classes:
        Output width for classification. Ignored for regression.
    hidden:
        Hidden layer widths, in order. The default ``(64, 32)`` narrows toward
        the output, which is conventional and works well enough to start with.
    dropout:
        Probability of zeroing each hidden unit during training. Regularisation:
        by preventing the network from relying on any single unit it is pushed
        toward redundant representations that generalise better. Set to 0 to
        disable.

    Returns
    -------
    torch.nn.Sequential
        The network, with ``task``, ``in_features``, and ``n_classes`` attached
        as attributes for downstream code to read.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If ``in_features`` is below 1, the task is unrecognised, classification
        is requested with fewer than two classes, or dropout is outside
        ``[0, 1)``.

    Notes
    -----
    **No softmax on the output.** The classification head emits raw logits
    because ``CrossEntropyLoss`` applies its own log-softmax internally, and
    applying it twice degrades the gradient. Call ``softmax`` yourself if you
    need probabilities at inference.

    **Dropout is active during training and disabled during evaluation.** Torch
    handles the switch through ``module.train()`` and ``module.eval()``, which
    the training and evaluation loops call for you.

    Examples
    --------
    Size the network from the loader contract::

        module = build_tabular_mlp(
            in_features=len(bundle.contract.feature_columns),
            task=bundle.contract.task,
            n_classes=len(bundle.contract.class_labels) or 2,
        )

    See Also
    --------
    TabularMLP : The same thing spelled as a class.
    buildml.dl.train.train_supervised_module : Training it.
    """
    torch = require_torch(feature="TabularMLP")
    if in_features < 1:
        raise ValidationError("in_features must be >= 1")
    if task not in {"classification", "regression"}:
        raise ValidationError("task must be 'classification' or 'regression'")
    if task == "classification" and n_classes < 2:
        raise ValidationError("n_classes must be >= 2 for classification")
    if dropout < 0 or dropout >= 1:
        raise ValidationError("dropout must be in [0, 1)")

    layers: list[Any] = []
    prev = int(in_features)
    for width in hidden:
        layers.append(torch.nn.Linear(prev, int(width)))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=float(dropout)))
        prev = int(width)
    out = int(n_classes) if task == "classification" else 1
    layers.append(torch.nn.Linear(prev, out))
    module = torch.nn.Sequential(*layers)
    module.task = task  # type: ignore[attr-defined]
    module.in_features = int(in_features)  # type: ignore[attr-defined]
    module.n_classes = int(n_classes) if task == "classification" else 1  # type: ignore[attr-defined]
    return module


class TabularMLP:
    """A class-shaped alias for :func:`build_tabular_mlp`.

    Exists purely so that ``TabularMLP(n_features)`` works, since a model that
    looks like a class is what most people reach for and search for.
    Constructing it calls :func:`build_tabular_mlp` and returns whatever that
    returns.

    Notes
    -----
    **This is not a real class.** ``__new__`` returns an ``nn.Sequential``, so
    there is no ``TabularMLP`` instance to subclass or ``isinstance``-check.
    Every argument, default, and error behaviour is
    :func:`build_tabular_mlp`'s.

    See Also
    --------
    build_tabular_mlp : The function, and the full documentation.
    """

    def __new__(
        cls,
        in_features: int,
        *,
        task: str = "classification",
        n_classes: int = 2,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ) -> Any:
        return build_tabular_mlp(
            in_features,
            task=task,
            n_classes=n_classes,
            hidden=hidden,
            dropout=dropout,
        )


def build_text_classifier(
    vocab_size: int,
    *,
    n_classes: int = 2,
    embed_dim: int = 32,
    hidden: int = 64,
    padding_idx: int = 0,
    dropout: float = 0.1,
) -> Any:
    """Build a simple text classifier over token id sequences.

    Each token becomes a learned vector, the vectors for a document are
    averaged, and the average is classified by a small MLP. Averaging discards
    word order entirely — "the dog bit the man" and "the man bit the dog" get
    identical representations — which is a real limitation and also why the
    model is small, fast, and surprisingly hard to beat on topic-style
    classification where vocabulary carries most of the signal.

    Parameters
    ----------
    vocab_size:
        Number of distinct token ids, including the padding token. Must cover
        every id the loaders can emit.
    n_classes:
        Number of output classes.
    embed_dim:
        Width of each token's learned vector. Larger captures more nuance and
        needs more data to fit.
    hidden:
        Width of the classifier's hidden layer.
    padding_idx:
        The token id used for padding. Its embedding stays at zero and is
        excluded from the average.
    dropout:
        Dropout probability, applied after pooling and inside the head.

    Returns
    -------
    torch.nn.Module
        Accepting ``(batch, seq_len)`` integer token ids and emitting
        ``(batch, n_classes)`` logits. Carries ``task``, ``vocab_size``,
        ``n_classes``, and ``padding_idx`` as attributes.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If ``vocab_size`` or ``n_classes`` is below 2, or if ``embed_dim`` or
        ``hidden`` is below 1.

    Notes
    -----
    **Padding is masked out of the average, and this matters more than it
    sounds.** Batching requires every sequence to be the same length, so short
    documents are padded. Averaging over the padded length would divide a short
    document's signal by the batch's longest length, systematically shrinking
    its representation. Masking divides by the real token count instead, so
    document length stops affecting the scale.

    **The token vocabulary must be built from training documents only**, and
    :mod:`buildml.dl.text` handles that. A vocabulary fitted across all
    partitions leaks holdout vocabulary into the model.

    **Word order is not represented.** If order carries the signal — negation,
    sequence, syntax — this architecture cannot capture it, and a transformer
    from :mod:`buildml.dl.zoo` is the alternative.

    Examples
    --------
    Size from the text loader's vocabulary::

        module = build_text_classifier(
            vocab_size=bundle.report.vocab_size,
            n_classes=len(bundle.contract.class_labels),
        )

    See Also
    --------
    buildml.dl.text : Building token loaders and the vocabulary.
    buildml.dl.zoo : Pretrained transformers, when order matters.
    """
    torch = require_torch(feature="TextClassifier")
    if vocab_size < 2:
        raise ValidationError("vocab_size must be >= 2")
    if n_classes < 2:
        raise ValidationError("n_classes must be >= 2")
    if embed_dim < 1 or hidden < 1:
        raise ValidationError("embed_dim and hidden must be >= 1")

    class _TextClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(
                int(vocab_size),
                int(embed_dim),
                padding_idx=int(padding_idx),
            )
            self.dropout = torch.nn.Dropout(p=float(dropout))
            self.head = torch.nn.Sequential(
                torch.nn.Linear(int(embed_dim), int(hidden)),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=float(dropout)),
                torch.nn.Linear(int(hidden), int(n_classes)),
            )
            self.padding_idx = int(padding_idx)
            self.task = "classification"
            self.vocab_size = int(vocab_size)
            self.n_classes = int(n_classes)

        def forward(self, token_ids: Any) -> Any:
            mask = (token_ids != self.padding_idx).unsqueeze(-1).float()
            embedded = self.embedding(token_ids) * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = embedded.sum(dim=1) / denom
            return self.head(self.dropout(pooled))

    return _TextClassifier()
