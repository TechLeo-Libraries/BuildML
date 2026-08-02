"""Small built-in Torch model zoo for the happy path (tabular + text).

Requires ``buildml[torch]``. Modules are plain ``nn.Module`` instances so they
compose with :func:`buildml.dl.train.train_supervised_module` and trainer bundles.
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
    """Construct a feed-forward MLP for numeric tabular tensors.

    Parameters
    ----------
    in_features:
        Number of input features (must match loader contract).
    task:
        ``classification`` (logits of size ``n_classes``) or ``regression``
        (single output).
    n_classes:
        Output classes for classification (ignored for regression).
    hidden:
        Hidden layer widths.
    dropout:
        Dropout probability after each hidden activation.
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
    """Factory helper mirroring :func:`build_tabular_mlp` for discoverability."""

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
    """Token embedding text classifier (masked mean pool → MLP).

    Input batch shape: ``(batch, seq_len)`` of integer token ids with pad=0.
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
