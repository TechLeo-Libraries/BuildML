"""Torch shared-trunk multi-head multi-task adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.multitask.extras import require_torch_multitask


@dataclass
class SharedTrunkMultiHeadEstimator:
    """Shared MLP trunk with per-task heads and joint training."""

    task_kinds: dict[str, str]
    target_columns: tuple[str, ...]
    classes_per_task_: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 64
    random_state: int | None = 0
    device: str = "cpu"
    module_: Any = field(default=None, repr=False)
    n_features_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> SharedTrunkMultiHeadEstimator:
        """Jointly train shared trunk and per-task heads on tabular features.

        Runs mini-batch AdamW optimization for ``epochs`` with classification
        cross-entropy and regression MSE losses summed across heads.

        Parameters
        ----------
        x:
            Float feature matrix of shape ``(n_samples, n_features)``.
        y:
            Encoded target matrix of shape ``(n_samples, n_tasks)``.

        Returns
        -------
        SharedTrunkMultiHeadEstimator
            Fitted module stored in ``module_`` (``self``).

        Raises
        ------
        ValidationError
            When target width mismatches ``target_columns`` or class counts are invalid.
        MissingExtraError
            When torch is not installed.
        """
        torch = require_torch_multitask()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.shape[1] != len(self.target_columns):
            raise ValidationError(
                f"Target width {y_arr.shape[1]} does not match "
                f"n_tasks={len(self.target_columns)}."
            )

        self.n_features_ = int(x_arr.shape[1])
        head_specs: list[tuple[str, int | None]] = []
        for i, col in enumerate(self.target_columns):
            kind = self.task_kinds[col]
            if kind == "classification":
                n_cls = len(self.classes_per_task_.get(col, ()))
                if n_cls < 2:
                    raise ValidationError(
                        f"Classification target {col!r} needs >= 2 classes."
                    )
                head_specs.append(("classification", n_cls))
            else:
                head_specs.append(("regression", None))

        device = torch.device(self.device)
        module = _MultiHeadModule(
            self.n_features_,
            head_specs,
            hidden_dim=int(self.hidden_dim),
        ).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        cls_loss = torch.nn.CrossEntropyLoss()
        reg_loss = torch.nn.MSELoss()

        tensor_x = torch.as_tensor(x_arr, device=device)
        rng = np.random.default_rng(self.random_state)
        n_rows = int(len(x_arr))

        for _epoch in range(int(self.epochs)):
            perm = rng.permutation(n_rows)
            for start in range(0, n_rows, int(self.batch_size)):
                batch_idx = perm[start : start + int(self.batch_size)]
                if len(batch_idx) < 1:
                    continue
                x_batch = tensor_x[batch_idx]
                y_batch = y_arr[batch_idx]
                optimizer.zero_grad()
                outputs = module(x_batch)
                loss = torch.zeros((), device=device)
                for head_idx, col in enumerate(self.target_columns):
                    kind = self.task_kinds[col]
                    if kind == "classification":
                        target = torch.as_tensor(
                            y_batch[:, head_idx], device=device, dtype=torch.long
                        )
                        loss = loss + cls_loss(outputs[head_idx], target)
                    else:
                        target = torch.as_tensor(
                            y_batch[:, head_idx], device=device, dtype=torch.float32
                        )
                        pred = outputs[head_idx].squeeze(-1)
                        loss = loss + reg_loss(pred, target)
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

        self.module_ = module
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict encoded targets for all tasks using the fitted module.

        Classification heads argmax to integer codes; regression heads return
        float values in a single stacked matrix.

        Parameters
        ----------
        x:
            Float feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(n_samples, n_tasks)``.

        Raises
        ------
        ValidationError
            When the module has not been fitted.
        MissingExtraError
            When torch is not installed.
        """
        if self.module_ is None:
            raise ValidationError("SharedTrunkMultiHeadEstimator is not fitted.")
        torch = require_torch_multitask()
        device = torch.device(self.device)
        x_arr = np.asarray(x, dtype=np.float32)
        tensor_x = torch.as_tensor(x_arr, device=device)
        self.module_.eval()
        with torch.no_grad():
            outputs = self.module_(tensor_x)
        cols: list[np.ndarray] = []
        for head_idx, col in enumerate(self.target_columns):
            kind = self.task_kinds[col]
            out = outputs[head_idx]
            if kind == "classification":
                cols.append(out.argmax(dim=-1).cpu().numpy().astype(int))
            else:
                cols.append(out.squeeze(-1).cpu().numpy().astype(float))
        return np.column_stack(cols)


def build_torch_estimator(
    *,
    target_columns: Sequence[str],
    task_kinds: dict[str, str],
    classes_per_task: dict[str, tuple[Any, ...]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int | None,
    device: str,
) -> SharedTrunkMultiHeadEstimator:
    """Construct a shared-trunk multi-head torch estimator for mixed targets.

    Packages per-target kinds, class lists, and training hyperparameters into
    :class:`SharedTrunkMultiHeadEstimator` for :func:`fit_multitask`.

    Parameters
    ----------
    target_columns:
        Target names defining head order.
    task_kinds:
        Per-target ``classification`` or ``regression`` mapping.
    classes_per_task:
        Class tuples for each classification target (for head output sizes).
    epochs:
        Number of full passes over the train set.
    batch_size:
        Minibatch size for joint head training.
    learning_rate:
        AdamW learning rate.
    random_state:
        Seed for batch shuffling.
    device:
        Torch device string (e.g. ``cpu``, ``cuda``).

    Returns
    -------
    SharedTrunkMultiHeadEstimator
        Unfitted estimator ready for :meth:`SharedTrunkMultiHeadEstimator.fit`.
    """
    return SharedTrunkMultiHeadEstimator(
        task_kinds=dict(task_kinds),
        target_columns=tuple(target_columns),
        classes_per_task_=dict(classes_per_task),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        random_state=random_state,
        device=device,
    )


def _MultiHeadModule(
    n_features: int,
    head_specs: list[tuple[str, int | None]],
    *,
    hidden_dim: int,
) -> Any:
    torch = require_torch_multitask()

    class Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            self.heads = torch.nn.ModuleList()
            for kind, n_cls in head_specs:
                if kind == "classification":
                    self.heads.append(torch.nn.Linear(hidden_dim, int(n_cls or 2)))
                else:
                    self.heads.append(torch.nn.Linear(hidden_dim, 1))

        def forward(self, x: Any) -> list[Any]:
            shared = self.trunk(x)
            return [head(shared) for head in self.heads]

    return Module()
