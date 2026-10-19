"""Torch replay / EWC continual tabular adapters (buildml[torch])."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.online.extras import require_torch_continual
from buildml.online.types import OnlineTask, TorchContinualMethod

_TORCH_CLASSIFIERS = {"replay_mlp", "ewc_mlp"}


def _integer_codes(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or (raw.dtype.kind == "f" and not np.isfinite(raw).all()):
        raise ValidationError(
            "Continual class codes must be a finite one-dimensional integer array."
        )
    try:
        codes = raw.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Continual labels must be integer class codes.") from exc
    if not np.array_equal(raw, codes):
        raise ValidationError("Continual labels must be integer class codes without truncation.")
    return codes


def resolve_torch_task(estimator: str, task: OnlineTask | None) -> OnlineTask:
    """Resolve the task type for a torch continual tabular estimator.

    Torch continual adapters currently support classification only; regression
    requests are rejected with a clear validation error.

    Parameters
    ----------
    estimator:
        Torch continual method key (``replay_mlp`` or ``ewc_mlp``).
    task:
        Optional explicit task override.

    Returns
    -------
    OnlineTask
        Resolved task (currently always ``classification``).

    Raises
    ------
    ValidationError
        When the estimator is unknown or ``task='regression'`` is requested.
    """
    if estimator in _TORCH_CLASSIFIERS:
        if task == "regression":
            raise ValidationError(
                f"Estimator {estimator!r} is a classifier; task cannot be 'regression'."
            )
        return "classification"
    raise ValidationError(
        f"Unknown torch continual estimator={estimator!r}. Supported: {sorted(_TORCH_CLASSIFIERS)}"
    )


def build_torch_continual_estimator(
    name: TorchContinualMethod,
    *,
    random_state: int | None = 0,
    buffer_size: int = 512,
    epochs_per_update: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    ewc_lambda: float = 100.0,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> ContinualTabularClassifier:
    """Build a lite replay-buffer or EWC tabular MLP continual learner.

    Requires ``buildml[torch]``; exposes a sklearn-like ``partial_fit`` API.

    Parameters
    ----------
    name:
        Continual method (``replay_mlp`` or ``ewc_mlp``).
    random_state:
        Seed for buffer minibatch shuffling.
    buffer_size:
        Maximum replay buffer capacity.
    epochs_per_update:
        Training epochs per ``partial_fit`` call.
    batch_size:
        Minibatch size during buffer training.
    learning_rate:
        AdamW learning rate.
    ewc_lambda:
        EWC penalty weight for ``ewc_mlp``.
    hidden_dim:
        MLP hidden layer width.
    device:
        Torch device string (``cpu`` or ``cuda``).

    Returns
    -------
    ContinualTabularClassifier
        Classifier wrapper with ``partial_fit`` and ``predict``.
    """
    return ContinualTabularClassifier(
        method=name,
        buffer_size=int(buffer_size),
        epochs_per_update=int(epochs_per_update),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        ewc_lambda=float(ewc_lambda),
        hidden_dim=int(hidden_dim),
        random_state=random_state,
        device=device,
    )


@dataclass
class ContinualTabularClassifier:
    """Lite replay-buffer or EWC tabular MLP with partial_fit interface."""

    method: TorchContinualMethod = "replay_mlp"
    buffer_size: int = 512
    epochs_per_update: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    ewc_lambda: float = 100.0
    hidden_dim: int = 64
    random_state: int | None = 0
    device: str = "cpu"
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))
    module_: Any = field(default=None, repr=False)
    buffer_x_: deque = field(default_factory=deque, repr=False)
    buffer_y_: deque = field(default_factory=deque, repr=False)
    fisher_diag_: Any = field(default=None, repr=False)
    anchor_params_: list[Any] = field(default_factory=list, repr=False)
    n_features_: int = 0
    n_updates_: int = 0

    def partial_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: Sequence[Any] | None = None,
    ) -> ContinualTabularClassifier:
        """Append a chunk to the replay buffer and train the MLP.

        For ``ewc_mlp``, captures empirical per-example Fisher information
        after the first usable update for subsequent elastic consolidation.

        Parameters
        ----------
        x:
            2-D float feature matrix.
        y:
            Integer class codes aligned with ``x`` rows.
        classes:
            Optional full class code vocabulary on first fit.

        Returns
        -------
        ContinualTabularClassifier
            ``self`` for chaining.

        Raises
        ------
        ValidationError
            When fewer than two classes are present on init.
        MissingExtraError
            When torch is not installed.
        """
        torch = require_torch_continual()
        if self.method not in _TORCH_CLASSIFIERS:
            raise ValidationError(f"Unknown continual method={self.method!r}.")
        if self.random_state is not None and (
            not isinstance(self.random_state, (int, np.integer))
            or not 0 <= self.random_state < 2**64
        ):
            raise ValidationError("random_state must be a nonnegative 64-bit integer or None.")
        if (
            self.buffer_size < 2
            or self.epochs_per_update < 1
            or self.batch_size < 2
            or self.hidden_dim < 1
        ):
            raise ValidationError(
                "Continual training needs buffer_size/batch_size >=2 and positive epochs/hidden_dim."
            )
        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not np.isfinite(self.ewc_lambda)
            or self.ewc_lambda < 0
        ):
            raise ValidationError(
                "learning_rate must be positive and ewc_lambda nonnegative, both finite."
            )
        try:
            x_arr = np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Continual features must be a finite numeric matrix.") from exc
        y_arr = _integer_codes(y)
        if (
            x_arr.ndim != 2
            or not x_arr.shape[0]
            or not x_arr.shape[1]
            or not np.isfinite(x_arr).all()
        ):
            raise ValidationError(
                "Continual features must be a nonempty finite two-dimensional matrix."
            )
        if len(x_arr) != len(y_arr):
            raise ValidationError("Continual feature and label row counts must match.")
        if self.module_ is not None and x_arr.shape[1] != self.n_features_:
            raise ValidationError("Continual feature count changed after initialization.")
        vocabulary = (
            _integer_codes(classes)
            if classes is not None
            else (self.classes_ if len(self.classes_) else np.unique(y_arr))
        )
        if len(vocabulary) < 2 or len(np.unique(vocabulary)) != len(vocabulary):
            raise ValidationError(
                "Torch continual classifier needs at least 2 distinct classes on init."
            )
        if len(self.classes_) and not np.array_equal(vocabulary, self.classes_):
            raise ValidationError("Continual class vocabulary cannot change after initialization.")
        if not np.isin(y_arr, vocabulary).all():
            raise ValidationError(
                "Continual labels contain class codes outside the fitted vocabulary."
            )
        code_map = {int(c): i for i, c in enumerate(vocabulary)}
        y_codes = np.asarray([code_map[int(label)] for label in y_arr], dtype=int)

        device = torch.device(self.device)
        if self.module_ is None:
            if self.random_state is None:
                module = _build_mlp(
                    x_arr.shape[1], len(vocabulary), hidden_dim=int(self.hidden_dim)
                )
            else:
                # Initialization runs on CPU; preserve caller and CUDA RNG states.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(int(self.random_state))
                    module = _build_mlp(
                        x_arr.shape[1], len(vocabulary), hidden_dim=int(self.hidden_dim)
                    )
            self.module_ = module.to(device)
        self.classes_ = vocabulary.copy()
        self.n_features_ = int(x_arr.shape[1])

        for i in range(len(x_arr)):
            self.buffer_x_.append(x_arr[i].copy())
            self.buffer_y_.append(int(y_codes[i]))
            if len(self.buffer_x_) > int(self.buffer_size):
                self.buffer_x_.popleft()
                self.buffer_y_.popleft()

        self._train_on_buffer(torch, device)
        if self.method == "ewc_mlp" and self.fisher_diag_ is None:
            self._capture_ewc_anchor(torch, device)
        self.n_updates_ += 1
        return self

    def _capture_ewc_anchor(self, torch: Any, device: Any) -> None:
        if self.module_ is None or len(self.buffer_x_) < 2:
            return
        x_stack = np.stack(list(self.buffer_x_), axis=0)
        y_stack = np.asarray(list(self.buffer_y_), dtype=int)
        tensor_x = torch.as_tensor(x_stack, device=device)
        tensor_y = torch.as_tensor(y_stack, device=device, dtype=torch.long)
        params = [p for p in self.module_.parameters() if p.requires_grad]
        self.anchor_params_ = [p.detach().clone() for p in params]
        fisher = [torch.zeros_like(p) for p in params]
        was_training = self.module_.training
        self.module_.eval()
        try:
            for row, target in zip(tensor_x, tensor_y):
                loss = torch.nn.functional.cross_entropy(
                    self.module_(row.unsqueeze(0)), target.unsqueeze(0)
                )
                gradients = torch.autograd.grad(loss, params)
                for diagonal, gradient in zip(fisher, gradients):
                    diagonal.add_(gradient.detach().pow(2) / len(tensor_x))
        finally:
            self.module_.train(was_training)
        self.fisher_diag_ = fisher

    def _ewc_penalty(self, torch: Any) -> Any:
        if self.method != "ewc_mlp" or self.fisher_diag_ is None or not self.anchor_params_:
            return torch.zeros((), device=self.module_.parameters().__next__().device)
        penalty = torch.zeros((), device=self.module_.parameters().__next__().device)
        idx = 0
        for p in self.module_.parameters():
            if not p.requires_grad:
                continue
            penalty = (
                penalty + (self.fisher_diag_[idx] * (p - self.anchor_params_[idx]).pow(2)).sum()
            )
            idx += 1
        return float(self.ewc_lambda) * penalty

    def _train_on_buffer(self, torch: Any, device: Any) -> None:
        if self.module_ is None or len(self.buffer_x_) < 2:
            return
        x_stack = np.stack(list(self.buffer_x_), axis=0)
        y_stack = np.asarray(list(self.buffer_y_), dtype=int)
        optimizer = torch.optim.AdamW(
            self.module_.parameters(),
            lr=float(self.learning_rate),
        )
        criterion = torch.nn.CrossEntropyLoss()
        rng = np.random.default_rng(self.random_state)
        tensor_x = torch.as_tensor(x_stack, device=device)
        tensor_y = torch.as_tensor(y_stack, device=device, dtype=torch.long)
        self.module_.train()
        for _ in range(int(self.epochs_per_update)):
            perm = rng.permutation(len(x_stack))
            for start in range(0, len(perm), int(self.batch_size)):
                batch_idx = perm[start : start + int(self.batch_size)]
                if len(batch_idx) < 2:
                    continue
                optimizer.zero_grad()
                logits = self.module_(tensor_x[batch_idx])
                loss = criterion(logits, tensor_y[batch_idx]) + self._ewc_penalty(torch)
                loss.backward()
                optimizer.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class codes for a batch of tabular rows.

        Runs the fitted MLP in eval mode and maps internal logits back to the
        original ``classes_`` vocabulary.

        Parameters
        ----------
        x:
            2-D float feature matrix.

        Returns
        -------
        numpy.ndarray
            Original class codes (not internal 0..K-1 indices).

        Raises
        ------
        ValidationError
            When the classifier has not been fitted yet.
        MissingExtraError
            When torch is not installed.
        """
        torch = require_torch_continual()
        if self.module_ is None:
            raise ValidationError("ContinualTabularClassifier is not fitted.")
        device = torch.device(self.device)
        x_arr = np.asarray(x, dtype=np.float32)
        self.module_.eval()
        with torch.no_grad():
            logits = self.module_(torch.as_tensor(x_arr, device=device))
            codes = torch.argmax(logits, dim=-1).cpu().numpy()
        return self.classes_[codes]


def _build_mlp(n_features: int, n_classes: int, *, hidden_dim: int) -> Any:
    torch = require_torch_continual()

    # Dynamic torch import: base class is only known at runtime.
    class _MLP(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return _MLP()
