"""Torch replay / EWC continual tabular adapters (buildml[torch])."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError
from buildml.online.extras import require_torch_continual
from buildml.online.types import OnlineTask, TorchContinualMethod

_TORCH_CLASSIFIERS = {"replay_mlp", "ewc_mlp"}


def resolve_torch_task(estimator: str, task: OnlineTask | None) -> OnlineTask:
    if estimator in _TORCH_CLASSIFIERS:
        if task == "regression":
            raise ValidationError(
                f"Estimator {estimator!r} is a classifier; task cannot be 'regression'."
            )
        return "classification"
    raise ValidationError(
        f"Unknown torch continual estimator={estimator!r}. "
        f"Supported: {sorted(_TORCH_CLASSIFIERS)}"
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
        torch = require_torch_continual()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)
        if classes is not None and len(self.classes_) == 0:
            self.classes_ = np.asarray(classes, dtype=int)
        if len(self.classes_) == 0:
            self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValidationError(
                "Torch continual classifier needs at least 2 classes on init."
            )
        code_map = {int(c): i for i, c in enumerate(self.classes_)}
        y_codes = np.vectorize(code_map.get)(y_arr).astype(int)

        self.n_features_ = int(x_arr.shape[1])
        device = torch.device(self.device)
        if self.module_ is None:
            self.module_ = _build_mlp(
                self.n_features_,
                len(self.classes_),
                hidden_dim=int(self.hidden_dim),
            ).to(device)

        for i in range(len(x_arr)):
            self.buffer_x_.append(x_arr[i].copy())
            self.buffer_y_.append(int(y_codes[i]))
            if len(self.buffer_x_) > int(self.buffer_size):
                self.buffer_x_.popleft()
                self.buffer_y_.popleft()

        if self.method == "ewc_mlp" and self.n_updates_ == 0:
            self._capture_ewc_anchor(torch, device)

        self._train_on_buffer(torch, device)
        self.n_updates_ += 1
        return self

    def _capture_ewc_anchor(self, torch: Any, device: Any) -> None:
        if self.module_ is None or len(self.buffer_x_) < 2:
            return
        x_stack = np.stack(list(self.buffer_x_), axis=0)
        y_stack = np.asarray(list(self.buffer_y_), dtype=int)
        tensor_x = torch.as_tensor(x_stack, device=device)
        tensor_y = torch.as_tensor(y_stack, device=device, dtype=torch.long)
        self.module_.eval()
        self.anchor_params_ = [
            p.detach().clone() for p in self.module_.parameters() if p.requires_grad
        ]
        fisher = [
            torch.zeros_like(p) for p in self.module_.parameters() if p.requires_grad
        ]
        logits = self.module_(tensor_x)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(1, tensor_y.unsqueeze(1)).squeeze(1)
        loss = -selected.mean()
        self.module_.zero_grad()
        loss.backward()
        idx = 0
        for p in self.module_.parameters():
            if not p.requires_grad:
                continue
            fisher[idx] = p.grad.detach().pow(2).clone()
            idx += 1
        self.fisher_diag_ = fisher

    def _ewc_penalty(self, torch: Any) -> Any:
        if (
            self.method != "ewc_mlp"
            or self.fisher_diag_ is None
            or not self.anchor_params_
        ):
            return torch.zeros((), device=self.module_.parameters().__next__().device)
        penalty = torch.zeros((), device=self.module_.parameters().__next__().device)
        idx = 0
        for p in self.module_.parameters():
            if not p.requires_grad:
                continue
            penalty = penalty + (
                self.fisher_diag_[idx] * (p - self.anchor_params_[idx]).pow(2)
            ).sum()
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

    class _MLP(torch.nn.Module):
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
