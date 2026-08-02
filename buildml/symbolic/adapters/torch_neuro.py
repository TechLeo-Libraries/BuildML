"""Torch lite concept-bottleneck and neural-additive tabular models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.symbolic.extras import require_torch_symbolic


@dataclass
class TabularConceptBottleneck:
    """Lite concept-bottleneck: MLP → sigmoid concepts → linear head."""

    n_concepts: int = 8
    hidden_dim: int = 64
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    random_state: int | None = 0
    device: str = "cpu"
    task: str = "classification"
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))
    module_: Any = field(default=None, repr=False)
    concept_module_: Any = field(default=None, repr=False)
    n_features_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> TabularConceptBottleneck:
        torch = require_torch_symbolic()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y)
        self.n_features_ = int(x_arr.shape[1])
        device = torch.device(self.device)

        if self.task == "classification":
            self.classes_ = np.unique(y_arr)
            if len(self.classes_) < 2:
                raise ValidationError("Concept bottleneck needs ≥2 classes.")
            n_out = int(len(self.classes_))
            code_map = {c: i for i, c in enumerate(self.classes_)}
            y_codes = np.vectorize(code_map.get)(y_arr).astype(int)
            loss_fn: Any = torch.nn.CrossEntropyLoss()
        else:
            self.classes_ = np.array([])
            y_codes = y_arr.astype(np.float32)
            n_out = 1
            loss_fn = torch.nn.MSELoss()

        encoder = _build_encoder(self.n_features_, self.hidden_dim, self.n_concepts).to(device)
        head = torch.nn.Linear(self.n_concepts, n_out).to(device)
        params = list(encoder.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=float(self.learning_rate))
        rng = np.random.default_rng(self.random_state)
        tensor_x = torch.as_tensor(x_arr, device=device)
        tensor_y = torch.as_tensor(y_codes, device=device)

        for _ in range(int(self.epochs)):
            perm = rng.permutation(len(x_arr))
            for start in range(0, len(perm), int(self.batch_size)):
                idx = perm[start : start + int(self.batch_size)]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                concepts = encoder(tensor_x[idx])
                logits = head(concepts)
                if self.task == "classification":
                    loss = loss_fn(logits, tensor_y[idx].long())
                else:
                    loss = loss_fn(logits.squeeze(-1), tensor_y[idx].float())
                loss.backward()
                optimizer.step()

        self.concept_module_ = encoder
        self.module_ = torch.nn.Sequential(encoder, head)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = require_torch_symbolic()
        self.module_.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.module_[0].weight.device if hasattr(self.module_[0], "weight") else "cpu")
            out = self.module_(x_t)
            if self.task == "classification":
                preds = out.argmax(dim=1).cpu().numpy()
                return self.classes_[preds]
            return out.squeeze(-1).cpu().numpy()

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        if self.task == "classification":
            return float(np.mean(pred == y))
        err = pred.astype(float) - np.asarray(y, dtype=float)
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@dataclass
class TabularNeuralAdditive:
    """Lite NAM: one small subnetwork per feature, outputs summed."""

    hidden_dim: int = 16
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    random_state: int | None = 0
    device: str = "cpu"
    task: str = "classification"
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))
    module_: Any = field(default=None, repr=False)
    n_features_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> TabularNeuralAdditive:
        torch = require_torch_symbolic()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y)
        self.n_features_ = int(x_arr.shape[1])
        device = torch.device(self.device)

        if self.task == "classification":
            self.classes_ = np.unique(y_arr)
            n_out = int(len(self.classes_))
            code_map = {c: i for i, c in enumerate(self.classes_)}
            y_codes = np.vectorize(code_map.get)(y_arr).astype(int)
            loss_fn: Any = torch.nn.CrossEntropyLoss()
        else:
            self.classes_ = np.array([])
            y_codes = y_arr.astype(np.float32)
            n_out = 1
            loss_fn = torch.nn.MSELoss()

        module = _build_nam(self.n_features_, self.hidden_dim, n_out).to(device)
        optimizer = torch.optim.AdamW(module.parameters(), lr=float(self.learning_rate))
        rng = np.random.default_rng(self.random_state)
        tensor_x = torch.as_tensor(x_arr, device=device)
        tensor_y = torch.as_tensor(y_codes, device=device)

        for _ in range(int(self.epochs)):
            perm = rng.permutation(len(x_arr))
            for start in range(0, len(perm), int(self.batch_size)):
                idx = perm[start : start + int(self.batch_size)]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                logits = module(tensor_x[idx])
                if self.task == "classification":
                    loss = loss_fn(logits, tensor_y[idx].long())
                else:
                    loss = loss_fn(logits.squeeze(-1), tensor_y[idx].float())
                loss.backward()
                optimizer.step()

        self.module_ = module
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = require_torch_symbolic()
        self.module_.eval()
        device = next(self.module_.parameters()).device
        with torch.no_grad():
            out = self.module_(torch.as_tensor(np.asarray(x, dtype=np.float32), device=device))
            if self.task == "classification":
                preds = out.argmax(dim=1).cpu().numpy()
                return self.classes_[preds]
            return out.squeeze(-1).cpu().numpy()

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        if self.task == "classification":
            return float(np.mean(pred == y))
        err = pred.astype(float) - np.asarray(y, dtype=float)
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def build_torch_neuro_estimator(
    *,
    method: str,
    task: str,
    random_state: int | None,
    epochs: int = 60,
    device: str = "cpu",
) -> Any:
    key = str(method).lower().replace("-", "_")
    if key == "concept_bottleneck_lite":
        return TabularConceptBottleneck(
            task=task,
            random_state=random_state,
            epochs=epochs,
            device=device,
        )
    if key == "neural_additive_lite":
        return TabularNeuralAdditive(
            task=task,
            random_state=random_state,
            epochs=epochs,
            device=device,
        )
    raise ValidationError(f"Unknown torch neuro-symbolic method {method!r}.")


def _build_encoder(n_in: int, hidden: int, n_concepts: int) -> Any:
    torch = require_torch_symbolic()

    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, n_concepts),
        torch.nn.Sigmoid(),
    )


def _build_nam(n_features: int, hidden: int, n_out: int) -> Any:
    torch = require_torch_symbolic()

    class _NAM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nets = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(1, hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden, 1),
                    )
                    for _ in range(n_features)
                ]
            )
            self.bias = torch.nn.Parameter(torch.zeros(n_out))
            self.out = torch.nn.Linear(n_features, n_out, bias=False)

        def forward(self, x: Any) -> Any:
            parts = [net(x[:, i : i + 1]) for i, net in enumerate(self.nets)]
            stacked = torch.cat(parts, dim=1)
            return self.out(stacked) + self.bias

    return _NAM()
