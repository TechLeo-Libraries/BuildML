"""Torch FixMatch/MixMatch-style tabular semi-supervised adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.semisupervised.extras import require_torch_semisupervised
from buildml.semisupervised.types import SKLEARN_UNLABELED, TorchSemiSupervisedMethod


@dataclass
class TabularConsistencyClassifier:
    """FixMatch/MixMatch-inspired tabular classifier with pseudo-labels."""

    method: TorchSemiSupervisedMethod = "fixmatch_tabular"
    threshold: float = 0.75
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    mixup_alpha: float = 0.75
    consistency_weight: float = 1.0
    random_state: int | None = 0
    device: str = "cpu"
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))
    module_: Any = field(default=None, repr=False)
    n_features_: int = 0
    n_pseudo_labels_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> TabularConsistencyClassifier:
        """Validate before refitting and isolate seeded Torch initialization/training."""
        if self.method not in {"fixmatch_tabular", "mixmatch_tabular"}:
            raise ValidationError(f"Unsupported torch method '{self.method}'")
        for name, minimum in (("epochs", 1), ("batch_size", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValidationError(f"{name} must be an integer >= {minimum}")
        for name in ("learning_rate", "weight_decay", "mixup_alpha", "consistency_weight", "threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValidationError(f"{name} must be finite")
            numeric_value = float(value)
            minimum_ok = numeric_value > 0 if name in {"learning_rate", "mixup_alpha"} else numeric_value >= 0
            if not minimum_ok or (name == "threshold" and numeric_value > 1):
                raise ValidationError(f"Invalid {name}: {value}")
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state < 2**64
        ):
            raise ValidationError("random_state must be a nonnegative integer below 2**64 or None")
        try:
            x_arr = np.asarray(x, dtype=np.float32)
            raw_y = np.asarray(y)
            numeric_y = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("Torch consistency requires numeric features and integer labels") from exc
        if x_arr.ndim != 2 or x_arr.shape[1] == 0 or not np.isfinite(x_arr).all():
            raise ValidationError("Features must be a finite two-dimensional matrix with nonzero width")
        if (raw_y.ndim != 1 or len(raw_y) != len(x_arr)
                or not np.isfinite(numeric_y).all()
                or not np.equal(numeric_y, np.floor(numeric_y)).all()
                or (numeric_y < -1).any() or (numeric_y >= 2**63).any()):
            raise ValidationError("Labels must be aligned integer codes >= -1")
        torch = require_torch_semisupervised()
        device = torch.device(self.device)
        if self.random_state is None:
            return self._fit_impl(x_arr, numeric_y.astype(np.int64))
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(self.random_state))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(self.random_state))
            return self._fit_impl(x_arr, numeric_y.astype(np.int64))

    def _fit_impl(self, x: np.ndarray, y: np.ndarray) -> TabularConsistencyClassifier:
        """Run fit on input data using the fitted internal state.

Parameters
----------
x:
    Feature matrix input rows.
y:
    Target vector or series aligned with ``x``.

Returns
-------
TabularConsistencyClassifier
    Return value (TabularConsistencyClassifier) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        torch = require_torch_semisupervised()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)
        labeled = y_arr != SKLEARN_UNLABELED
        if labeled.sum() < 2:
            raise ValidationError(
                "Torch consistency semi-supervised needs at least 2 labeled train rows."
            )
        classes = np.unique(y_arr[labeled])
        if len(classes) < 2:
            raise ValidationError(
                "Torch consistency semi-supervised needs ≥2 classes among labeled rows."
            )
        n_features = int(x_arr.shape[1])
        n_classes = int(len(classes))
        code_map = {int(c): i for i, c in enumerate(classes)}
        y_codes = np.full(shape=len(y_arr), fill_value=-1, dtype=int)
        y_codes[labeled] = np.vectorize(code_map.get)(y_arr[labeled])

        device = torch.device(self.device)
        module = _build_mlp(n_features, n_classes).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        criterion = torch.nn.CrossEntropyLoss()
        rng = np.random.default_rng(self.random_state)
        tensor_x = torch.as_tensor(x_arr, device=device)
        labeled_idx = np.flatnonzero(labeled)
        unlabeled_idx = np.flatnonzero(~labeled)
        n_pseudo = 0

        for _epoch in range(int(self.epochs)):
            perm = rng.permutation(len(x_arr))
            for start in range(0, len(perm), int(self.batch_size)):
                batch_idx = perm[start : start + int(self.batch_size)]
                if len(batch_idx) < 2:
                    continue
                x_batch = tensor_x[batch_idx]
                y_batch = y_codes[batch_idx]
                batch_labeled = y_batch >= 0

                optimizer.zero_grad()
                logits = module(x_batch)
                loss = torch.zeros((), device=device)

                if batch_labeled.any():
                    loss = loss + criterion(logits[batch_labeled], torch.as_tensor(
                        y_batch[batch_labeled], device=device, dtype=torch.long
                    ))

                if (~batch_labeled).any() and len(unlabeled_idx) > 0:
                    unlab_in_batch = batch_idx[~batch_labeled]
                    x_u = tensor_x[unlab_in_batch]
                    if self.method == "fixmatch_tabular":
                        weak = _weak_augment(x_u, rng)
                        strong = _strong_augment(x_u, module, rng)
                        with torch.no_grad():
                            weak_probs = torch.softmax(module(weak), dim=-1)
                            conf, pseudo = weak_probs.max(dim=-1)
                            mask = conf >= float(self.threshold)
                        if mask.any():
                            pseudo_loss = criterion(
                                module(strong)[mask],
                                pseudo[mask],
                            )
                            loss = loss + float(self.consistency_weight) * pseudo_loss
                            n_pseudo += int(mask.sum().item())
                    elif self.method == "mixmatch_tabular":
                        u_logits = module(x_u)
                        u_probs = torch.softmax(u_logits, dim=-1)
                        sharpened = u_probs ** (1.0 / 0.5)
                        sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        if batch_labeled.any():
                            x_l = x_batch[batch_labeled]
                            y_l = torch.as_tensor(
                                y_batch[batch_labeled], device=device, dtype=torch.long
                            )
                            lam = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
                            mix_idx = rng.permutation(int(x_l.shape[0]))
                            x_mix = lam * x_l + (1.0 - lam) * x_l[mix_idx]
                            y_a = torch.nn.functional.one_hot(y_l, n_classes).float()
                            y_b = y_a[mix_idx]
                            y_mix = lam * y_a + (1.0 - lam) * y_b
                            mix_logits = module(x_mix)
                            mix_loss = -(y_mix * torch.log_softmax(mix_logits, dim=-1)).sum(dim=-1).mean()
                            loss = loss + mix_loss
                        if u_probs.shape[0] >= 2:
                            u_mix_idx = rng.permutation(int(u_probs.shape[0]))
                            lam_u = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
                            x_u_mix = lam_u * x_u + (1.0 - lam_u) * x_u[u_mix_idx]
                            target_u = lam_u * sharpened + (1.0 - lam_u) * sharpened[u_mix_idx]
                            u_mix_logits = module(x_u_mix)
                            u_loss = -(target_u * torch.log_softmax(u_mix_logits, dim=-1)).sum(dim=-1).mean()
                            loss = loss + float(self.consistency_weight) * u_loss
                    else:
                        raise ValidationError(f"Unsupported torch method '{self.method}'")

                if loss.item() > 0:
                    loss.backward()
                    optimizer.step()

        self.classes_ = classes
        self.n_features_ = n_features
        self.module_ = module
        self.n_pseudo_labels_ = n_pseudo
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run predict on input data using the fitted internal state.

Parameters
----------
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        torch = require_torch_semisupervised()
        if self.module_ is None:
            raise ValidationError("TabularConsistencyClassifier is not fitted.")
        self.module_.eval()
        device = next(self.module_.parameters()).device
        x_arr = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            logits = self.module_(torch.as_tensor(x_arr, device=device))
            codes = logits.argmax(dim=-1).cpu().numpy()
        return np.asarray([int(self.classes_[c]) for c in codes], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Perform predict proba for the Session-facing workflow step.

Parameters
----------
x:
    Feature matrix input rows.

Returns
-------
np.ndarray
    NumPy array aligned with input rows.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        torch = require_torch_semisupervised()
        if self.module_ is None:
            raise ValidationError("TabularConsistencyClassifier is not fitted.")
        self.module_.eval()
        device = next(self.module_.parameters()).device
        x_arr = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            probs = torch.softmax(
                self.module_(torch.as_tensor(x_arr, device=device)), dim=-1
            ).cpu().numpy()
        return np.asarray(probs, dtype=float)


def _build_mlp(n_features: int, n_classes: int) -> Any:
    torch = require_torch_semisupervised()
    hidden = max(16, min(128, n_features * 4))
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, n_classes),
    )


def _weak_augment(x: Any, rng: np.random.Generator) -> Any:
    torch = require_torch_semisupervised()
    noise = torch.as_tensor(
        rng.normal(0.0, 0.05, size=tuple(x.shape)), device=x.device, dtype=x.dtype
    )
    return x + noise


def _strong_augment(x: Any, module: Any, rng: np.random.Generator) -> Any:
    torch = require_torch_semisupervised()
    noise = torch.as_tensor(
        rng.normal(0.0, 0.15, size=tuple(x.shape)), device=x.device, dtype=x.dtype
    )
    dropout = torch.nn.Dropout(p=0.2)
    return dropout(x + noise)


def build_torch_estimator(
    *,
    method: TorchSemiSupervisedMethod,
    threshold: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    consistency_weight: float,
    mixup_alpha: float,
    random_state: int | None,
    device: str = "cpu",
) -> TabularConsistencyClassifier:
    """Construct a torch estimator ready for fit or scoring.

Parameters
----------
method:
    Method or strategy identifier for the resolved backend.
threshold:
    Decision threshold applied to anomaly scores.
epochs:
    Training epochs for torch-backed estimators.
batch_size:
    Number of rows to select per query or training minibatch.
learning_rate:
    Optimizer learning rate for torch training.
consistency_weight:
    consistency weight (float).
mixup_alpha:
    mixup alpha (float).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
device:
    Torch device string (``cpu`` or ``cuda``).

Returns
-------
TabularConsistencyClassifier
    Return value (TabularConsistencyClassifier) produced by this operation.
    """
    return TabularConsistencyClassifier(
        method=method,
        threshold=threshold,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        consistency_weight=consistency_weight,
        mixup_alpha=mixup_alpha,
        random_state=random_state,
        device=device,
    )
