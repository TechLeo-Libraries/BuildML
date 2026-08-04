"""Torch MC-dropout tabular classifier + BALD / MC-dropout query scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from buildml.activelearning.extras import require_torch_activelearning
from buildml.core.errors import ValidationError

TorchALStrategy = Literal["bald", "mc_dropout"]


@dataclass
class TabularMCDropoutClassifier:
    """Tabular MLP with MC dropout for deep active-learning query strategies."""

    dropout_rate: float = 0.25
    hidden_dim: int = 64
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    mc_samples: int = 20
    random_state: int | None = 0
    device: str = "cpu"
    classes_: np.ndarray = field(default_factory=lambda: np.array([]))
    module_: Any = field(default=None, repr=False)
    n_features_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> TabularMCDropoutClassifier:
        """Fit the MC-dropout MLP on labeled train rows.

        Trains a small tabular MLP with dropout enabled so later query scoring
        can estimate predictive uncertainty via stochastic forward passes.

        Parameters
        ----------
        x:
            Labeled training feature matrix.
        y:
            Integer class codes aligned with ``x``.

        Returns
        -------
        TabularMCDropoutClassifier
            Fitted classifier (``self``) with ``module_`` set.

        Raises
        ------
        ValidationError
            When fewer than two labeled rows or classes are present.
        MissingExtraError
            When torch is not installed.
        """
        torch = require_torch_activelearning()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)
        if len(y_arr) < 2:
            raise ValidationError(
                "Torch active-learning classifier needs at least 2 labeled train rows."
            )
        classes = np.unique(y_arr)
        if len(classes) < 2:
            raise ValidationError(
                "Torch active-learning classifier needs ≥2 classes among labeled rows."
            )
        self.classes_ = classes
        self.n_features_ = int(x_arr.shape[1])
        n_classes = int(len(classes))
        code_map = {int(c): i for i, c in enumerate(classes)}
        y_codes = np.vectorize(code_map.get)(y_arr).astype(int)

        device = torch.device(self.device)
        module = _build_mlp(
            self.n_features_,
            n_classes,
            hidden_dim=int(self.hidden_dim),
            dropout_rate=float(self.dropout_rate),
        ).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        criterion = torch.nn.CrossEntropyLoss()
        rng = np.random.default_rng(self.random_state)
        tensor_x = torch.as_tensor(x_arr, device=device)
        tensor_y = torch.as_tensor(y_codes, device=device, dtype=torch.long)

        for _epoch in range(int(self.epochs)):
            perm = rng.permutation(len(x_arr))
            for start in range(0, len(perm), int(self.batch_size)):
                batch_idx = perm[start : start + int(self.batch_size)]
                if len(batch_idx) < 2:
                    continue
                module.train()
                optimizer.zero_grad()
                logits = module(tensor_x[batch_idx])
                loss = criterion(logits, tensor_y[batch_idx])
                loss.backward()
                optimizer.step()

        self.module_ = module
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels via argmax of MC-dropout probability estimates.

        Uses a reduced MC sample count relative to full BALD scoring for faster
        point predictions during evaluation.

        Parameters
        ----------
        x:
            Feature matrix to classify.

        Returns
        -------
        np.ndarray
            Predicted class labels from ``classes_``.
        """
        proba = self.predict_proba(x)
        codes = np.argmax(proba, axis=1)
        return self.classes_[codes]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return MC-dropout averaged class probabilities for ``x``.

        Uses fewer MC samples than full BALD scoring for faster point predictions.

        Parameters
        ----------
        x:
            Feature matrix to score.

        Returns
        -------
        np.ndarray
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        return mc_dropout_proba(
            self,
            x,
            n_samples=max(5, int(self.mc_samples // 2)),
        )

    def mc_dropout_proba(self, x: np.ndarray, *, n_samples: int | None = None) -> np.ndarray:
        """Run MC-dropout forward passes and average resulting probabilities.

        Delegates to :func:`mc_dropout_proba` so both instance and module-level
        callers share the same sampling contract.

        Parameters
        ----------
        x:
            Feature matrix to score.
        n_samples:
            Number of stochastic forward passes; defaults to ``self.mc_samples``.

        Returns
        -------
        np.ndarray
            Mean probability matrix over MC samples.
        """
        return mc_dropout_proba(self, x, n_samples=n_samples)


def mc_dropout_proba(
    model: TabularMCDropoutClassifier,
    x: np.ndarray,
    *,
    n_samples: int | None = None,
) -> np.ndarray:
    """Average softmax probabilities over MC-dropout forward passes.

    Leaves the module in eval mode after sampling so subsequent predict calls
    are deterministic.

    Parameters
    ----------
    model:
        Fitted :class:`TabularMCDropoutClassifier` with ``module_`` set.
    x:
        Feature matrix to score.
    n_samples:
        Number of stochastic forward passes; defaults to ``model.mc_samples``.

    Returns
    -------
    np.ndarray
        Mean probability matrix of shape ``(n_samples, n_classes)``.

    Raises
    ------
    ValidationError
        When the classifier has not been fitted.
    MissingExtraError
        When torch is not installed.
    """
    torch = require_torch_activelearning()
    if model.module_ is None:
        raise ValidationError("TabularMCDropoutClassifier is not fitted.")
    n = int(n_samples if n_samples is not None else model.mc_samples)
    device = torch.device(model.device)
    x_arr = np.asarray(x, dtype=np.float32)
    samples = []
    model.module_.train()
    with torch.no_grad():
        for _ in range(n):
            logits = model.module_(torch.as_tensor(x_arr, device=device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            samples.append(probs)
    model.module_.eval()
    return np.mean(np.stack(samples, axis=0), axis=0)


def score_torch_pool(
    *,
    strategy: TorchALStrategy,
    x_pool: np.ndarray,
    estimator: TabularMCDropoutClassifier,
    mc_samples: int | None = None,
) -> np.ndarray:
    """Score pool rows with BALD or MC-dropout entropy query strategies.

    Runs multiple MC-dropout forward passes and ranks pool rows by predictive
    uncertainty (entropy) or BALD (mutual information).

    Parameters
    ----------
    strategy:
        Torch query strategy: ``bald`` or ``mc_dropout``.
    x_pool:
        Feature matrix for unlabeled pool rows.
    estimator:
        Fitted :class:`TabularMCDropoutClassifier`.
    mc_samples:
        Optional override for number of MC forward passes.

    Returns
    -------
    np.ndarray
        One score per pool row; higher means higher labeling priority.

    Raises
    ------
    ValidationError
        When the estimator type or strategy is unsupported.
    MissingExtraError
        When torch is not installed.
    """
    x_pool = np.asarray(x_pool, dtype=float)
    if x_pool.shape[0] == 0:
        return np.empty(0, dtype=float)
    if not isinstance(estimator, TabularMCDropoutClassifier):
        raise ValidationError(
            "Torch active-learning strategies require a fitted TabularMCDropoutClassifier."
        )
    n_samples = int(mc_samples if mc_samples is not None else estimator.mc_samples)
    proba_samples = []
    torch = require_torch_activelearning()
    device = torch.device(estimator.device)
    x_arr = np.asarray(x_pool, dtype=np.float32)
    estimator.module_.train()
    with torch.no_grad():
        for _ in range(n_samples):
            logits = estimator.module_(torch.as_tensor(x_arr, device=device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            proba_samples.append(probs)
    estimator.module_.eval()
    stack = np.stack(proba_samples, axis=0)  # (n_samples, n_pool, n_classes)
    mean_proba = np.clip(stack.mean(axis=0), 1e-12, 1.0)

    if strategy == "mc_dropout":
        return -np.sum(mean_proba * np.log(mean_proba), axis=1)

    if strategy == "bald":
        # BALD = H[E[p]] - E[H[p]] over MC samples.
        entropy_mean = -np.sum(mean_proba * np.log(mean_proba), axis=1)
        sample_entropy = -np.sum(np.clip(stack, 1e-12, 1.0) * np.log(np.clip(stack, 1e-12, 1.0)), axis=2)
        expected_entropy = sample_entropy.mean(axis=0)
        return entropy_mean - expected_entropy

    raise ValidationError(
        f"Unsupported torch active-learning strategy {strategy!r}. "
        "Supported: bald, mc_dropout."
    )


def _build_mlp(
    n_features: int,
    n_classes: int,
    *,
    hidden_dim: int,
    dropout_rate: float,
) -> Any:
    torch = require_torch_activelearning()

    # Dynamic torch import: base class is only known at runtime.
    class _MLP(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return _MLP()


def build_torch_estimator(
    *,
    random_state: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mc_samples: int,
    device: str,
) -> TabularMCDropoutClassifier:
    """Construct a default :class:`TabularMCDropoutClassifier` for active learning.

    Hyperparameters are taken from the Session fit call and stored on the plan
    config for later query scoring.

    Parameters
    ----------
    random_state:
        Seed for torch training minibatch shuffling.
    epochs:
        Training epochs for the tabular MLP.
    batch_size:
        Minibatch size during fit.
    learning_rate:
        AdamW learning rate.
    mc_samples:
        Default MC-dropout samples for query scoring.
    device:
        Torch device string (``cpu`` or ``cuda``).

    Returns
    -------
    TabularMCDropoutClassifier
        Unfitted classifier ready for :meth:`TabularMCDropoutClassifier.fit`.
    """
    return TabularMCDropoutClassifier(
        random_state=random_state,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        mc_samples=int(mc_samples),
        device=str(device),
    )
