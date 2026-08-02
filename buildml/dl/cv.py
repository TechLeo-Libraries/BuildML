"""Fold-local Torch cross-validation utilities.

Fits transforms / train loops inside each fold to honor leakage rules. For
nested Torch hyperparameter search (outer estimate after inner selection), use
:func:`buildml.dl.search.nested_cv_torch` / ``Session.nested_cv_torch``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.dl.dataset import arrays_to_tensor_dataset, infer_task, resolve_feature_target
from buildml.dl.extras import require_torch
from buildml.dl.labels import encode_class_targets, fit_class_labels, n_classes_from_labels
from buildml.dl.models import build_tabular_mlp
from buildml.dl.transforms import apply_standardize, fit_standardize, frame_to_numeric_matrix
from buildml.dl.types import TrainConfig

ModuleFactory = Callable[[int, str, int], Any]


@dataclass(slots=True)
class TorchFoldScore:
    """Metrics for one outer fold."""

    fold: int
    train_size: int
    val_size: int
    metrics: dict[str, float]
    n_epochs_ran: int
    device: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "metrics": dict(self.metrics),
            "n_epochs_ran": self.n_epochs_ran,
            "device": self.device,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class TorchCVResult:
    """Aggregate fold-local Torch CV outcome."""

    n_folds: int
    task: str
    fold_scores: tuple[TorchFoldScore, ...]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "task": self.task,
            "fold_scores": [f.to_dict() for f in self.fold_scores],
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


def _default_module_factory(in_features: int, task: str, n_classes: int) -> Any:
    return build_tabular_mlp(in_features, task=task, n_classes=max(n_classes, 2), hidden=(32, 16))


def _make_fold_indices(
    n_rows: int,
    *,
    n_folds: int,
    seed: int,
    y: np.ndarray | None,
    stratify: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds < 2:
        raise ValidationError("n_folds must be >= 2")
    if n_rows < n_folds:
        raise ValidationError(f"Need at least {n_folds} rows for {n_folds}-fold CV")
    rng = np.random.default_rng(seed)
    if stratify and y is not None:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n_rows), y))
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        pairs.append((train, val))
    return pairs


def _partition_xy(
    frame: pd.DataFrame,
    indices: np.ndarray,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    part = frame.iloc[list(indices)]
    x = frame_to_numeric_matrix(part, list(feature_columns))
    y_series = part[target_column]
    if not pd.api.types.is_numeric_dtype(y_series):
        raise ValidationError(
            f"Target '{target_column}' must be numeric for Torch CV "
            "(encode class labels to integers first)."
        )
    y = y_series.to_numpy(dtype=np.float64, copy=True)
    if np.isnan(y).any():
        raise ValidationError("Target contains NaN; clean labels before Torch CV")
    return x, y


def _eval_accuracy_or_mse(module: Any, loader: Any, *, task: str, device: str) -> dict[str, float]:
    torch = require_torch(feature="Torch CV metrics")
    module.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = module(xb)
            if task == "classification":
                loss = criterion(logits, yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
            else:
                loss = criterion(logits, yb)
            batch_n = int(xb.shape[0])
            total_loss += float(loss.item()) * batch_n
            n += batch_n
    metrics = {"loss": total_loss / max(n, 1)}
    if task == "classification":
        metrics["accuracy"] = correct / max(n, 1)
    else:
        metrics["mse"] = metrics["loss"]
    return metrics


def cross_validate_torch(
    dataset: Dataset,
    *,
    n_folds: int = 3,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    normalize: bool = True,
    seed: int = 0,
    stratify: bool = True,
    task: Literal["classification", "regression", "auto"] = "auto",
    module_factory: ModuleFactory | None = None,
    train_config: TrainConfig | None = None,
    apply_session_plans: Callable[[pd.DataFrame, np.ndarray], pd.DataFrame] | None = None,
) -> TorchCVResult:
    """Run fold-local supervised Torch CV on numeric tabular features.

    For each fold:
    1. Split train/val indices (optionally stratified).
    2. Optionally apply a caller-supplied fold-local plan transform on train rows.
    3. Fit normalize stats on train only when ``normalize=True``.
    4. Train a fresh module from ``module_factory`` (default tabular MLP).
    5. Score the held-out fold.

    Limitations (encoded in the result):
    - Not nested CV: no inner hyperparameter search loop (see
      :func:`buildml.dl.search.nested_cv_torch`).
    - Classical Session plans are not auto-refit per fold unless
      ``apply_session_plans`` is provided (receives train frame + train indices).
    - Text modality is not covered; use :func:`make_text_loaders` separately.
    """
    from buildml.dl.metrics import resolve_device
    from buildml.dl.results import TorchLoaderBundle
    from buildml.dl.train import train_supervised_module
    from buildml.dl.types import FeatureContract

    require_torch(feature="Torch cross-validation")
    feature_cols, target = resolve_feature_target(dataset)
    frame = dataset._ensure_pandas().copy()
    y_all = frame[target]
    if not pd.api.types.is_numeric_dtype(y_all):
        raise ValidationError(
            f"Target '{target}' must be numeric for Torch CV "
            "(encode class labels to integers first)."
        )
    y_values = y_all.to_numpy()
    resolved_task = infer_task(y_all, task)
    fold_pairs = _make_fold_indices(
        len(frame),
        n_folds=n_folds,
        seed=seed,
        y=y_values if resolved_task == "classification" else None,
        stratify=stratify and resolved_task == "classification",
    )
    factory = module_factory or _default_module_factory
    cfg = train_config or TrainConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
        seed=seed,
    )
    device_spec = resolve_device(cfg.device)
    fold_scores: list[TorchFoldScore] = []
    warnings: list[str] = []

    torch = require_torch(feature="Torch cross-validation")
    for fold_i, (train_idx, val_idx) in enumerate(fold_pairs):
        fold_warnings: list[str] = []
        work = frame
        if apply_session_plans is not None:
            work = apply_session_plans(frame.copy(), train_idx)
            fold_warnings.append("Applied fold-local plan transform supplied by caller.")

        x_train, y_train = _partition_xy(work, train_idx, feature_cols, target)
        x_val, y_val = _partition_xy(work, val_idx, feature_cols, target)
        mean = std = None
        if normalize:
            mean, std = fit_standardize(x_train)
            x_train = apply_standardize(x_train, mean, std)
            x_val = apply_standardize(x_val, mean, std)

        if resolved_task == "classification":
            class_labels = fit_class_labels(y_train)
            n_classes = n_classes_from_labels(class_labels)
            y_train = encode_class_targets(y_train, class_labels).astype(np.float64, copy=False)
            y_val = encode_class_targets(y_val, class_labels).astype(np.float64, copy=False)
        else:
            class_labels = ()
            n_classes = 1
        module = factory(len(feature_cols), resolved_task, n_classes)
        contract = FeatureContract(
            feature_columns=tuple(feature_cols),
            target_column=target,
            task=resolved_task,
            class_labels=class_labels,
            normalize_mean=None if mean is None else tuple(float(v) for v in mean),
            normalize_std=None if std is None else tuple(float(v) for v in std),
        )
        train_ds = arrays_to_tensor_dataset(x_train, y_train, task=resolved_task)
        val_ds = arrays_to_tensor_dataset(x_val, y_val, task=resolved_task)
        generator = torch.Generator().manual_seed(int(seed) + fold_i)
        loaders = {
            "train": torch.utils.data.DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                generator=generator,
            ),
            "validation": torch.utils.data.DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False
            ),
        }
        from buildml.dl.results import LoaderReport

        bundle = TorchLoaderBundle(
            loaders=loaders,
            contract=contract,
            report=LoaderReport(
                batch_size=cfg.batch_size,
                shuffle_train=True,
                normalize=normalize,
                feature_columns=contract.feature_columns,
                target_column=contract.target_column,
                task=resolved_task,
                n_train=len(train_idx),
                n_validation=len(val_idx),
                n_test=0,
                class_labels=contract.class_labels,
                warnings=fold_warnings,
            ),
        )
        result = train_supervised_module(module, bundle, config=cfg)
        metrics = _eval_accuracy_or_mse(
            result.module, loaders["validation"], task=resolved_task, device=result.device.resolved
        )
        fold_scores.append(
            TorchFoldScore(
                fold=fold_i,
                train_size=len(train_idx),
                val_size=len(val_idx),
                metrics=metrics,
                n_epochs_ran=result.n_epochs_ran,
                device=result.device.resolved,
                warnings=tuple(fold_warnings),
            )
        )

    metric_keys = sorted({k for fs in fold_scores for k in fs.metrics})
    mean_metrics = {
        k: float(np.mean([fs.metrics[k] for fs in fold_scores if k in fs.metrics]))
        for k in metric_keys
    }
    std_metrics = {
        k: float(np.std([fs.metrics[k] for fs in fold_scores if k in fs.metrics], ddof=0))
        for k in metric_keys
    }
    disclosures = (
        f"Fold-local Torch CV with n_folds={n_folds}, task={resolved_task}.",
        "Normalize statistics (when enabled) are fit per fold on train indices only.",
        f"Device resolved to {device_spec.resolved} (requested={device_spec.requested}).",
    )
    limitations = (
        "Not nested Torch CV: no inner hyperparameter search / selection loop "
        "(use nested_cv_torch for that).",
        "Classical Session preprocess plans are not auto-refit per fold unless "
        "apply_session_plans is provided.",
        "Text/sequence modality is out of scope for this helper.",
    )
    return TorchCVResult(
        n_folds=n_folds,
        task=resolved_task,
        fold_scores=tuple(fold_scores),
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        disclosures=disclosures,
        limitations=limitations,
        warnings=tuple(warnings),
        config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "normalize": normalize,
            "seed": seed,
            "stratify": stratify and resolved_task == "classification",
            "device": device_spec.to_dict(),
        },
    )
