"""Partition evaluation metrics for Torch modules."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch
from buildml.dl.results import DLEvaluateResult, TorchLoaderBundle, TrainResult
from buildml.dl.types import DeviceSpec


def resolve_device(requested: str = "auto") -> DeviceSpec:
    """Resolve a device string with explicit CPU fallback warnings."""
    torch = require_torch(feature="Torch device selection")
    want = (requested or "auto").lower()
    warning: str | None = None

    if want == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
        return DeviceSpec(requested=want, resolved=resolved, fallback_warning=None)

    if want == "cuda":
        if torch.cuda.is_available():
            return DeviceSpec(requested=want, resolved="cuda")
        warning = "CUDA was requested but is unavailable; using cpu."
        return DeviceSpec(requested=want, resolved="cpu", fallback_warning=warning)

    if want == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceSpec(requested=want, resolved="mps")
        warning = "MPS was requested but is unavailable; using cpu."
        return DeviceSpec(requested=want, resolved="cpu", fallback_warning=warning)

    if want == "cpu":
        return DeviceSpec(requested=want, resolved="cpu")

    if want.startswith("cuda:"):
        if torch.cuda.is_available():
            try:
                idx = int(want.split(":", 1)[1])
            except ValueError as exc:
                raise ValidationError(
                    f"Unknown device '{requested}'. Use cuda:N with integer N."
                ) from exc
            if idx < 0 or idx >= torch.cuda.device_count():
                warning = (
                    f"CUDA device {idx} unavailable "
                    f"(count={torch.cuda.device_count()}); using cpu."
                )
                return DeviceSpec(requested=want, resolved="cpu", fallback_warning=warning)
            return DeviceSpec(requested=want, resolved=want)
        warning = "CUDA was requested but is unavailable; using cpu."
        return DeviceSpec(requested=want, resolved="cpu", fallback_warning=warning)

    raise ValidationError(
        f"Unknown device '{requested}'. Use cpu, cuda, cuda:N, mps, or auto."
    )


def _predict_partition(
    module: Any,
    loader: Any,
    *,
    task: Literal["classification", "regression"],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch(feature="Torch evaluation")
    module.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    dev = torch.device(device)
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValidationError("Loader batch must be (inputs..., y)")
            inputs = batch[:-1]
            yb = batch[-1]
            if len(inputs) == 1:
                xb: Any = inputs[0].to(dev)
            else:
                xb = tuple(t.to(dev) for t in inputs)
            out = module(xb)
            if task == "classification":
                if out.ndim == 1:
                    pred = out.detach().cpu().numpy()
                else:
                    pred = out.argmax(dim=1).detach().cpu().numpy()
                truth = yb.detach().cpu().numpy().reshape(-1)
            else:
                pred = out.detach().cpu().numpy().reshape(-1)
                truth = yb.detach().cpu().numpy().reshape(-1)
            preds.append(np.asarray(pred))
            truths.append(np.asarray(truth))
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(truths)


def evaluate_module(
    train_result: TrainResult,
    loader_bundle: TorchLoaderBundle,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    device: str | None = None,
) -> DLEvaluateResult:
    """Score a trained module on one loader partition."""
    if partition not in loader_bundle.loaders:
        raise ValidationError(
            f"No DataLoader for partition '{partition}'. "
            "Rebuild loaders after split, or choose a non-empty partition."
        )
    device_name = device or train_result.device.resolved
    y_pred, y_true = _predict_partition(
        train_result.module,
        loader_bundle.loaders[partition],
        task=train_result.task,
        device=device_name,
    )
    tips: list[str] = []
    metrics: dict[str, float] = {}
    cm_list: list[list[int]] | None = None
    labels = train_result.contract.class_labels
    residuals: dict[str, float] | None = None

    if len(y_true) == 0:
        tips.append("Partition produced no rows; metrics are empty.")
        return DLEvaluateResult(
            partition=partition,
            task=train_result.task,
            metrics=metrics,
            n_rows=0,
            device=device_name,
            recommendations=tips,
            class_labels=labels,
        )

    if train_result.task == "regression":
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        residuals = {
            "mean": float(np.mean(resid)),
            "std": float(np.std(resid)),
            "p05": float(np.quantile(resid, 0.05)),
            "p50": float(np.quantile(resid, 0.50)),
            "p95": float(np.quantile(resid, 0.95)),
            "max_abs": float(np.max(np.abs(resid))),
        }
        if metrics["r2"] < 0:
            tips.append("Negative R² — model underperforms a mean baseline on this partition.")
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # Loaders encode targets to 0..K-1; class_labels[i] is the original id.
        if labels:
            inv = np.asarray(list(labels))
            idx_true = np.asarray(y_true, dtype=np.int64)
            idx_pred = np.asarray(y_pred, dtype=np.int64)
            y_true_cm = inv[idx_true]
            y_pred_cm = inv[idx_pred]
            label_values = list(labels)
            cm = confusion_matrix(y_true_cm, y_pred_cm, labels=label_values)
        else:
            label_values = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))
            cm = confusion_matrix(y_true, y_pred, labels=label_values)
        cm_list = cm.astype(int).tolist()

    early = train_result.early_stop
    if early and early.enabled:
        tips.append(
            f"Early stopping used monitor={early.monitor} on partition={early.partition} "
            f"(triggered={early.triggered}). Metrics below are for partition '{partition}'."
        )
    tips.append(
        f"Metrics are for partition '{partition}'. "
        "Do not tune on test; use validation for early stopping and model choices."
    )
    tips.append(f"Evaluation device={device_name}.")
    return DLEvaluateResult(
        partition=partition,
        task=train_result.task,
        metrics=metrics,
        n_rows=int(len(y_true)),
        device=device_name,
        recommendations=tips,
        confusion_matrix=cm_list,
        class_labels=labels,
        residuals_summary=residuals,
    )
