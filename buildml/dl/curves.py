"""Training-curve structured data and teaching interpretation (Torch-free)."""

from __future__ import annotations

from typing import Any

from buildml.dl.results import EarlyStopInfo, TrainingCurveReport, TrainResult


def build_training_curve(result: TrainResult) -> TrainingCurveReport:
    """Derive curve series, disclosures, and limits from a :class:`TrainResult`."""
    history = list(result.history)
    epochs = [int(row.get("epoch", 0)) for row in history]
    train_loss = [float(row["train_loss"]) for row in history if "train_loss" in row]
    # Align lengths when log_every skipped some rows — use history order as-is.
    if len(train_loss) != len(history):
        train_loss = [float(row.get("train_loss", float("nan"))) for row in history]
    val_loss: list[float | None] = [
        None if row.get("val_loss") is None else float(row["val_loss"]) for row in history
    ]
    learning_rates: list[float | None] = [
        None if row.get("lr") is None else float(row["lr"]) for row in history
    ]

    early = result.early_stop
    monitor = early.monitor if early and early.enabled else (
        "val_loss" if any(v is not None for v in val_loss) else "train_loss"
    )
    monitor_values: list[float | None] = []
    for row in history:
        if monitor in row:
            monitor_values.append(float(row[monitor]))
        elif monitor == "val_loss":
            monitor_values.append(
                None if row.get("val_loss") is None else float(row["val_loss"])
            )
        else:
            monitor_values.append(
                None if row.get("train_loss") is None else float(row["train_loss"])
            )

    early_stop_epoch = early.best_epoch if early and early.triggered else None
    partition = early.partition if early and early.enabled else None

    interpretation = _interpret(
        train_loss=train_loss,
        val_loss=val_loss,
        early=early,
        n_epochs_ran=result.n_epochs_ran,
        resumed_from=result.resumed_from_epochs,
    )
    limitations = [
        "Epoch aggregates are preferred for claims; batch losses are noisier and not stored here.",
        (
            "Curves describe this run under the recorded device, split, and "
            "TrainConfig — not deployment risk."
        ),
        (
            "Validation improvement does not prove test performance; "
            "call evaluate_torch(partition='test') once frozen."
        ),
    ]
    if early is None or not early.enabled:
        limitations.append(
            "Early stopping was disabled; the final epoch may not be the best validation epoch."
        )
    disclosures = [
        f"Device resolved={result.device.resolved} (requested={result.device.requested}).",
        f"Scheduler={result.scheduler_name}; grad_clip_norm={result.config.grad_clip_norm}.",
    ]
    if result.device.fallback_warning:
        disclosures.append(result.device.fallback_warning)
    if early and early.enabled:
        disclosures.append(
            f"Early-stopping monitor={early.monitor} on partition={early.partition}; "
            f"triggered={early.triggered}; reason={early.reason}."
        )
    else:
        disclosures.append(
            "No early-stopping monitor was active; training ran for the configured epoch budget."
        )
    if result.resumed_from_epochs:
        disclosures.append(
            f"Training resumed from epoch offset {result.resumed_from_epochs} "
            f"(total epochs recorded={result.n_epochs_ran})."
        )

    return TrainingCurveReport(
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rates=learning_rates,
        monitor=monitor,
        monitor_values=monitor_values,
        early_stop_epoch=early_stop_epoch,
        device_resolved=result.device.resolved,
        early_stop_partition=partition,
        interpretation=interpretation,
        limitations=limitations,
        disclosures=disclosures,
    )


def torch_training_status(
    *,
    train_result: TrainResult | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Walkthrough / Teaching Studio disclosure for Torch training curves."""
    if train_result is None:
        # Fallback: detect fit_torch in Session history without a live result.
        records = list(history or [])
        saw = any(
            str(r.get("operation_id") or r.get("action")) == "fit_torch" for r in records
        )
        return {
            "enabled": False,
            "present": saw,
            "disclosures": (
                [
                    "fit_torch appears in Session history, but no live dl_train_result "
                    "is attached for curve disclosure."
                ]
                if saw
                else []
            ),
            "training_curve": None,
            "early_stop": None,
            "device": None,
        }

    curve = train_result.training_curve or build_training_curve(train_result)
    early = None if train_result.early_stop is None else train_result.early_stop.to_dict()
    return {
        "enabled": True,
        "present": True,
        "disclosures": list(curve.disclosures),
        "interpretation": list(curve.interpretation),
        "limitations": list(curve.limitations),
        "training_curve": curve.to_dict(),
        "early_stop": early,
        "device": train_result.device.to_dict(),
        "n_epochs_ran": train_result.n_epochs_ran,
        "scheduler_name": train_result.scheduler_name,
        "resumed_from_epochs": train_result.resumed_from_epochs,
    }


def _interpret(
    *,
    train_loss: list[float],
    val_loss: list[float | None],
    early: EarlyStopInfo | None,
    n_epochs_ran: int,
    resumed_from: int,
) -> list[str]:
    tips: list[str] = []
    if not train_loss:
        return ["No epoch history rows were recorded."]
    tips.append(
        f"Recorded {len(train_loss)} history row(s); n_epochs_ran={n_epochs_ran}"
        + (f" (resumed from {resumed_from})." if resumed_from else ".")
    )
    if train_loss[0] == train_loss[0] and train_loss[-1] == train_loss[-1]:
        if train_loss[-1] < train_loss[0]:
            tips.append("Train loss decreased from the first recorded epoch to the last.")
        elif train_loss[-1] > train_loss[0]:
            tips.append(
                "Train loss rose versus the first recorded epoch — check LR, clipping, or labels."
            )
    finite_val = [v for v in val_loss if v is not None]
    if len(finite_val) >= 2 and train_loss:
        last_train = train_loss[-1]
        last_val = finite_val[-1]
        if last_val > last_train * 1.25 and last_train > 0:
            tips.append(
                "Late validation loss sits well above train loss — possible overfitting; "
                "prefer early-stop on validation or simplify the module."
            )
    if early and early.enabled:
        if early.triggered:
            tips.append(
                f"Early stopping triggered at epoch {early.stopped_epoch} "
                f"(best {early.monitor}={early.best_value} at epoch {early.best_epoch} "
                f"on {early.partition})."
            )
        else:
            tips.append(
                f"Early stopping was enabled (patience={early.patience}) but did not trigger; "
                f"reason={early.reason}."
            )
    return tips
