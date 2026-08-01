"""Helpers for explain / history summaries of DL operations."""

from __future__ import annotations

from typing import Any

from buildml.dl.curves import build_training_curve, torch_training_status
from buildml.dl.results import DLEvaluateResult, LoaderReport, TrainResult


def loader_summary(report: LoaderReport) -> dict[str, Any]:
    """Compact result_summary for ``make_torch_loaders`` history."""
    return report.to_dict()


def train_summary(result: TrainResult) -> dict[str, Any]:
    """Compact result_summary for ``fit_torch`` history."""
    return result.to_dict()


def evaluate_summary(result: DLEvaluateResult) -> dict[str, Any]:
    """Compact result_summary for ``evaluate_torch`` history."""
    return result.to_dict()


def curve_summary(result: TrainResult) -> dict[str, Any]:
    """Compact training-curve payload for teaching surfaces."""
    curve = result.training_curve or build_training_curve(result)
    return curve.to_dict()


def training_status_for_session(session: Any) -> dict[str, Any]:
    """Build walkthrough / Studio torch_training_status from a Session."""
    return torch_training_status(
        train_result=getattr(session, "dl_train_result", None),
        history=list(getattr(session, "history", []) or []),
    )
