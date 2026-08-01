"""Helpers for explain / history summaries of DL operations."""

from __future__ import annotations

from typing import Any

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
