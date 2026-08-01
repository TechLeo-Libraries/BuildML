"""Multi-model comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.supervised import EvaluateResult, FitResult, evaluate_estimator, fit_estimator


@dataclass(slots=True)
class ModelComparison:
    """Ranked comparison of multiple estimators on the same split."""

    task: Literal["classification", "regression"]
    ranking_metric: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    fits: dict[str, FitResult] = field(default_factory=dict)
    evaluations: dict[str, EvaluateResult] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "ranking_metric": self.ranking_metric,
            "rows": list(self.rows),
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        print(self.to_frame().to_string(index=False))
        for tip in self.recommendations:
            print(f"- {tip}")


def compare_estimators(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimators: dict[str, Any],
    *,
    task: Literal["classification", "regression", "auto"] = "auto",
    partition: Literal["train", "validation", "test"] = "test",
    ranking_metric: str | None = None,
) -> ModelComparison:
    """Fit and evaluate multiple estimators; return a ranked comparison card.

    Parameters
    ----------
    dataset / split_plan:
        Prepared session data with roles and split.
    estimators:
        Mapping of display name → unfitted estimator.
    task:
        Task type or auto.
    partition:
        Evaluation partition.
    ranking_metric:
        Metric used for ranking. Defaults to ``f1_weighted`` / ``r2``.
    """
    if not estimators:
        raise ValueError("estimators mapping must not be empty")

    fits: dict[str, FitResult] = {}
    evaluations: dict[str, EvaluateResult] = {}
    rows: list[dict[str, Any]] = []
    resolved_task: Literal["classification", "regression"] | None = None

    for name, estimator in estimators.items():
        fit = fit_estimator(dataset, split_plan, estimator, task=task)
        ev = evaluate_estimator(dataset, split_plan, fit, partition=partition)
        fits[name] = fit
        evaluations[name] = ev
        resolved_task = fit.task
        row = {"model": name, **ev.metrics, "n_rows": ev.n_rows}
        rows.append(row)

    assert resolved_task is not None
    metric = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    higher_is_better = metric not in {"mae", "mse", "rmse", "log_loss", "median_ae", "mape"}
    rows.sort(key=lambda item: item.get(metric, float("-inf")), reverse=higher_is_better)

    tips = [
        f"Ranked by '{metric}' on partition='{partition}'.",
        "Refit the chosen winner on the full training recipe before deployment.",
    ]
    if len(rows) >= 2 and metric in rows[0] and metric in rows[1]:
        gap = abs(float(rows[0][metric]) - float(rows[1][metric]))
        tips.append(f"Top-2 gap on {metric}: {gap:.6f}")

    return ModelComparison(
        task=resolved_task,
        ranking_metric=metric,
        rows=rows,
        fits=fits,
        evaluations=evaluations,
        recommendations=tips,
    )
