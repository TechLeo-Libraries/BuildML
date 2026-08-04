"""Enrich ensemble evaluation with base-learner contribution and diversity.

Classical :func:`~buildml.model.supervised.evaluate_estimator` scores the
combined estimator. This module additionally scores each fitted base on the
same partition (without refitting) and summarises pairwise disagreement so
operators can see whether the ensemble is combining diverse signals or
averaging near-duplicates.

Leakage contract
----------------
Bases are the already train-fitted members of the ensemble. Contribution and
diversity use predict-only scoring on the named partition. Session test never
re-enters fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.ensemble.results import EnsemblePlan
from buildml.model.supervised import FitResult, _feature_target_frames


@dataclass(slots=True)
class BaseLearnerContribution:
    """Per-base metrics on one evaluation partition (predict-only)."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0
    agree_with_ensemble: float | None = None
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize one base contribution for diagnostics and proofs."""
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "n_rows": self.n_rows,
            "agree_with_ensemble": self.agree_with_ensemble,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class EnsembleDiversitySummary:
    """Pairwise disagreement / correlation among base learners."""

    n_bases: int
    n_rows: int
    mean_pairwise_disagreement: float | None = None
    pairwise_disagreement: dict[str, float] = field(default_factory=dict)
    mean_pairwise_corr: float | None = None
    pairwise_corr: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize diversity summary for diagnostics and proofs."""
        return {
            "n_bases": self.n_bases,
            "n_rows": self.n_rows,
            "mean_pairwise_disagreement": self.mean_pairwise_disagreement,
            "pairwise_disagreement": dict(self.pairwise_disagreement),
            "mean_pairwise_corr": self.mean_pairwise_corr,
            "pairwise_corr": dict(self.pairwise_corr),
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class EnsembleEvalReport:
    """Ensemble metrics plus base contribution and diversity summaries."""

    partition: str
    strategy: str
    task: Literal["classification", "regression"]
    ensemble_metrics: dict[str, float] = field(default_factory=dict)
    base_contributions: tuple[BaseLearnerContribution, ...] = ()
    diversity: EnsembleDiversitySummary | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full ensemble evaluation report."""
        return {
            "partition": self.partition,
            "strategy": self.strategy,
            "task": self.task,
            "ensemble_metrics": dict(self.ensemble_metrics),
            "base_contributions": [c.to_dict() for c in self.base_contributions],
            "diversity": None if self.diversity is None else self.diversity.to_dict(),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a compact contribution / diversity digest."""
        print(
            f"EnsembleEval · {self.strategy} · {self.task} · "
            f"partition={self.partition} · bases={len(self.base_contributions)}"
        )
        for tip in self.disclosures[:6]:
            print(f"  - {tip}")
        for contrib in self.base_contributions:
            key = "accuracy" if self.task == "classification" else "r2"
            score = contrib.metrics.get(key, contrib.metrics.get("mae"))
            agree = contrib.agree_with_ensemble
            agree_s = "" if agree is None else f" · agree={agree:.3f}"
            print(f"  {contrib.name}: {key}={score}{agree_s}")
        if self.diversity is not None and self.diversity.mean_pairwise_disagreement is not None:
            print(
                f"  diversity mean_pairwise_disagreement="
                f"{self.diversity.mean_pairwise_disagreement:.4f}"
            )


def _named_fitted_bases(estimator: Any) -> list[tuple[str, Any]]:
    named = getattr(estimator, "named_estimators_", None)
    if isinstance(named, dict) and named:
        return [(str(k), v) for k, v in named.items() if v is not None and v != "drop"]
    estimators = getattr(estimator, "estimators_", None)
    if isinstance(estimators, list) and estimators:
        out: list[tuple[str, Any]] = []
        for item in estimators:
            if isinstance(item, tuple) and len(item) == 2:
                out.append((str(item[0]), item[1]))
            else:
                out.append((type(item).__name__, item))
        return out
    raise ValidationError(
        "Fitted ensemble has no named_estimators_ / estimators_ to score for contributions."
    )


def _base_metrics(
    *,
    task: Literal["classification", "regression"],
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    if task == "regression":
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _agreement(a: np.ndarray, b: np.ndarray, *, task: str) -> float:
    if task == "regression":
        # Relative agreement within a small absolute tolerance of scale.
        scale = float(np.std(b)) if np.std(b) > 1e-12 else 1.0
        return float(np.mean(np.abs(a - b) <= 0.05 * scale))
    return float(np.mean(np.asarray(a).astype(str) == np.asarray(b).astype(str)))


def build_ensemble_eval_report(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult,
    plan: EnsemblePlan,
    *,
    partition: Literal["train", "validation", "test"] = "test",
    ensemble_metrics: dict[str, float] | None = None,
) -> EnsembleEvalReport:
    """Score fitted bases and summarise diversity on one partition.

    Parameters
    ----------
    dataset:
        Active dataset with feature/target roles.
    split_plan:
        Partition membership; required.
    fit_result:
        Classical fit result holding the fitted ensemble estimator.
    plan:
        Ensemble plan (strategy, task, base names).
    partition:
        Partition to score (default ``test``).
    ensemble_metrics:
        Optional already-computed ensemble metrics to attach.

    Returns
    -------
    EnsembleEvalReport
        Base contributions, diversity summary, and leakage disclosures.

    Raises
    ------
    ValidationError
        When split/plan/estimator state is insufficient.
    """
    if split_plan is None:
        raise ValidationError("A split is required for ensemble contribution reporting.")
    if fit_result.estimator is None:
        raise ValidationError("No fitted ensemble estimator on FitResult.")

    x_raw, y_true, _, _, _ = _feature_target_frames(dataset, split_plan, partition)
    selected = x_raw.loc[:, list(fit_result.feature_columns)]
    if not isinstance(selected, pd.DataFrame):
        raise ValidationError(
            "Ensemble contribution requires a DataFrame feature matrix after column selection"
        )
    x = selected
    ensemble_pred = np.asarray(fit_result.estimator.predict(x))
    bases = _named_fitted_bases(fit_result.estimator)
    task = plan.task

    pred_by_name: dict[str, np.ndarray] = {}
    contributions: list[BaseLearnerContribution] = []
    warnings: list[str] = []
    for name, est in bases:
        try:
            pred = np.asarray(est.predict(x))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Base {name!r} predict failed during contribution: {exc}")
            continue
        pred_by_name[name] = pred
        contributions.append(
            BaseLearnerContribution(
                name=name,
                metrics=_base_metrics(task=task, y_true=y_true, y_pred=pred),
                n_rows=int(len(y_true)),
                agree_with_ensemble=_agreement(pred, ensemble_pred, task=task),
                disclosures=(
                    "Predict-only scoring of a train-fitted base; no refit during evaluate.",
                ),
            )
        )

    pairwise_dis: dict[str, float] = {}
    pairwise_corr: dict[str, float] = {}
    names = list(pred_by_name)
    for a, b in combinations(names, 2):
        key = f"{a}|{b}"
        pa, pb = pred_by_name[a], pred_by_name[b]
        if task == "classification":
            pairwise_dis[key] = float(
                np.mean(np.asarray(pa).astype(str) != np.asarray(pb).astype(str))
            )
        else:
            pairwise_dis[key] = float(np.mean(np.abs(pa.astype(float) - pb.astype(float)) > 1e-12))
            if len(pa) >= 2 and float(np.std(pa)) > 0 and float(np.std(pb)) > 0:
                pairwise_corr[key] = float(np.corrcoef(pa.astype(float), pb.astype(float))[0, 1])

    mean_dis = (
        float(np.mean(list(pairwise_dis.values()))) if pairwise_dis else None
    )
    mean_corr = (
        float(np.mean(list(pairwise_corr.values()))) if pairwise_corr else None
    )
    diversity = EnsembleDiversitySummary(
        n_bases=len(names),
        n_rows=int(len(y_true)),
        mean_pairwise_disagreement=mean_dis,
        pairwise_disagreement=pairwise_dis,
        mean_pairwise_corr=mean_corr,
        pairwise_corr=pairwise_corr,
        disclosures=(
            "Diversity is computed from base predictions on the evaluation partition "
            "using train-fitted bases only.",
            "High mean_pairwise_disagreement usually indicates complementary bases; "
            "near-zero suggests redundant voters.",
        ),
    )

    disclosures = (
        f"Ensemble strategy={plan.strategy}; partition={partition}; "
        f"bases={list(plan.estimator_names)}.",
        "Base-learner contributions and diversity use predict-only scoring; "
        "Session test never re-enters ensemble fitting.",
        "Leakage-safe default: evaluate after fit; meta-learner / blend holdout "
        "were constrained to train during fit.",
    )
    return EnsembleEvalReport(
        partition=partition,
        strategy=plan.strategy,
        task=task,
        ensemble_metrics=dict(ensemble_metrics or {}),
        base_contributions=tuple(contributions),
        diversity=diversity,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )


__all__ = [
    "BaseLearnerContribution",
    "EnsembleDiversitySummary",
    "EnsembleEvalReport",
    "build_ensemble_eval_report",
]
