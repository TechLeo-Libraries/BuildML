"""Typed results for AutoML pipeline / model-family search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from buildml.automl.types import AutoMLMethod, AutoMLSelection, CandidateKind
from buildml.model.supervised import FitResult


@dataclass(slots=True)
class AutoMLTrial:
    """One AutoML candidate with selection evidence."""

    trial: int
    kind: CandidateKind
    family: str
    recipe_strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)
    mean_score: float = float("nan")
    std_score: float = float("nan")
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    ensemble_bases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize one trial for history and comparison exports.

        Captures family, recipe strategy, hyperparameters, and fold CV scores
        without embedding fitted estimators.

        Returns
        -------
        dict[str, Any]
            Family, recipe strategy, params, and fold CV score summary.
        """
        return {
            "trial": self.trial,
            "kind": self.kind,
            "family": self.family,
            "recipe_strategy": self.recipe_strategy,
            "params": dict(self.params),
            "recipe": dict(self.recipe),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "ensemble_bases": list(self.ensemble_bases),
        }


@dataclass(slots=True)
class AutoMLPlan:
    """Train-selected AutoML plan (best pipeline + disclosures).

    The fitted sklearn-compatible estimator (often a Pipeline of fold-local
    preprocess + model) also lives on Session ``FitResult`` so classical
    ``evaluate`` / ``predict`` / ``save_pipeline`` keep working.
    Persist via ``buildml.automl_bundle.v1``.
    """

    task: Literal["classification", "regression"]
    method: AutoMLMethod
    selection: AutoMLSelection
    ranking_metric: str
    best_family: str
    best_recipe_strategy: str
    best_kind: CandidateKind
    best_params: dict[str, Any]
    best_recipe: dict[str, Any]
    best_score: float
    best_std: float
    feature_columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    estimator_: Any = field(repr=False)
    ensemble_bases: tuple[str, ...] = ()
    n_trials: int = 0
    families_searched: tuple[str, ...] = ()
    recipe_strategies_searched: tuple[str, ...] = ()
    outer_score_mean: float | None = None
    outer_score_std: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a JSON-friendly dict (no private estimator).

        Omits ``estimator_`` so bundles and history stay lightweight; stores
        only the estimator class name.

        Returns
        -------
        dict[str, Any]
            Best family/recipe, feature contract, and disclosure fields.
        """
        return {
            "task": self.task,
            "method": self.method,
            "selection": self.selection,
            "ranking_metric": self.ranking_metric,
            "best_family": self.best_family,
            "best_recipe_strategy": self.best_recipe_strategy,
            "best_kind": self.best_kind,
            "best_params": dict(self.best_params),
            "best_recipe": dict(self.best_recipe),
            "best_score": self.best_score,
            "best_std": self.best_std,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "estimator": type(self.estimator_).__name__,
            "ensemble_bases": list(self.ensemble_bases),
            "n_trials": self.n_trials,
            "families_searched": list(self.families_searched),
            "recipe_strategies_searched": list(self.recipe_strategies_searched),
            "outer_score_mean": self.outer_score_mean,
            "outer_score_std": self.outer_score_std,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class AutoMLResult:
    """Outcome of an AutoML search with ranked trials and disclosures."""

    task: Literal["classification", "regression"]
    method: AutoMLMethod
    selection: AutoMLSelection
    ranking_metric: str
    trials: list[AutoMLTrial] = field(default_factory=list)
    best_family: str = ""
    best_recipe_strategy: str = ""
    best_kind: CandidateKind = "single"
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    best_std: float | None = None
    outer_score_mean: float | None = None
    outer_score_std: float | None = None
    families_searched: tuple[str, ...] = ()
    recipe_strategies_searched: tuple[str, ...] = ()
    n_train_rows: int = 0
    feature_columns: tuple[str, ...] = ()
    target_column: str = ""
    ensemble_bases: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return ranked trials as a pandas DataFrame for inspection.

        One row per trial with family, recipe strategy, scores, and flattened
        hyperparameter columns prefixed with ``param_``.

        Returns
        -------
        pandas.DataFrame
            Tabular trial comparison suitable for sorting and export.
        """
        return self.leaderboard()

    def leaderboard(self, *, top_n: int | None = None) -> pd.DataFrame:
        """Return a rich AutoML leaderboard with selection / nested-CV fields.

        Ranks trials by ``mean_score`` (descending), adds ``rank`` and
        ``gap_to_best``, and broadcasts selection-mode disclosures
        (``selection``, ``outer_score_mean`` / ``outer_score_std``) so a
        default ``selection='cv'`` run cannot be mistaken for nested outer
        evidence.

        Parameters
        ----------
        top_n:
            Optional cap on rows after ranking. ``None`` returns every trial.

        Returns
        -------
        pandas.DataFrame
            Leaderboard with family, recipe, kind, scores, selection context,
            and flattened ``param_*`` columns.
        """
        ranked = sorted(
            self.trials,
            key=lambda t: (
                -1.0 if t.mean_score != t.mean_score else -float(t.mean_score)
            ),
        )
        best = ranked[0].mean_score if ranked else float("nan")
        rows: list[dict[str, Any]] = []
        for rank, t in enumerate(ranked, start=1):
            gap = (
                float("nan")
                if best != best or t.mean_score != t.mean_score
                else float(best - t.mean_score)
            )
            rows.append(
                {
                    "rank": rank,
                    "trial": t.trial,
                    "kind": t.kind,
                    "family": t.family,
                    "recipe_strategy": t.recipe_strategy,
                    "mean_score": t.mean_score,
                    "std_score": t.std_score,
                    "gap_to_best": gap,
                    "selection": self.selection,
                    "ranking_metric": self.ranking_metric,
                    "outer_score_mean": self.outer_score_mean,
                    "outer_score_std": self.outer_score_std,
                    "nested_cv_disclosed": self.selection == "nested",
                    "ensemble_bases": list(t.ensemble_bases),
                    **{f"mean_{k}": v for k, v in t.mean_metrics.items()},
                    **{f"param_{k}": v for k, v in t.params.items()},
                }
            )
            if top_n is not None and len(rows) >= int(top_n):
                break
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full search outcome for history and bundle metadata.

        Includes ranked trials, best candidate summary, and teaching disclosures.

        Returns
        -------
        dict[str, Any]
            Task, method, trials, best family/recipe, and limitation notes.
        """
        return {
            "task": self.task,
            "method": self.method,
            "selection": self.selection,
            "ranking_metric": self.ranking_metric,
            "trials": [t.to_dict() for t in self.trials],
            "best_family": self.best_family,
            "best_recipe_strategy": self.best_recipe_strategy,
            "best_kind": self.best_kind,
            "best_params": dict(self.best_params),
            "best_score": self.best_score,
            "best_std": self.best_std,
            "outer_score_mean": self.outer_score_mean,
            "outer_score_std": self.outer_score_std,
            "families_searched": list(self.families_searched),
            "recipe_strategies_searched": list(self.recipe_strategies_searched),
            "n_train_rows": self.n_train_rows,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "ensemble_bases": list(self.ensemble_bases),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "recommendations": list(self.recommendations),
            "config": dict(self.config),
        }

    def show(self) -> None:
        """Print a human-readable summary of the search to stdout.

        Shows method, selection mode, best trial score, optional outer CV
        estimate, and the first few disclosure lines.
        """
        print(
            f"AutoML · {self.method}/{self.selection} · {self.task} · "
            f"ranked by {self.ranking_metric} · trials={len(self.trials)}"
        )
        if self.selection == "cv":
            print(
                "  note: default selection='cv' ranks by train-fold CV; "
                "use selection='nested' for an outer post-selection estimate."
            )
        if self.best_score is not None:
            std = "" if self.best_std is None else f" ± {self.best_std:.6f}"
            print(
                f"  best: {self.best_family}/{self.best_recipe_strategy} "
                f"({self.best_kind}) = {self.best_score:.6f}{std}"
            )
        if self.outer_score_mean is not None:
            std = (
                ""
                if self.outer_score_std is None
                else f" ± {self.outer_score_std:.6f}"
            )
            print(f"  outer (nested): {self.outer_score_mean:.6f}{std}")
        elif self.selection == "nested":
            print("  outer (nested): unavailable (all outer folds failed)")
        for tip in self.disclosures[:6]:
            print(f"  - {tip}")


def fit_result_from_plan(plan: AutoMLPlan) -> FitResult:
    """Build a classical FitResult from an AutoMLPlan estimator.

    Bridges AutoML search output to Session evaluate/predict/save_pipeline by
    wrapping the train-refit estimator and feature contract from the plan.

    Parameters
    ----------
    plan:
        Fitted :class:`AutoMLPlan` with ``estimator_`` attached.

    Returns
    -------
    FitResult
        Classical fit result ready for :func:`buildml.model.supervised.evaluate_estimator`.
    """
    return FitResult(
        estimator=plan.estimator_,
        task=plan.task,
        feature_columns=plan.feature_columns,
        target_column=plan.target_column,
        n_train_rows=plan.n_train_rows,
    )
