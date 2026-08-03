"""AutoGluon TabularPredictor adapter with Session leakage discipline."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from buildml.automl.extras import require_autogluon
from buildml.automl.results import AutoMLPlan, AutoMLResult, AutoMLTrial
from buildml.automl.types import AutoMLBudget, AutoMLConfig, AutoMLSelection
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.model.selection import _LOWER_IS_BETTER
from buildml.model.supervised import (
    FitResult,
    TaskType,
    _feature_target_frames,
    _infer_task,
    evaluate_estimator,
    weight_column,
)


def run_autogluon_adapter(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    task: TaskType = "auto",
    selection: AutoMLSelection = "cv",
    ranking_metric: str | None = None,
    time_budget: float | None = None,
    budget: AutoMLBudget | None = None,
    random_state: int | None = 0,
    refit: bool = True,
    presets: str | None = "medium_quality",
) -> tuple[AutoMLPlan, AutoMLResult, FitResult | None]:
    """Run AutoGluon TabularPredictor on Session train only; never touch test.

    AutoGluon performs internal model selection and featurization on train-only
    data. BuildML fold-local PreprocessRecipe search is bypassed — disclosures
    state this explicitly. Session test never enters fit or selection scoring.

    Parameters
    ----------
    dataset:
        BuildML dataset with features and a single target column.
    split_plan:
        Train/validation/test split; only train rows are passed to AutoGluon.
    task:
        ``classification``, ``regression``, or ``auto`` to infer from the target.
    selection:
        ``cv`` (train-only internal scoring), or ``validation`` to re-rank on
        the Session validation partition. ``nested`` is not supported.
    ranking_metric:
        Metric for leaderboard display; mapped to AutoGluon eval_metric names.
    time_budget:
        Wall-clock seconds for AutoGluon fit; overrides ``budget.max_time_seconds``.
    budget:
        Optional :class:`~buildml.automl.types.AutoMLBudget`; defaults when omitted.
    random_state:
        Seed hint passed where AutoGluon exposes reproducibility knobs.
    refit:
        When True, wrap the best model in a classical FitResult after search.
    presets:
        AutoGluon preset string (e.g. ``medium_quality``).

    Returns
    -------
    tuple[AutoMLPlan, AutoMLResult, FitResult | None]
        Train-selected plan, ranked leaderboard trials, and optional FitResult.

    Raises
    ------
    ValidationError
        When ``selection='nested'`` or validation partition is missing.
    MissingExtraError
        When autogluon is not installed.
    """
    TabularPredictor = require_autogluon(feature="AutoGluon AutoML backend")

    assert_fit_partition(split_plan, "train")
    if selection == "nested":
        raise ValidationError(
            "backend='autogluon' does not support selection='nested'. "
            "Use backend='native' with selection='nested' or selection='validation'."
        )
    if selection == "validation" and not split_plan.validation_indices:
        raise ValidationError(
            "selection='validation' requires a Session validation partition."
        )

    budget = budget or AutoMLBudget()
    seconds = time_budget if time_budget is not None else budget.max_time_seconds
    if seconds is None:
        seconds = max(60.0, float(budget.max_trials) * 8.0)

    x_train, y_train, feature_cols, target, _sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, None)
    metric = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    higher_is_better = metric not in _LOWER_IS_BETTER
    ag_metric = _autogluon_metric(metric, task=resolved_task)

    train_frame = x_train.copy()
    train_frame[target] = y_train.values

    predictor = TabularPredictor(
        label=target,
        problem_type=_autogluon_problem_type(resolved_task),
        eval_metric=ag_metric,
    )
    fit_kwargs: dict[str, Any] = {
        "time_limit": int(seconds),
        "presets": presets,
    }
    if random_state is not None:
        fit_kwargs["hyperparameters"] = {"GBM": {"ag_args_fit": {"num_cpus": 1}}}

    predictor.fit(train_frame, **fit_kwargs)

    leaderboard = predictor.leaderboard(silent=True)
    trials = _leaderboard_trials(leaderboard, metric=metric, higher_is_better=higher_is_better)

    best_model_name = str(predictor.get_model_best())
    train_eval = predictor.evaluate(train_frame, silent=True)
    best_score = float(train_eval.get(ag_metric, float("nan")))

    wrapper = _AutoGluonSklearnWrapper(predictor, target=target, feature_cols=feature_cols)
    fit_result: FitResult | None = None
    display_score = best_score
    std_score = 0.0

    if selection == "validation":
        fit_result = FitResult(
            estimator=wrapper,
            task=resolved_task,
            feature_columns=tuple(feature_cols),
            target_column=str(target),
            n_train_rows=int(len(x_train)),
            weight_column=weight_column(dataset),
        )
        ev = evaluate_estimator(dataset, split_plan, fit_result, partition="validation")
        display_score = float(ev.metrics.get(metric, best_score))
    elif refit:
        fit_result = FitResult(
            estimator=wrapper,
            task=resolved_task,
            feature_columns=tuple(feature_cols),
            target_column=str(target),
            n_train_rows=int(len(x_train)),
            weight_column=weight_column(dataset),
        )

    config = AutoMLConfig(
        method="randomized",
        selection=selection,
        task=resolved_task,
        n_trials=len(trials),
        ranking_metric=metric,
        include_recipe_search=False,
        random_state=random_state,
        families=("autogluon",),
        budget=budget,
        extras={
            "backend": "autogluon",
            "time_budget_seconds": seconds,
            "presets": presets,
            "best_model": best_model_name,
        },
    )
    disclosures = [
        "backend=autogluon (buildml[automl-industry]); AutoGluon internal stacking on train only.",
        "Fold-local PreprocessRecipe search bypassed — AutoGluon handles featurization internally.",
        "Session test never entered AutoGluon fit or selection scoring.",
        f"time_limit={int(seconds)}s; eval_metric={ag_metric}; best_model={best_model_name}.",
    ]
    limitations = [
        "AutoGluon adapter does not support nested CV or fold-local recipe strategy search.",
        "Saved estimator is a thin sklearn wrapper around TabularPredictor — use save_automl_bundle.",
    ]

    plan = AutoMLPlan(
        task=resolved_task,
        method="randomized",
        selection=selection,
        ranking_metric=metric,
        best_family="autogluon",
        best_recipe_strategy="autogluon_internal",
        best_kind="single",
        best_params={"model": best_model_name},
        best_recipe={},
        best_score=float(display_score),
        best_std=float(std_score),
        feature_columns=tuple(feature_cols),
        target_column=str(target),
        n_train_rows=int(len(x_train)),
        estimator_=wrapper if fit_result is None else fit_result.estimator,
        n_trials=len(trials),
        families_searched=("autogluon",),
        recipe_strategies_searched=("autogluon_internal",),
        disclosures=tuple(disclosures),
        warnings=(),
        config=config.to_dict(),
    )
    result = AutoMLResult(
        task=resolved_task,
        method="randomized",
        selection=selection,
        ranking_metric=metric,
        trials=trials,
        best_family="autogluon",
        best_recipe_strategy="autogluon_internal",
        best_kind="single",
        best_params={"model": best_model_name},
        best_score=float(display_score),
        best_std=float(std_score),
        families_searched=("autogluon",),
        recipe_strategies_searched=("autogluon_internal",),
        n_train_rows=int(len(x_train)),
        feature_columns=tuple(feature_cols),
        target_column=str(target),
        disclosures=tuple(disclosures),
        warnings=(),
        limitations=tuple(limitations),
        recommendations=(
            f"Winner: autogluon model={best_model_name}, score={display_score:.6f}.",
            "Confirm once on held-out test via evaluate_automl(partition='test').",
        ),
        config=config.to_dict(),
    )
    return plan, result, fit_result


def _autogluon_problem_type(task: Literal["classification", "regression"]) -> str:
    return "binary" if task == "classification" else "regression"


def _autogluon_metric(metric: str, *, task: Literal["classification", "regression"]) -> str:
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1",
        "f1_weighted": "f1",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
        "log_loss": "log_loss",
        "r2": "r2",
        "mse": "mean_squared_error",
        "mae": "mean_absolute_error",
        "rmse": "root_mean_squared_error",
    }
    if metric in mapping:
        return mapping[metric]
    return "f1" if task == "classification" else "r2"


def _leaderboard_trials(
    leaderboard: pd.DataFrame,
    *,
    metric: str,
    higher_is_better: bool,
) -> list[AutoMLTrial]:
    rows: list[AutoMLTrial] = []
    if leaderboard is None or leaderboard.empty:
        return rows
    score_col = None
    for col in leaderboard.columns:
        if "score" in col.lower() or col in {"score_val", "score_test"}:
            score_col = col
            break
    if score_col is None:
        score_col = leaderboard.columns[-1]
    for i, row in leaderboard.reset_index(drop=True).iterrows():
        score = float(row.get(score_col, float("nan")))
        model = str(row.get("model", f"model_{i}"))
        rows.append(
            AutoMLTrial(
                trial=int(i),
                kind="single",
                family="autogluon",
                recipe_strategy="autogluon_internal",
                params={"model": model},
                mean_score=score,
                std_score=0.0,
                mean_metrics={metric: score},
                std_metrics={metric: 0.0},
            )
        )
    rows.sort(key=lambda t: t.mean_score, reverse=higher_is_better)
    return rows


class _AutoGluonSklearnWrapper:
    """Minimal sklearn-compatible wrapper for Session evaluate/predict."""

    def __init__(
        self,
        predictor: Any,
        *,
        target: str,
        feature_cols: list[str],
    ) -> None:
        """Store the fitted TabularPredictor and feature contract.

        Retains column names and the underlying predictor so Session
        evaluate/predict can call AutoGluon through a sklearn-compatible API.

        Parameters
        ----------
        predictor:
            Fitted AutoGluon ``TabularPredictor`` instance.
        target:
            Target column name excluded from feature predictions.
        feature_cols:
            Feature column names passed to AutoGluon at fit time.
        """
        self._predictor = predictor
        self._target = target
        self._feature_cols = list(feature_cols)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> _AutoGluonSklearnWrapper:
        """No-op fit for sklearn compatibility; predictor is already trained.

        AutoGluon fit happens in :func:`run_autogluon_adapter`; this method
        exists so sklearn pipelines and Session wrappers accept the object.

        Parameters
        ----------
        X:
            Ignored; AutoGluon was fit externally on train data.
        y:
            Ignored.
        **kwargs:
            Ignored.

        Returns
        -------
        _AutoGluonSklearnWrapper
            ``self`` for sklearn pipeline compatibility.
        """
        del kwargs
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict labels using the wrapped TabularPredictor.

        Subsets ``X`` to ``feature_cols`` before delegating to AutoGluon so
        Session frames with extra columns still score correctly.

        Parameters
        ----------
        X:
            Feature frame; only ``feature_cols`` are forwarded to AutoGluon.

        Returns
        -------
        array-like
            Predictions from ``TabularPredictor.predict``.
        """
        frame = X[self._feature_cols].copy()
        return self._predictor.predict(frame)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Predict class probabilities when the underlying model supports them.

        Forwards only ``feature_cols`` to AutoGluon; classification models
        must expose ``predict_proba`` on the TabularPredictor.

        Parameters
        ----------
        X:
            Feature frame; only ``feature_cols`` are forwarded to AutoGluon.

        Returns
        -------
        array-like
            Probability matrix from ``TabularPredictor.predict_proba``.

        Raises
        ------
        AttributeError
            When the underlying TabularPredictor has no ``predict_proba``.
        """
        frame = X[self._feature_cols].copy()
        if hasattr(self._predictor, "predict_proba"):
            return self._predictor.predict_proba(frame)
        raise AttributeError("Underlying TabularPredictor has no predict_proba.")
