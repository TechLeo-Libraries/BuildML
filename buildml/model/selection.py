"""Honest cross-validation and hyperparameter search for classical estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ParameterSampler,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    check_cv,
)
from sklearn.pipeline import Pipeline as SkPipeline

from buildml.core.errors import LeakageError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.model.supervised import (
    FitResult,
    TaskType,
    _feature_target_frames,
    _infer_task,
    fit_estimator,
    fit_kwargs_for_sample_weight,
    validate_sample_weights,
    weight_column,
)
from buildml.preprocess.fold import (
    SAFE_RECIPE_KNOBS,
    PreprocessRecipe,
    build_fold_preprocessor,
    transform_fold_features,
)

CvStrategy = Literal["auto", "kfold", "stratified", "group", "stratified_group", "time"]
SearchMethod = Literal["grid", "randomized", "optuna"]
InnerSearchMethod = Literal["auto", "grid", "randomized", "optuna"]
OptunaSpace = Callable[[Any], dict[str, Any]] | dict[str, Any]

_LOWER_IS_BETTER = {"mae", "mse", "rmse", "log_loss", "median_ae", "mape"}


@dataclass(slots=True)
class FoldScore:
    """Metrics for a single CV fold."""

    fold: int
    n_train: int
    n_eval: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class CVScoreResult:
    """Structured cross-validation card with fold spread and interpretation."""

    task: Literal["classification", "regression"]
    scoring_metric: str
    cv_strategy: str
    n_splits: int
    folds: list[FoldScore] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    population: str = "train"
    held_out_partitions: tuple[str, ...] = ()
    fold_preprocess: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"fold": fold.fold, "n_train": fold.n_train, "n_eval": fold.n_eval, **fold.metrics}
            for fold in self.folds
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "scoring_metric": self.scoring_metric,
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "folds": [fold.to_dict() for fold in self.folds],
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "population": self.population,
            "held_out_partitions": list(self.held_out_partitions),
            "fold_preprocess": self.fold_preprocess,
            "limitations": list(self.limitations),
            "interpretation": list(self.interpretation),
            "recommendations": list(self.recommendations),
            "params": dict(self.params),
        }

    def show(self) -> None:
        metric = self.scoring_metric
        mean = self.mean_metrics.get(metric)
        std = self.std_metrics.get(metric)
        print(
            f"CVScore · {self.task} · {self.cv_strategy} · {self.n_splits}-fold · "
            f"population={self.population}"
        )
        if mean is not None and std is not None:
            print(f"  {metric}: {mean:.6f} ± {std:.6f}")
        for key, value in self.mean_metrics.items():
            if key == metric:
                continue
            print(f"  {key}: {value:.6f} ± {self.std_metrics.get(key, float('nan')):.6f}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")


@dataclass(slots=True)
class SearchTrial:
    """One hyperparameter trial with nested CV evidence."""

    trial: int
    params: dict[str, Any]
    mean_score: float
    std_score: float
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    recipe_knobs: dict[str, Any] = field(default_factory=dict)
    cv: CVScoreResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "params": dict(self.params),
            "recipe_knobs": dict(self.recipe_knobs),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
        }


@dataclass(slots=True)
class SearchResult:
    """Ranked hyperparameter search card with best params and evidence."""

    method: SearchMethod
    task: Literal["classification", "regression"]
    ranking_metric: str
    trials: list[SearchTrial] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_recipe_knobs: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    best_std: float | None = None
    best_cv: CVScoreResult | None = None
    refit_result: FitResult | None = None
    interpretation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    study: Any | None = None

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "trial": trial.trial,
                "mean_score": trial.mean_score,
                "std_score": trial.std_score,
                **{f"param_{k}": v for k, v in trial.params.items()},
                **{f"recipe_{k}": v for k, v in trial.recipe_knobs.items()},
            }
            for trial in self.trials
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "ranking_metric": self.ranking_metric,
            "trials": [trial.to_dict() for trial in self.trials],
            "best_params": dict(self.best_params),
            "best_recipe_knobs": dict(self.best_recipe_knobs),
            "best_score": self.best_score,
            "best_std": self.best_std,
            "interpretation": list(self.interpretation),
            "recommendations": list(self.recommendations),
            "limitations": list(self.limitations),
            "refit": None if self.refit_result is None else self.refit_result.to_dict(),
        }

    def show(self) -> None:
        print(f"Search · {self.method} · ranked by {self.ranking_metric}")
        if self.best_score is not None and self.best_std is not None:
            print(f"  best: {self.best_score:.6f} ± {self.best_std:.6f}")
            print(f"  params: {self.best_params}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")


@dataclass(slots=True)
class OuterFoldResult:
    """One outer-fold score after inner-loop selection."""

    fold: int
    n_train: int
    n_eval: int
    best_params: dict[str, Any] = field(default_factory=dict)
    best_recipe_knobs: dict[str, Any] = field(default_factory=dict)
    inner_best_score: float | None = None
    inner_best_std: float | None = None
    inner_n_trials: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "best_params": dict(self.best_params),
            "best_recipe_knobs": dict(self.best_recipe_knobs),
            "inner_best_score": self.inner_best_score,
            "inner_best_std": self.inner_best_std,
            "inner_n_trials": self.inner_n_trials,
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class NestedCVResult:
    """Outer-loop estimate after inner hyperparameter / selection search.

    Outer folds score a configuration chosen only from that fold's inner
    train-CV. Session test / validation partitions never enter outer or inner
    membership.
    """

    task: Literal["classification", "regression"]
    scoring_metric: str
    outer_cv_strategy: str
    inner_cv_strategy: str
    n_outer_splits: int
    n_inner_splits: int
    search_method: SearchMethod
    outer_folds: list[OuterFoldResult] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    inner_selection_summary: dict[str, Any] = field(default_factory=dict)
    population: str = "train"
    held_out_partitions: tuple[str, ...] = ()
    fold_preprocess: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    warm_start_studies: bool = False

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "fold": fold.fold,
                "n_train": fold.n_train,
                "n_eval": fold.n_eval,
                "inner_best_score": fold.inner_best_score,
                "inner_best_std": fold.inner_best_std,
                **fold.metrics,
                **{f"param_{k}": v for k, v in fold.best_params.items()},
                **{f"recipe_{k}": v for k, v in fold.best_recipe_knobs.items()},
            }
            for fold in self.outer_folds
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "scoring_metric": self.scoring_metric,
            "outer_cv_strategy": self.outer_cv_strategy,
            "inner_cv_strategy": self.inner_cv_strategy,
            "n_outer_splits": self.n_outer_splits,
            "n_inner_splits": self.n_inner_splits,
            "search_method": self.search_method,
            "outer_folds": [fold.to_dict() for fold in self.outer_folds],
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "inner_selection_summary": dict(self.inner_selection_summary),
            "population": self.population,
            "held_out_partitions": list(self.held_out_partitions),
            "fold_preprocess": self.fold_preprocess,
            "limitations": list(self.limitations),
            "interpretation": list(self.interpretation),
            "recommendations": list(self.recommendations),
            "warm_start_studies": self.warm_start_studies,
        }

    def show(self) -> None:
        metric = self.scoring_metric
        mean = self.mean_metrics.get(metric)
        std = self.std_metrics.get(metric)
        print(
            f"NestedCV · {self.task} · outer={self.outer_cv_strategy}/"
            f"{self.n_outer_splits} · inner={self.search_method}/"
            f"{self.n_inner_splits} · population={self.population}"
        )
        if mean is not None and std is not None:
            print(f"  outer {metric}: {mean:.6f} ± {std:.6f}")
        for tip in self.recommendations[:8]:
            print(f"  - {tip}")


def _refuse_session_global_cv_leakage(
    *,
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
    allow_session_global_preprocess: bool,
) -> None:
    """Hard-refuse CV/search when Session-global prep already poisoned the frame.

    A fold-local :class:`~buildml.preprocess.fold.PreprocessRecipe` does **not**
    undo Session-global transforms — recipes run on the current (already
    transformed) design matrix unless the caller re-ingests / reattaches
    unpoisoned data. Opt in only via ``allow_session_global_preprocess=True``.
    """
    if not session_preprocess_applied:
        return
    if allow_session_global_preprocess:
        return
    recipe_note = ""
    if preprocess is not None and not preprocess.is_empty():
        recipe_note = (
            " A fold-local PreprocessRecipe was provided, but Session data is already "
            "transformed with train-global statistics — the recipe cannot rebuild from "
            "raw/unpoisoned rows. Re-ingest or checkpoint_load an unpoisoned frame, then "
            "use fold-local recipes without Session-global impute/encode/scale/select/…"
            " first."
        )
    else:
        recipe_note = (
            " Pass preprocess=PreprocessRecipe(...) on unpoisoned data for fold-local "
            "refits, or set allow_session_global_preprocess=True to override explicitly "
            "(scores remain leakage-biased)."
        )
    raise LeakageError(
        "Refusing CV/search because Session-global preprocess plans were already "
        "fitted on the full train partition (fold-eval rows influenced those frozen "
        f"statistics).{recipe_note}"
    )


def cv_score(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    scoring_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    params: dict[str, Any] | None = None,
    recipe_knobs: dict[str, Any] | None = None,
) -> CVScoreResult:
    """Cross-validate an estimator on the train partition only.

    The Session test (and validation) partitions are never used for fold
    membership or scoring. Optional ``preprocess`` recipes are fitted on each
    fold's training rows only. ``recipe_knobs`` override safe fold-local
    controls (for example ``select_k``, ``n_bins``) on that recipe copy.

    When Session-global fit-capable plans already exist, this refuses with
    :class:`~buildml.core.errors.LeakageError` unless
    ``allow_session_global_preprocess=True``. A fold-local ``preprocess`` recipe
    alone is not enough — Session data may already be poisoned.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _refuse_session_global_cv_leakage(
        session_preprocess_applied=session_preprocess_applied,
        preprocess=preprocess,
        allow_session_global_preprocess=allow_session_global_preprocess,
    )

    x_train, y_train, _feature_cols, _target, sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, estimator)
    metric = scoring_metric or ("r2" if resolved_task == "regression" else "f1_weighted")

    held_out: list[str] = ["test"]
    if split_plan.validation_indices:
        held_out.append("validation")

    train_set = set(split_plan.train_indices)
    if train_set & set(split_plan.test_indices):
        raise LeakageError("Train and test partitions overlap; refusing cross-validation")
    if train_set & set(split_plan.validation_indices):
        raise LeakageError("Train and validation partitions overlap; refusing cross-validation")

    est_params, knob_params = _split_trial_params(params or {})
    if recipe_knobs:
        unknown = sorted(set(recipe_knobs) - SAFE_RECIPE_KNOBS)
        if unknown:
            raise ValidationError(
                f"Unsupported recipe knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
            )
        knob_params = {**knob_params, **dict(recipe_knobs)}
    active_recipe = _recipe_with_knobs(preprocess, knob_params)

    model = clone(estimator)
    if est_params:
        model.set_params(**est_params)
    # Validate weight support once up front (clone keeps the same fit signature).
    fit_kwargs_for_sample_weight(model, sample_weight)

    group_values, strategy_name, splitter, row_order = _resolve_splitter(
        dataset=dataset,
        split_plan=split_plan,
        y_train=y_train,
        cv=cv,
        cv_strategy=cv_strategy,
        groups=groups,
        task=resolved_task,
    )

    x_reset = x_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)
    w_reset = None if sample_weight is None else sample_weight.reset_index(drop=True)
    if row_order is not None:
        x_reset = x_reset.iloc[row_order].reset_index(drop=True)
        y_reset = y_reset.iloc[row_order].reset_index(drop=True)
        if w_reset is not None:
            w_reset = w_reset.iloc[row_order].reset_index(drop=True)
        if group_values is not None:
            group_values = pd.Series(group_values).iloc[row_order]

    group_reset = None if group_values is None else pd.Series(group_values).reset_index(drop=True)

    folds: list[FoldScore] = []
    metric_rows: list[dict[str, float]] = []
    split_iter = (
        splitter.split(x_reset, y_reset, group_reset)
        if group_reset is not None
        else splitter.split(x_reset, y_reset)
    )

    for fold_id, (train_pos, eval_pos) in enumerate(split_iter):
        if set(train_pos) & set(eval_pos):
            raise LeakageError("CV fold train/eval indices overlap")
        x_fold_train = x_reset.iloc[list(train_pos)]
        y_fold_train = y_reset.iloc[list(train_pos)]
        x_fold_eval = x_reset.iloc[list(eval_pos)]
        y_fold_eval = y_reset.iloc[list(eval_pos)]
        w_fold_train = None if w_reset is None else w_reset.iloc[list(train_pos)]
        w_fold_eval = None if w_reset is None else w_reset.iloc[list(eval_pos)]

        if active_recipe is not None and not active_recipe.is_empty():
            prep = build_fold_preprocessor(x_fold_train, active_recipe, y_fold_train)
            x_fit = transform_fold_features(prep, x_fold_train)
            x_score = transform_fold_features(prep, x_fold_eval)
        else:
            x_fit = x_fold_train
            x_score = x_fold_eval

        fold_model = clone(model)
        fold_model.fit(
            x_fit,
            y_fold_train,
            **fit_kwargs_for_sample_weight(fold_model, w_fold_train),
        )
        y_pred = fold_model.predict(x_score)
        fold_metrics = _score_predictions(
            resolved_task, y_fold_eval, y_pred, sample_weight=w_fold_eval
        )
        folds.append(
            FoldScore(
                fold=fold_id,
                n_train=int(len(train_pos)),
                n_eval=int(len(eval_pos)),
                metrics=fold_metrics,
            )
        )
        metric_rows.append(fold_metrics)

    if not metric_rows:
        raise ValidationError("Cross-validation produced no folds")

    mean_metrics, std_metrics = _aggregate_metrics(metric_rows)
    recorded_params = {**est_params, **{f"recipe__{k}": v for k, v in knob_params.items()}}
    session_global_override = bool(session_preprocess_applied and allow_session_global_preprocess)
    return CVScoreResult(
        task=resolved_task,
        scoring_metric=metric,
        cv_strategy=strategy_name,
        n_splits=len(folds),
        folds=folds,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        population="train",
        held_out_partitions=tuple(held_out),
        fold_preprocess=None if active_recipe is None else active_recipe.to_dict(),
        limitations=_cv_limitations(
            session_preprocess_applied=session_global_override,
            preprocess=active_recipe,
            strategy_name=strategy_name,
            n_folds=len(folds),
        ),
        interpretation=_cv_interpretation(
            metric=metric,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            n_folds=len(folds),
            task=resolved_task,
        ),
        recommendations=_cv_recommendations(
            metric=metric,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            held_out=held_out,
            session_preprocess_applied=session_global_override,
            preprocess=active_recipe,
        ),
        params=recorded_params,
    )


def grid_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    param_grid: dict[str, list[Any]] | None = None,
    *,
    recipe_grid: dict[str, list[Any]] | None = None,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Exhaustive grid search with nested fold scoring on the train partition.

    Provide ``param_grid`` (estimator), ``recipe_grid`` (fold-local knobs such
    as ``select_k`` / ``n_bins``), or both. Recipe knobs require ``preprocess``.
    Keys in ``param_grid`` may also use a ``recipe__`` prefix.
    """
    trials = _expand_grid_trials(param_grid=param_grid, recipe_grid=recipe_grid)
    _require_recipe_for_knobs(preprocess, any(t[1] for t in trials))
    return _run_search(
        dataset,
        split_plan,
        estimator,
        trials,
        method="grid",
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )


def randomized_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    param_distributions: dict[str, Any] | None = None,
    *,
    recipe_distributions: dict[str, Any] | None = None,
    n_iter: int = 10,
    random_state: int | None = 42,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Randomized hyperparameter search with nested fold scoring on train.

    Provide ``param_distributions``, ``recipe_distributions``, or both.
    """
    if n_iter < 1:
        raise ValidationError("n_iter must be >= 1")
    trials = _expand_randomized_trials(
        param_distributions=param_distributions,
        recipe_distributions=recipe_distributions,
        n_iter=n_iter,
        random_state=random_state,
    )
    _require_recipe_for_knobs(preprocess, any(t[1] for t in trials))
    return _run_search(
        dataset,
        split_plan,
        estimator,
        trials,
        method="randomized",
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )


def optuna_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    param_space: OptunaSpace | None = None,
    recipe_space: OptunaSpace | None = None,
    n_trials: int = 20,
    random_state: int | None = 42,
    task: TaskType = "auto",
    cv: int | Any = 5,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
    study: Any | None = None,
) -> SearchResult:
    """Optuna-backed search with the same leakage-safe CV contract as grid search.

    Parameters
    ----------
    param_space:
        Either a callable ``trial -> estimator_params`` or a declare-style dict
        (see :func:`_suggest_from_space`). Keys may use a ``recipe__`` prefix.
    recipe_space:
        Optional callable or declare-style dict for fold-local recipe knobs.
        Requires ``preprocess``.
    n_trials:
        Number of Optuna trials.
    ranking_metric:
        Metric maximized (or minimized for loss-like names) via train-fold CV.
    study:
        Optional existing Optuna study to continue (warm-start). When omitted, a
        fresh study is created. Trial objectives still score only the current
        ``split_plan`` train partition via inner CV — never Session test rows.

    Notes
    -----
    Requires ``pip install 'buildml[optuna]'``. Folds stay inside the Session
    train partition; Session test/validation never enter trial scoring.
    """
    try:
        import optuna
    except ImportError as exc:
        from buildml.core.errors import MissingExtraError

        raise MissingExtraError("optuna", "Optuna hyperparameter search") from exc

    if n_trials < 1:
        raise ValidationError("n_trials must be >= 1")
    if param_space is None and recipe_space is None:
        raise ValidationError("Provide param_space and/or recipe_space")
    _require_recipe_for_knobs(preprocess, recipe_space is not None)

    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _x_train, y_train, _feature_cols, _target, _sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, estimator)
    metric_name = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    higher_is_better = metric_name not in _LOWER_IS_BETTER
    direction = "maximize" if higher_is_better else "minimize"

    # Keep Optuna console quiet; BuildML returns a SearchResult card instead.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if study is None:
        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)
    else:
        study_direction = getattr(study, "direction", None)
        actual = getattr(study_direction, "name", str(study_direction))
        if direction.upper() not in str(actual).upper():
            raise ValidationError(
                f"warm-start Optuna study direction {actual!r} does not match "
                f"required {direction.upper()!r} for metric {metric_name!r}"
            )
    trial_rows: list[SearchTrial] = []

    def _objective(trial: Any) -> float:
        est_params: dict[str, Any] = {}
        recipe_knobs: dict[str, Any] = {}
        if param_space is not None:
            raw = (
                param_space(trial)
                if callable(param_space)
                else _suggest_from_space(trial, param_space, prefix="param")
            )
            est_params, embedded = _split_trial_params(dict(raw))
            recipe_knobs.update(embedded)
        if recipe_space is not None:
            from_recipe = (
                dict(recipe_space(trial))
                if callable(recipe_space)
                else _suggest_from_space(trial, recipe_space, prefix="recipe")
            )
            unknown = sorted(set(from_recipe) - SAFE_RECIPE_KNOBS)
            if unknown:
                raise ValidationError(
                    f"Unsupported recipe knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
                )
            recipe_knobs.update(from_recipe)

        cv_result = cv_score(
            dataset,
            split_plan,
            estimator,
            task=resolved_task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=metric_name,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            params=est_params,
            recipe_knobs=recipe_knobs,
        )
        score = float(cv_result.mean_metrics[metric_name])
        trial_rows.append(
            SearchTrial(
                trial=int(trial.number),
                params=dict(est_params),
                recipe_knobs=dict(recipe_knobs),
                mean_score=score,
                std_score=float(cv_result.std_metrics.get(metric_name, float("nan"))),
                mean_metrics=dict(cv_result.mean_metrics),
                std_metrics=dict(cv_result.std_metrics),
                cv=cv_result,
            )
        )
        return score

    study.optimize(_objective, n_trials=n_trials)

    if not trial_rows:
        raise ValidationError("Optuna search produced no trials")

    trials = sorted(trial_rows, key=lambda item: item.mean_score, reverse=higher_is_better)
    # Re-number display order after ranking while keeping Optuna trial ids in params lineage.
    ranked = [
        SearchTrial(
            trial=i,
            params=dict(t.params),
            recipe_knobs=dict(t.recipe_knobs),
            mean_score=t.mean_score,
            std_score=t.std_score,
            mean_metrics=dict(t.mean_metrics),
            std_metrics=dict(t.std_metrics),
            cv=t.cv,
        )
        for i, t in enumerate(trials)
    ]
    return _finalize_search_result(
        method="optuna",
        resolved_task=resolved_task,
        metric_name=metric_name,
        trials=ranked,
        estimator=estimator,
        dataset=dataset,
        split_plan=split_plan,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
        study=study,
    )


def nested_cv_score(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    *,
    param_grid: dict[str, list[Any]] | None = None,
    param_distributions: dict[str, Any] | None = None,
    recipe_grid: dict[str, list[Any]] | None = None,
    recipe_distributions: dict[str, Any] | None = None,
    param_space: OptunaSpace | None = None,
    recipe_space: OptunaSpace | None = None,
    inner_search: InnerSearchMethod = "auto",
    n_iter: int = 10,
    n_trials: int = 20,
    random_state: int | None = 42,
    task: TaskType = "auto",
    outer_cv: int | Any = 5,
    inner_cv: int | Any = 3,
    cv_strategy: CvStrategy = "auto",
    scoring_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    warm_start_studies: bool = False,
) -> NestedCVResult:
    """Honest outer estimate after inner-loop hyperparameter / recipe-knob search.

    Parameters
    ----------
    param_grid / param_distributions:
        Estimator search space for grid / randomized inner search. At most one
        may be set. May be omitted when a recipe space is provided.
    recipe_grid / recipe_distributions:
        Fold-local recipe knob space (``select_k``, ``n_bins``, …). At most one
        may be set. Requires ``preprocess``.
    param_space / recipe_space:
        Optuna declare-style or callable spaces for ``inner_search='optuna'``.
        Require ``pip install 'buildml[optuna]'``.
    inner_search:
        ``auto`` (infer from provided spaces), ``grid``, ``randomized``, or
        ``optuna``.
    n_iter:
        Randomized inner trials when using distributions.
    n_trials:
        Optuna inner trials when ``inner_search`` resolves to ``optuna``.
    outer_cv / inner_cv:
        Outer and inner fold counts (or sklearn splitters).
    cv_strategy:
        Shared fold builder for integer outer/inner CV when roles allow.
    preprocess:
        Fold-local :class:`PreprocessRecipe` used in both loops. Required when
        searching recipe knobs.
    allow_session_global_preprocess:
        Explicit opt-in when Session-global preprocess already ran and no
        fold-local recipe is provided (default False; refuses otherwise).
    warm_start_studies:
        Default False. When True with Optuna inner search, reuse one Optuna
        study across outer folds so later folds inherit prior trial history
        (TPE priors). See Notes for the leakage-audit policy.

    Notes
    -----
    **Leakage:** Outer-eval rows never enter inner CV membership or inner
    ranking. Session test/validation partitions are never used for outer or
    inner folds. Chosen recipe knobs are recorded per outer fold. Optuna trials
    run only on each outer-train subset via :func:`optuna_search`.

    **Warm-start policy (``warm_start_studies=True``):** Shared study state
    carries only prior *inner*-CV trial scores from earlier outer-train
    subsets. Outer-eval rows and Session test/validation still never enter
    trial objectives. This can couple search trajectories across outer folds
    (mild optimism vs independent studies) and is therefore opt-in.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    search_method = _resolve_inner_search(
        inner_search=inner_search,
        param_grid=param_grid,
        param_distributions=param_distributions,
        recipe_grid=recipe_grid,
        recipe_distributions=recipe_distributions,
        param_space=param_space,
        recipe_space=recipe_space,
    )
    has_est = param_grid is not None or param_distributions is not None or param_space is not None
    has_recipe = (
        recipe_grid is not None or recipe_distributions is not None or recipe_space is not None
    )
    if not has_est and not has_recipe:
        raise ValidationError(
            "nested_cv_score requires an estimator and/or recipe search space "
            "(param_grid/param_distributions/param_space and/or "
            "recipe_grid/recipe_distributions/recipe_space)"
        )
    if param_grid is not None and param_distributions is not None:
        raise ValidationError(
            "nested_cv_score accepts at most one of param_grid or param_distributions"
        )
    if recipe_grid is not None and recipe_distributions is not None:
        raise ValidationError(
            "nested_cv_score accepts at most one of recipe_grid or recipe_distributions"
        )
    if search_method == "optuna":
        if param_space is None and recipe_space is None:
            raise ValidationError("inner_search='optuna' requires param_space and/or recipe_space")
        if param_grid is not None or param_distributions is not None:
            raise ValidationError(
                "inner_search='optuna' uses param_space; omit param_grid/param_distributions"
            )
        if recipe_grid is not None or recipe_distributions is not None:
            raise ValidationError(
                "inner_search='optuna' uses recipe_space; omit recipe_grid/recipe_distributions"
            )
    if warm_start_studies and search_method != "optuna":
        raise ValidationError(
            "warm_start_studies=True requires Optuna inner search "
            "(inner_search='optuna' or auto with param_space/recipe_space)"
        )
        if n_trials < 1:
            raise ValidationError("n_trials must be >= 1")
    if param_grid is not None and not param_grid:
        raise ValidationError("param_grid must not be empty when provided")
    if param_distributions is not None and not param_distributions:
        raise ValidationError("param_distributions must not be empty when provided")
    if recipe_grid is not None and not recipe_grid:
        raise ValidationError("recipe_grid must not be empty when provided")
    if recipe_distributions is not None and not recipe_distributions:
        raise ValidationError("recipe_distributions must not be empty when provided")
    _require_recipe_for_knobs(preprocess, has_recipe)
    _refuse_session_global_cv_leakage(
        session_preprocess_applied=session_preprocess_applied,
        preprocess=preprocess,
        allow_session_global_preprocess=allow_session_global_preprocess,
    )

    x_train, y_train, _feature_cols, _target, _sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, estimator)
    metric = scoring_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    weight_col = weight_column(dataset)

    held_out: list[str] = ["test"]
    if split_plan.validation_indices:
        held_out.append("validation")

    train_set = set(split_plan.train_indices)
    session_test = set(split_plan.test_indices)
    session_valid = set(split_plan.validation_indices)
    if train_set & session_test:
        raise LeakageError("Train and test partitions overlap; refusing nested CV")
    if train_set & session_valid:
        raise LeakageError("Train and validation partitions overlap; refusing nested CV")

    train_positions = list(split_plan.train_indices)
    group_values, outer_strategy, outer_splitter, row_order = _resolve_splitter(
        dataset=dataset,
        split_plan=split_plan,
        y_train=y_train,
        cv=outer_cv,
        cv_strategy=cv_strategy,
        groups=groups,
        task=resolved_task,
    )

    x_reset = x_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)
    position_map = np.asarray(train_positions, dtype=int)
    if row_order is not None:
        x_reset = x_reset.iloc[row_order].reset_index(drop=True)
        y_reset = y_reset.iloc[row_order].reset_index(drop=True)
        position_map = position_map[row_order]
        if group_values is not None:
            group_values = pd.Series(group_values).iloc[row_order]

    group_reset = None if group_values is None else pd.Series(group_values).reset_index(drop=True)
    split_iter = (
        outer_splitter.split(x_reset, y_reset, group_reset)
        if group_reset is not None
        else outer_splitter.split(x_reset, y_reset)
    )

    outer_folds: list[OuterFoldResult] = []
    metric_rows: list[dict[str, float]] = []
    selected_params: list[dict[str, Any]] = []
    selected_recipe_knobs: list[dict[str, Any]] = []
    inner_strategy_name = outer_strategy
    inner_n_splits = int(inner_cv) if isinstance(inner_cv, int) else 0
    shared_study: Any | None = None

    for fold_id, (outer_train_pos, outer_eval_pos) in enumerate(split_iter):
        if set(outer_train_pos) & set(outer_eval_pos):
            raise LeakageError("Nested CV outer fold train/eval indices overlap")
        outer_train_idx = tuple(int(i) for i in position_map[list(outer_train_pos)])
        outer_eval_idx = tuple(int(i) for i in position_map[list(outer_eval_pos)])
        if set(outer_train_idx) & session_test or set(outer_eval_idx) & session_test:
            raise LeakageError("Nested CV outer fold intersected Session test partition")
        if set(outer_train_idx) & session_valid or set(outer_eval_idx) & session_valid:
            raise LeakageError("Nested CV outer fold intersected Session validation partition")

        inner_plan = SplitPlan(
            kind="nested_inner",
            test_size=None,
            validation_size=None,
            random_state=None,
            stratify_column=None,
            train_indices=outer_train_idx,
            validation_indices=(),
            test_indices=outer_eval_idx,
        )
        # Explicit groups are aligned to Session train-row order; subset to outer-train.
        inner_groups = None
        if groups is not None:
            aligned = pd.Series(groups).reset_index(drop=True)
            if len(aligned) != len(split_plan.train_indices):
                raise ValidationError("groups length must match the train partition")
            train_pos_lookup = {int(idx): i for i, idx in enumerate(split_plan.train_indices)}
            inner_groups = pd.Series(
                [aligned.iloc[train_pos_lookup[idx]] for idx in outer_train_idx],
                dtype=aligned.dtype,
            )

        if search_method == "grid":
            fold_search = grid_search(
                dataset,
                inner_plan,
                estimator,
                param_grid,
                recipe_grid=recipe_grid,
                task=resolved_task,
                cv=inner_cv,
                cv_strategy=cv_strategy,
                ranking_metric=metric,
                groups=inner_groups,
                preprocess=preprocess,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
                refit=False,
            )
        elif search_method == "optuna":
            fold_search = optuna_search(
                dataset,
                inner_plan,
                estimator,
                param_space=param_space,
                recipe_space=recipe_space,
                n_trials=n_trials,
                random_state=None if random_state is None else int(random_state) + fold_id,
                task=resolved_task,
                cv=inner_cv,
                cv_strategy=cv_strategy,
                ranking_metric=metric,
                groups=inner_groups,
                preprocess=preprocess,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
                refit=False,
                study=shared_study if warm_start_studies else None,
            )
            if warm_start_studies:
                shared_study = fold_search.study
        else:
            fold_search = randomized_search(
                dataset,
                inner_plan,
                estimator,
                param_distributions,
                recipe_distributions=recipe_distributions,
                n_iter=n_iter,
                random_state=None if random_state is None else int(random_state) + fold_id,
                task=resolved_task,
                cv=inner_cv,
                cv_strategy=cv_strategy,
                ranking_metric=metric,
                groups=inner_groups,
                preprocess=preprocess,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
                refit=False,
            )
        if fold_search.best_cv is not None:
            inner_strategy_name = fold_search.best_cv.cv_strategy
            inner_n_splits = fold_search.best_cv.n_splits

        # Refit winner on outer-train only; score outer-eval.
        model = clone(estimator)
        if fold_search.best_params:
            model.set_params(**fold_search.best_params)
        active_recipe = _recipe_with_knobs(preprocess, fold_search.best_recipe_knobs)
        base = dataset._ensure_pandas()
        x_outer_train = base.iloc[list(outer_train_idx)][list(_feature_cols)]
        y_outer_train = base.iloc[list(outer_train_idx)][_target]
        x_outer_eval = base.iloc[list(outer_eval_idx)][list(_feature_cols)]
        y_outer_eval = base.iloc[list(outer_eval_idx)][_target]
        w_outer_train = None
        w_outer_eval = None
        if weight_col is not None:
            w_outer_train = validate_sample_weights(
                base.iloc[list(outer_train_idx)][weight_col], column=weight_col
            )
            w_outer_eval = validate_sample_weights(
                base.iloc[list(outer_eval_idx)][weight_col], column=weight_col
            )
        if active_recipe is not None and not active_recipe.is_empty():
            prep = build_fold_preprocessor(x_outer_train, active_recipe, y_outer_train)
            x_fit = transform_fold_features(prep, x_outer_train)
            x_score = transform_fold_features(prep, x_outer_eval)
        else:
            x_fit = x_outer_train
            x_score = x_outer_eval
        model.fit(
            x_fit,
            y_outer_train,
            **fit_kwargs_for_sample_weight(model, w_outer_train),
        )
        y_pred = model.predict(x_score)
        fold_metrics = _score_predictions(
            resolved_task, y_outer_eval, y_pred, sample_weight=w_outer_eval
        )
        outer_folds.append(
            OuterFoldResult(
                fold=fold_id,
                n_train=int(len(outer_train_idx)),
                n_eval=int(len(outer_eval_idx)),
                best_params=dict(fold_search.best_params),
                best_recipe_knobs=dict(fold_search.best_recipe_knobs),
                inner_best_score=fold_search.best_score,
                inner_best_std=fold_search.best_std,
                inner_n_trials=len(fold_search.trials),
                metrics=fold_metrics,
            )
        )
        metric_rows.append(fold_metrics)
        selected_params.append(dict(fold_search.best_params))
        selected_recipe_knobs.append(dict(fold_search.best_recipe_knobs))

    if not metric_rows:
        raise ValidationError("Nested cross-validation produced no outer folds")

    mean_metrics, std_metrics = _aggregate_metrics(metric_rows)
    summary = _inner_selection_summary(
        selected_params, outer_folds, metric, selected_recipe_knobs=selected_recipe_knobs
    )
    session_global_override = bool(session_preprocess_applied and allow_session_global_preprocess)
    limitations = _nested_limitations(
        session_preprocess_applied=session_global_override,
        preprocess=preprocess,
        outer_strategy=outer_strategy,
        inner_strategy=inner_strategy_name,
        n_outer=len(outer_folds),
        n_inner=inner_n_splits,
        held_out=held_out,
    )
    if any(selected_recipe_knobs):
        limitations.append(
            "Inner search chose fold-local recipe knobs "
            f"({sorted({k for knobs in selected_recipe_knobs for k in knobs})}); "
            "outer-eval rows never contributed to those choices."
        )
    if warm_start_studies:
        limitations.append(
            "warm_start_studies=True shared one Optuna study across outer folds. "
            "Trial objectives still scored only each outer-train subset via inner "
            "CV (no Session test/validation or outer-eval peeking), but search "
            "priors couple outer folds and can be mildly optimistic versus "
            "independent studies."
        )
    interpretation = [
        (
            f"Outer-loop mean {metric}={mean_metrics[metric]:.6f} ± "
            f"{std_metrics.get(metric, 0.0):.6f} across {len(outer_folds)} folds "
            f"after inner {search_method} selection ({resolved_task})."
        )
    ]
    if summary.get("param_stability") == "low":
        interpretation.append(
            "Inner search selected different parameter sets across outer folds — "
            "treat a single full-train refit as one of several plausible winners."
        )
    recommendations = [
        (
            f"Report the outer mean±std of '{metric}' as the post-selection estimate; "
            "do not substitute inner CV means."
        ),
        (
            "After nested CV, refit the chosen recipe on full train and confirm "
            f"once on {held_out[0]}."
        ),
    ]
    if session_global_override:
        recommendations.append(
            "allow_session_global_preprocess=True was set; Session-global preprocess "
            "poisoned folds. Re-ingest unpoisoned data before fold-local "
            "PreprocessRecipe CV next time."
        )
    elif preprocess is not None and not preprocess.is_empty():
        recommendations.append(
            "Fold-local preprocess was refit inside outer and inner loops on their "
            "respective training rows only."
        )
    if any(selected_recipe_knobs):
        recommendations.append(
            "Inspect outer_folds[*].best_recipe_knobs and "
            "inner_selection_summary.selected_recipe_knobs_by_fold before freezing "
            "a single production recipe."
        )

    return NestedCVResult(
        task=resolved_task,
        scoring_metric=metric,
        outer_cv_strategy=outer_strategy,
        inner_cv_strategy=inner_strategy_name,
        n_outer_splits=len(outer_folds),
        n_inner_splits=inner_n_splits,
        search_method=search_method,
        outer_folds=outer_folds,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        inner_selection_summary=summary,
        population="train",
        held_out_partitions=tuple(held_out),
        fold_preprocess=None if preprocess is None else preprocess.to_dict(),
        limitations=limitations,
        interpretation=interpretation,
        recommendations=recommendations,
        warm_start_studies=bool(warm_start_studies),
    )


def _inner_selection_summary(
    selected_params: list[dict[str, Any]],
    outer_folds: list[OuterFoldResult],
    metric: str,
    *,
    selected_recipe_knobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from collections import Counter

    recipe_knobs = selected_recipe_knobs or [dict(f.best_recipe_knobs) for f in outer_folds]
    frozen: list[tuple[tuple[str, str], ...]] = []
    for params, knobs in zip(selected_params, recipe_knobs, strict=True):
        merged = dict(params)
        merged.update({f"recipe__{rk}": rv for rk, rv in knobs.items()})
        frozen.append(tuple(sorted((k, repr(v)) for k, v in merged.items())))
    counts = Counter(frozen)
    unique = len(counts)
    top = counts.most_common(1)[0] if counts else ((), 0)
    if unique == 1:
        stability = "high"
    elif unique <= max(2, len(frozen) // 2):
        stability = "medium"
    else:
        stability = "low"
    inner_means = [f.inner_best_score for f in outer_folds if f.inner_best_score is not None]
    return {
        "n_outer_folds": len(outer_folds),
        "n_unique_param_sets": unique,
        "most_common_params": dict(selected_params[0]) if selected_params and unique == 1 else None,
        "most_common_recipe_knobs": (
            dict(recipe_knobs[0]) if recipe_knobs and unique == 1 else None
        ),
        "most_common_count": int(top[1]),
        "param_stability": stability,
        "inner_best_score_mean": float(np.mean(inner_means)) if inner_means else None,
        "inner_best_score_std": (
            float(np.std(inner_means, ddof=1)) if len(inner_means) > 1 else 0.0
        ),
        "outer_metric": metric,
        "selected_params_by_fold": [dict(p) for p in selected_params],
        "selected_recipe_knobs_by_fold": [dict(k) for k in recipe_knobs],
    }


def _nested_limitations(
    *,
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
    outer_strategy: str,
    inner_strategy: str,
    n_outer: int,
    n_inner: int,
    held_out: list[str],
) -> list[str]:
    tips = [
        (
            f"Outer scores summarize {n_outer} folds drawn only from the train partition; "
            f"inner search used {n_inner}-fold CV on each outer-train subset."
        ),
        (
            "Outer-eval rows never enter inner CV membership or inner ranking "
            f"(outer={outer_strategy}, inner={inner_strategy})."
        ),
        (f"Session held-out partition(s) stay untouched during nested CV: {', '.join(held_out)}."),
        (
            "Inner CV means are selection evidence only; the outer mean±std is the "
            "post-selection estimate."
        ),
    ]
    tips.extend(
        _cv_limitations(
            session_preprocess_applied=session_preprocess_applied,
            preprocess=preprocess,
            strategy_name=outer_strategy,
            n_folds=n_outer,
        )
    )
    return tips


def _run_search(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimator: Any,
    combos: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    method: SearchMethod,
    task: TaskType,
    cv: int | Any,
    cv_strategy: CvStrategy,
    ranking_metric: str | None,
    groups: pd.Series | None,
    preprocess: PreprocessRecipe | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
    refit: bool,
) -> SearchResult:
    trials: list[SearchTrial] = []
    resolved_task: Literal["classification", "regression"] | None = None
    metric_name: str | None = ranking_metric

    for idx, (est_params, recipe_knobs) in enumerate(combos):
        cv_result = cv_score(
            dataset,
            split_plan,
            estimator,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=metric_name,
            groups=groups,
            preprocess=preprocess,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            params=est_params,
            recipe_knobs=recipe_knobs,
        )
        resolved_task = cv_result.task
        metric_name = cv_result.scoring_metric
        score = cv_result.mean_metrics[metric_name]
        trials.append(
            SearchTrial(
                trial=idx,
                params=dict(est_params),
                recipe_knobs=dict(recipe_knobs),
                mean_score=score,
                std_score=cv_result.std_metrics.get(metric_name, float("nan")),
                mean_metrics=dict(cv_result.mean_metrics),
                std_metrics=dict(cv_result.std_metrics),
                cv=cv_result,
            )
        )

    assert resolved_task is not None and metric_name is not None
    higher_is_better = metric_name not in _LOWER_IS_BETTER
    trials.sort(key=lambda item: item.mean_score, reverse=higher_is_better)
    return _finalize_search_result(
        method=method,
        resolved_task=resolved_task,
        metric_name=metric_name,
        trials=trials,
        estimator=estimator,
        dataset=dataset,
        split_plan=split_plan,
        preprocess=preprocess,
        session_preprocess_applied=session_preprocess_applied,
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )


def _finalize_search_result(
    *,
    method: SearchMethod,
    resolved_task: Literal["classification", "regression"],
    metric_name: str,
    trials: list[SearchTrial],
    estimator: Any,
    dataset: Dataset,
    split_plan: SplitPlan | None,
    preprocess: PreprocessRecipe | None,
    session_preprocess_applied: bool,
    refit: bool,
    allow_session_global_preprocess: bool = False,
    study: Any | None = None,
) -> SearchResult:
    best = trials[0]
    session_global_override = bool(session_preprocess_applied and allow_session_global_preprocess)

    refit_result = None
    if refit:
        model = clone(estimator)
        if best.params:
            model.set_params(**best.params)
        active_recipe = _recipe_with_knobs(preprocess, best.recipe_knobs)
        if active_recipe is not None and not active_recipe.is_empty():
            x_train, y_train, feature_cols, target, sample_weight = _feature_target_frames(
                dataset,
                split_plan,
                "train",  # type: ignore[arg-type]
            )
            prep = build_fold_preprocessor(x_train, active_recipe, y_train)
            x_fit = transform_fold_features(prep, x_train)
            fitted = clone(model)
            fitted.fit(
                x_fit,
                y_train,
                **fit_kwargs_for_sample_weight(fitted, sample_weight),
            )
            bundled = SkPipeline([("preprocess", prep), ("model", fitted)])
            refit_result = FitResult(
                estimator=bundled,
                task=resolved_task,
                feature_columns=tuple(feature_cols),
                target_column=target,
                n_train_rows=int(len(x_train)),
                weight_column=weight_column(dataset),
            )
        else:
            refit_result = fit_estimator(dataset, split_plan, model, task=resolved_task)

    interpretation = [
        (
            f"Best {metric_name} over {len(trials)} {method} trial(s): "
            f"{best.mean_score:.6f} ± {best.std_score:.6f} on train-fold CV."
        )
    ]
    if best.recipe_knobs:
        interpretation.append(f"Best recipe knobs: {best.recipe_knobs}.")
    if len(trials) >= 2:
        gap = abs(trials[0].mean_score - trials[1].mean_score)
        interpretation.append(
            f"Top-2 mean {metric_name} gap is {gap:.6f}; "
            f"second-best std is {trials[1].std_score:.6f}."
        )
        if gap < max(trials[0].std_score, 1e-12):
            interpretation.append(
                "Top-2 gap is within the leading trial's fold standard deviation — "
                "treat rank as fragile without a confirmation holdout."
            )

    held = list(best.cv.held_out_partitions) if best.cv is not None else ["test"]
    recommendations = [
        f"Selected params by mean {metric_name} across CV folds on the train population.",
        f"Confirm the winner once on {held[0]} after search.",
    ]
    if session_global_override:
        recommendations.append(
            "allow_session_global_preprocess=True was set; Session preprocess was "
            "train-global. Re-ingest unpoisoned data, then use fold-local "
            "PreprocessRecipe without Session.impute/scale before search."
        )
    if best.std_score > abs(best.mean_score) * 0.15 and abs(best.mean_score) > 1e-9:
        recommendations.append(
            "Fold spread is large relative to the mean — prefer simpler params or more data."
        )
    if method == "optuna":
        recommendations.append(
            "Optuna TPE sampled this budget; raise n_trials only while fold std still "
            "informs whether gaps are real."
        )

    limitations = list(best.cv.limitations) if best.cv is not None else []
    limitations.append(
        "Search ranks configurations by nested train-fold CV; it does not peek at held-out "
        f"partition(s): {', '.join(held)}."
    )

    return SearchResult(
        method=method,
        task=resolved_task,
        ranking_metric=metric_name,
        trials=trials,
        best_params=dict(best.params),
        best_recipe_knobs=dict(best.recipe_knobs),
        best_score=best.mean_score,
        best_std=best.std_score,
        best_cv=best.cv,
        refit_result=refit_result,
        interpretation=interpretation,
        recommendations=recommendations,
        limitations=limitations,
        study=study,
    )


def _suggest_from_space(
    trial: Any,
    space: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Suggest values from a declare-style Optuna space mapping.

    Supported value forms:

    - ``{"type": "float", "low": ..., "high": ..., "log": bool}``
    - ``{"type": "int", "low": ..., "high": ...}``
    - ``{"type": "categorical", "choices": [...]}``
    - plain list/tuple → categorical choices
    """
    out: dict[str, Any] = {}
    for name, spec in space.items():
        key = f"{prefix}__{name}"
        if isinstance(spec, (list, tuple)):
            out[name] = trial.suggest_categorical(key, list(spec))
            continue
        if not isinstance(spec, dict):
            raise ValidationError(
                f"Optuna space entry '{name}' must be a dict spec or a list of choices"
            )
        kind = str(spec.get("type", "")).lower()
        if kind == "float":
            out[name] = trial.suggest_float(
                key,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "int":
            out[name] = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
        elif kind == "categorical":
            choices = list(spec.get("choices") or [])
            if not choices:
                raise ValidationError(f"Optuna categorical space '{name}' needs non-empty choices")
            out[name] = trial.suggest_categorical(key, choices)
        else:
            raise ValidationError(
                f"Unsupported Optuna space type '{kind}' for '{name}'. "
                "Use float, int, or categorical."
            )
    return out


def _split_trial_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a flat trial dict into estimator params and recipe knobs."""
    est: dict[str, Any] = {}
    recipe: dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith("recipe__"):
            recipe[key[len("recipe__") :]] = value
        elif key in SAFE_RECIPE_KNOBS:
            recipe[key] = value
        else:
            est[key] = value
    unknown = sorted(set(recipe) - SAFE_RECIPE_KNOBS)
    if unknown:
        raise ValidationError(
            f"Unsupported recipe knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
        )
    return est, recipe


def _recipe_with_knobs(
    preprocess: PreprocessRecipe | None,
    knobs: dict[str, Any] | None,
) -> PreprocessRecipe | None:
    if preprocess is None:
        if knobs:
            raise ValidationError("recipe knobs require a base PreprocessRecipe")
        return None
    if not knobs:
        return preprocess
    return preprocess.with_knobs(knobs)


def _resolve_inner_search(
    *,
    inner_search: InnerSearchMethod,
    param_grid: dict[str, list[Any]] | None,
    param_distributions: dict[str, Any] | None,
    recipe_grid: dict[str, list[Any]] | None,
    recipe_distributions: dict[str, Any] | None,
    param_space: OptunaSpace | None,
    recipe_space: OptunaSpace | None,
) -> SearchMethod:
    """Resolve nested-CV inner search method from explicit choice or spaces."""
    has_optuna = param_space is not None or recipe_space is not None
    has_grid = param_grid is not None or recipe_grid is not None
    has_random = param_distributions is not None or recipe_distributions is not None

    if inner_search == "optuna":
        return "optuna"
    if inner_search == "grid":
        if has_optuna:
            raise ValidationError(
                "inner_search='grid' cannot be combined with param_space/recipe_space"
            )
        if has_random and not has_grid:
            raise ValidationError("inner_search='grid' requires param_grid and/or recipe_grid")
        return "grid"
    if inner_search == "randomized":
        if has_optuna:
            raise ValidationError(
                "inner_search='randomized' cannot be combined with param_space/recipe_space"
            )
        return "randomized"
    # auto
    if has_optuna:
        if has_grid or has_random:
            raise ValidationError(
                "Provide either Optuna spaces (param_space/recipe_space) or "
                "grid/randomized spaces, not both; or set inner_search explicitly"
            )
        return "optuna"
    if has_grid and not has_random:
        return "grid"
    if has_random:
        return "randomized"
    if has_grid:
        return "grid"
    raise ValidationError(
        "nested_cv_score requires an estimator and/or recipe search space "
        "(param_grid/param_distributions/param_space and/or "
        "recipe_grid/recipe_distributions/recipe_space)"
    )


def _require_recipe_for_knobs(preprocess: PreprocessRecipe | None, needs_recipe: bool) -> None:
    if needs_recipe and preprocess is None:
        raise ValidationError(
            "recipe_grid/recipe_distributions/recipe_space require preprocess=PreprocessRecipe(...)"
        )


def _grid_dicts(space: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not space:
        return [{}]
    keys = list(space)
    return [dict(zip(keys, values, strict=True)) for values in product(*(space[k] for k in keys))]


def _expand_grid_trials(
    *,
    param_grid: dict[str, list[Any]] | None,
    recipe_grid: dict[str, list[Any]] | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if (param_grid is None or not param_grid) and (recipe_grid is None or not recipe_grid):
        raise ValidationError("Provide a non-empty param_grid and/or recipe_grid")
    if recipe_grid:
        unknown = sorted(set(recipe_grid) - SAFE_RECIPE_KNOBS)
        if unknown:
            raise ValidationError(
                f"Unsupported recipe_grid knobs: {unknown}. Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
            )
    est_raw = _grid_dicts(param_grid)
    # Allow recipe__ keys inside param_grid for convenience.
    est_combos: list[dict[str, Any]] = []
    recipe_from_params: list[dict[str, Any]] = []
    for raw in est_raw:
        est, recipe = _split_trial_params(raw)
        est_combos.append(est)
        recipe_from_params.append(recipe)
    recipe_combos = _grid_dicts(recipe_grid)
    trials: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for est, embedded in zip(est_combos, recipe_from_params, strict=True):
        for recipe in recipe_combos:
            merged = {**embedded, **recipe}
            trials.append((dict(est), merged))
    return trials


def _expand_randomized_trials(
    *,
    param_distributions: dict[str, Any] | None,
    recipe_distributions: dict[str, Any] | None,
    n_iter: int,
    random_state: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if (param_distributions is None or not param_distributions) and (
        recipe_distributions is None or not recipe_distributions
    ):
        raise ValidationError("Provide a non-empty param_distributions and/or recipe_distributions")
    if recipe_distributions:
        unknown = sorted(set(recipe_distributions) - SAFE_RECIPE_KNOBS)
        if unknown:
            raise ValidationError(
                f"Unsupported recipe_distributions knobs: {unknown}. "
                f"Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
            )
    # Sample a joint space so n_iter bounds total trials.
    joint: dict[str, Any] = {}
    if param_distributions:
        joint.update(param_distributions)
    if recipe_distributions:
        for key, values in recipe_distributions.items():
            joint[f"recipe__{key}"] = values
    sampler = ParameterSampler(joint, n_iter=n_iter, random_state=random_state)
    return [_split_trial_params(dict(params)) for params in sampler]


def _resolve_splitter(
    *,
    dataset: Dataset,
    split_plan: SplitPlan,
    y_train: pd.Series,
    cv: int | Any,
    cv_strategy: CvStrategy,
    groups: pd.Series | None,
    task: Literal["classification", "regression"],
) -> tuple[pd.Series | None, str, Any, np.ndarray | None]:
    if not isinstance(cv, int):
        splitter = check_cv(cv, y=y_train, classifier=task == "classification")
        return groups, type(splitter).__name__, splitter, None

    if cv < 2:
        raise ValidationError("cv must be an integer >= 2 or a CV splitter")

    strategy = cv_strategy
    if strategy == "auto":
        if groups is not None or dataset.role_columns(ColumnRole.GROUP):
            strategy = "stratified_group" if task == "classification" else "group"
        elif dataset.role_columns(ColumnRole.TIME):
            strategy = "time"
        elif task == "classification":
            strategy = "stratified"
        else:
            strategy = "kfold"

    group_values = groups
    if strategy in {"group", "stratified_group"}:
        if group_values is None:
            group_cols = dataset.role_columns(ColumnRole.GROUP)
            if not group_cols:
                raise ValidationError(
                    "Group CV requires a column with role 'group' or an explicit groups series"
                )
            if len(group_cols) != 1:
                raise ValidationError("Group CV expects exactly one group-role column")
            train_frame = frame_for_partition(dataset, split_plan, "train")
            group_values = train_frame[group_cols[0]]
        n_groups = int(pd.Series(group_values).nunique(dropna=False))
        if n_groups < cv:
            raise ValidationError(
                f"Need at least {cv} distinct groups for {cv}-fold group CV; found {n_groups}"
            )
        if strategy == "stratified_group":
            if task != "classification":
                raise ValidationError("stratified_group CV is only valid for classification")
            return (
                group_values,
                "stratified_group",
                StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42),
                None,
            )
        return group_values, "group", GroupKFold(n_splits=cv), None

    if strategy == "time":
        time_cols = dataset.role_columns(ColumnRole.TIME)
        if not time_cols:
            raise ValidationError("Time CV requires a column with role 'time'")
        train_frame = frame_for_partition(dataset, split_plan, "train")
        stamps = pd.to_datetime(train_frame[time_cols[0]], errors="coerce")
        if stamps.isna().any():
            raise ValidationError("Time CV requires parseable values in the time-role column")
        order = np.argsort(stamps.to_numpy(), kind="mergesort")
        return None, "time", TimeSeriesSplit(n_splits=cv), order

    if strategy == "stratified":
        if task != "classification":
            raise ValidationError("stratified CV is only valid for classification")
        return None, "stratified", StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), None

    if strategy == "kfold":
        return None, "kfold", KFold(n_splits=cv, shuffle=True, random_state=42), None

    raise ValidationError(f"Unknown cv_strategy '{cv_strategy}'")


def _score_predictions(
    task: Literal["classification", "regression"],
    y_true: pd.Series,
    y_pred: Any,
    *,
    sample_weight: pd.Series | None = None,
) -> dict[str, float]:
    sw = None if sample_weight is None else sample_weight.to_numpy(dtype=float)
    if task == "regression":
        mse = float(mean_squared_error(y_true, y_pred, sample_weight=sw))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sw)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred, sample_weight=sw)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sw)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred, sample_weight=sw)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw)
        ),
    }


def _aggregate_metrics(
    rows: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        mean_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_metrics, std_metrics


def _cv_limitations(
    *,
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
    strategy_name: str,
    n_folds: int,
) -> list[str]:
    tips = [
        f"Scores summarize {n_folds} folds drawn only from the train partition.",
        "The Session test partition is not used for fold membership or fold scoring.",
    ]
    if session_preprocess_applied:
        tips.append(
            "allow_session_global_preprocess=True: Session-global preprocess plans "
            "(impute/encode/scale/outliers/binning/feature_select/dates/text/reduce/"
            "resample) were fitted on the full train partition before CV, so "
            "fold-eval rows influenced those frozen statistics."
        )
        tips.append(
            "Session-global target encoding uses out-of-fold values on train, but still "
            "freezes full-train category maps before CV; prefer fold-local "
            "PreprocessRecipe(encode='target') on unpoisoned data when selection itself "
            "uses CV."
        )
        if preprocess is not None and not preprocess.is_empty():
            tips.append(
                "A fold-local PreprocessRecipe was also provided, but Session data was "
                "already transformed with train-global statistics — the recipe does not "
                "rebuild from raw/unpoisoned rows."
            )
    if preprocess is not None and not preprocess.is_empty() and not session_preprocess_applied:
        tips.append(
            "Fold-local PreprocessRecipe statistics were refit on each fold's training rows only."
        )
        if preprocess.encode == "target":
            tips.append(
                "Fold-local target encoding fits category means on fold-train labels only; "
                "fold-eval rows receive those frozen means and never contribute label stats."
            )
        if preprocess.encode == "infrequent":
            tips.append("Infrequent-level maps are learned from fold-train category counts only.")
        if preprocess.select is not None:
            tips.append(
                "Fold-local feature selection fits on fold-train transformed features only "
                "(variance, univariate, and model-based SelectFromModel)."
            )
        if preprocess.outliers is not None:
            tips.append(
                "Fold-local outlier fences are fit on fold-train only and applied to "
                "fold-eval with frozen bounds (detect/cap; no row drops inside CV)."
            )
        if preprocess.binning is not None:
            tips.append("Fold-local bin edges are learned from fold-train finite values only.")
        if preprocess.dates:
            tips.append(
                "Fold-local date expansion is row-wise deterministic; including it in "
                "the recipe avoids Session-global extract_dates before CV."
            )
        if preprocess.text is not None:
            tips.append(
                "Fold-local text vectorizers fit vocabulary/IDF on fold-train documents "
                "only; fold-eval rows use the frozen mapping."
            )
        if preprocess.reduce is not None:
            tips.append(
                "Fold-local PCA fits the rotation on fold-train numeric columns only; "
                "fold-eval rows use the frozen components."
            )
    if strategy_name == "time":
        tips.append(
            "Time-series folds respect row order by the time-role column within train; "
            "they do not invent a calendar-aware embargo."
        )
    return tips


def _cv_interpretation(
    *,
    metric: str,
    mean_metrics: dict[str, float],
    std_metrics: dict[str, float],
    n_folds: int,
    task: str,
) -> list[str]:
    mean = mean_metrics.get(metric)
    std = std_metrics.get(metric, 0.0)
    lines = [
        (
            f"Observed mean {metric}={mean:.6f} with fold std={std:.6f} "
            f"across {n_folds} folds ({task})."
        )
    ]
    if mean is not None and std > 0 and abs(mean) > 1e-12 and (std / abs(mean)) > 0.2:
        lines.append(
            "Fold coefficient of variation exceeds 0.2 — estimate instability is material."
        )
    if task == "classification" and "balanced_accuracy" in mean_metrics:
        gap = mean_metrics.get("accuracy", 0.0) - mean_metrics["balanced_accuracy"]
        if gap > 0.05:
            lines.append(
                f"Mean accuracy exceeds mean balanced accuracy by {gap:.3f}; "
                "class imbalance may dominate the primary score."
            )
    return lines


def _cv_recommendations(
    *,
    metric: str,
    mean_metrics: dict[str, float],
    std_metrics: dict[str, float],
    held_out: list[str],
    session_preprocess_applied: bool,
    preprocess: PreprocessRecipe | None,
) -> list[str]:
    tips = [
        f"Use mean±std of '{metric}' for model comparison; report fold count and population=train.",
        f"After selection, evaluate once on {held_out[0]} (held out from CV).",
    ]
    std = std_metrics.get(metric, 0.0)
    if std > 0.05:
        tips.append("Consider more folds, grouped/time-aware splits, or a simpler estimator.")
    if session_preprocess_applied:
        tips.append(
            "This run used allow_session_global_preprocess=True. For honest selection, "
            "re-ingest or reattach unpoisoned data, pass preprocess=PreprocessRecipe(...), "
            "and avoid Session-global impute/encode/scale/select/outliers/text/reduce "
            "before CV."
        )
    elif preprocess is not None:
        tips.append(
            "Refit the winning preprocess+estimator on the full train partition before deployment."
        )
        if (
            preprocess.encode in {"target", "infrequent"}
            or preprocess.select is not None
            or preprocess.text is not None
            or preprocess.reduce is not None
        ):
            tips.append(
                "Fold-local encode/select/text/reduce inside CV is leakage-safer than "
                "Session-global equivalents for model selection; persist the final "
                "Session plans via save_pipeline after the confirmed refit."
            )
    return tips
