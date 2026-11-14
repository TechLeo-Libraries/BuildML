"""FLAML tabular AutoML adapter with Session leakage discipline."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score

from buildml.automl.extras import require_flaml
from buildml.automl.results import AutoMLPlan, AutoMLResult, AutoMLTrial
from buildml.automl.types import AutoMLBudget, AutoMLConfig, AutoMLSelection
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.model.supervised import (
    FitResult,
    TaskType,
    _feature_target_frames,
    _infer_task,
    evaluate_estimator,
    weight_column,
)


def run_flaml_adapter(
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
) -> tuple[AutoMLPlan, AutoMLResult, FitResult | None]:
    """Run FLAML AutoML on Session train only; never touch test.

    FLAML handles internal model selection and preprocessing. BuildML fold-local
    PreprocessRecipe search is bypassed: disclosures state this explicitly.
    Session test never enters fit or selection scoring.

    Parameters
    ----------
    dataset:
        BuildML dataset with features and a single target column.
    split_plan:
        Train/validation/test split; only train rows are passed to FLAML.
    task:
        ``classification``, ``regression``, or ``auto`` to infer from the target.
    selection:
        ``cv`` reports train-only internal scoring. ``validation`` evaluates
        the selected model on the Session validation partition. The best
        score uses that evaluation when finite, with the internal score as
        a fallback. Trial rows summarize each learner's best internal score.
    ranking_metric:
        Metric passed to FLAML; converted to FLAML loss orientation internally.
    time_budget:
        Wall-clock seconds for FLAML fit; overrides ``budget.max_time_seconds``.
    budget:
        Optional :class:`~buildml.automl.types.AutoMLBudget`; defaults when omitted.
    random_state:
        Seed for FLAML reproducibility.
    refit:
        When True, wrap the best model in a classical FitResult after search.

    Returns
    -------
    tuple[AutoMLPlan, AutoMLResult, FitResult | None]
        Train-selected plan, ranked trial summary, and optional FitResult.

    Raises
    ------
    ValidationError
        When ``selection='nested'``, validation partition is missing, or FLAML
        finishes without a fitted model.
    MissingExtraError
        When flaml is not installed.
    """
    require_flaml(feature="FLAML AutoML backend")
    from flaml import AutoML as FlamlAutoML
    from flaml.automl.ml import is_min_metric

    assert_fit_partition(split_plan, "train")
    if selection == "nested":
        raise ValidationError(
            "backend='flaml' does not support selection='nested'. "
            "Use backend='native' with selection='nested' or selection='validation'."
        )
    if selection == "validation" and not split_plan.validation_indices:
        raise ValidationError(
            "selection='validation' requires a Session validation partition."
        )

    budget = budget or AutoMLBudget()
    seconds = time_budget if time_budget is not None else budget.max_time_seconds
    if seconds is None:
        seconds = max(30.0, float(budget.max_trials) * 5.0)

    x_train, y_train, feature_cols, target, sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    resolved_task = _infer_task(y_train, task, None)
    flaml_task = "classification" if resolved_task == "classification" else "regression"
    metric = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    provider_metric = _flaml_metric(metric)
    higher_is_better = not isinstance(provider_metric, str) or not is_min_metric(provider_metric)

    automl = FlamlAutoML()
    fit_kwargs: dict[str, Any] = {
        "X_train": x_train.copy(),
        "y_train": y_train.copy(),
        "task": flaml_task,
        "time_budget": float(seconds),
        "metric": provider_metric,
        "seed": random_state,
        "verbose": 0,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.to_numpy(copy=True)

    automl.fit(**fit_kwargs)

    if getattr(automl, "model", None) is None:
        raise ValidationError("FLAML AutoML finished without a fitted model")
    # Wrap the full AutoML object: ``automl.model`` / ``.estimator`` peel away
    # FLAML's categorical handling and break string columns on Session evaluate.
    best_estimator = _FlamlSklearnWrapper(automl, feature_cols=feature_cols)
    best_config = dict(getattr(automl, "best_config", {}) or {})
    best_loss = float(getattr(automl, "best_loss", float("nan")))
    if not isfinite(best_loss):
        raise ValidationError("FLAML AutoML finished without a finite internal validation loss")
    internal_score = _flaml_score(best_loss, higher_is_better=higher_is_better)
    display_score = internal_score
    score_source = "provider_internal"
    warnings: list[str] = []
    trials = _flaml_trials(automl, metric=metric, higher_is_better=higher_is_better)

    fit_result: FitResult | None = None
    if selection == "validation" or refit:
        fit_result = FitResult(
            estimator=best_estimator,
            task=resolved_task,
            feature_columns=tuple(feature_cols),
            target_column=str(target),
            n_train_rows=int(len(x_train)),
            weight_column=weight_column(dataset),
        )
    if selection == "validation":
        assert fit_result is not None
        ev = evaluate_estimator(dataset, split_plan, fit_result, partition="validation")
        validation_score = ev.metrics.get(metric, float("nan"))
        if isfinite(validation_score):
            display_score = validation_score
            score_source = "session_validation"
        else:
            warnings.append(
                f"Session validation did not produce a finite '{metric}' score. "
                "Best score uses FLAML's internal validation on the training partition."
            )
    std_score = 0.0

    if not trials:
        trials.append(AutoMLTrial(
            trial=0,
            kind="single",
            family=getattr(automl, "best_estimator", None) or "flaml",
            recipe_strategy="flaml_internal",
            params=best_config,
            mean_score=internal_score,
            std_score=std_score,
            mean_metrics={metric: internal_score},
            std_metrics={metric: std_score},
        ))

    config = AutoMLConfig(
        method="randomized",
        selection=selection,
        task=resolved_task,
        n_trials=len(trials),
        ranking_metric=metric,
        include_recipe_search=False,
        random_state=random_state,
        families=("flaml",),
        budget=budget,
        extras={
            "backend": "flaml",
            "time_budget_seconds": seconds,
            "best_config": best_config,
            "best_score_source": score_source,
        },
    )
    disclosures = [
        "backend=flaml (buildml[automl-industry]); FLAML internal model search on train only.",
        "Fold-local PreprocessRecipe search bypassed: FLAML handles preprocessing internally.",
        "Session test never entered FLAML fit or selection scoring.",
        "Trial rows summarize each learner's best finite internal result when available. "
        "The best score source is recorded in the config.",
        f"time_budget={seconds:.1f}s; metric={metric}; best_loss={best_loss}.",
    ]
    limitations = [
        "FLAML adapter does not support nested CV or fold-local recipe strategy search.",
        "Industry adapter scores are not directly comparable to native fold-local AutoML trials.",
    ]
    plan = AutoMLPlan(
        task=resolved_task,
        method="randomized",
        selection=selection,
        ranking_metric=metric,
        best_family="flaml",
        best_recipe_strategy="flaml_internal",
        best_kind="single",
        best_params=best_config,
        best_recipe={},
        best_score=float(display_score),
        best_std=float(std_score),
        feature_columns=tuple(feature_cols),
        target_column=str(target),
        n_train_rows=int(len(x_train)),
        estimator_=best_estimator if fit_result is None else fit_result.estimator,
        n_trials=len(trials),
        families_searched=("flaml",),
        recipe_strategies_searched=("flaml_internal",),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = AutoMLResult(
        task=resolved_task,
        method="randomized",
        selection=selection,
        ranking_metric=metric,
        trials=trials,
        best_family="flaml",
        best_recipe_strategy="flaml_internal",
        best_kind="single",
        best_params=best_config,
        best_score=float(display_score),
        best_std=float(std_score),
        families_searched=("flaml",),
        recipe_strategies_searched=("flaml_internal",),
        n_train_rows=int(len(x_train)),
        feature_columns=tuple(feature_cols),
        target_column=str(target),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        limitations=tuple(limitations),
        recommendations=(
            f"Winner: flaml config={best_config}, score={display_score:.6f}.",
            "Confirm once on held-out test via evaluate_automl(partition='test').",
        ),
        config=config.to_dict(),
    )
    return plan, result, fit_result


class _FlamlSklearnWrapper:
    """sklearn-compatible wrapper that keeps FLAML's full predict path.

    ``AutoML.predict`` applies FLAML's internal feature transforms; calling
    ``automl.model.predict`` (or ``.estimator``) does not and fails on string
    columns under modern XGBoost.
    """

    def __init__(self, automl: Any, *, feature_cols: list[str]) -> None:
        """Store the fitted FLAML AutoML object and feature column list.

        Keeps the full FLAML ``AutoML`` instance (not ``automl.model`` alone)
        so categorical transforms run on predict.

        Parameters
        ----------
        automl:
            Fitted FLAML ``AutoML`` instance (full object, not ``automl.model``).
        feature_cols:
            Feature columns used at fit time for column subsetting at predict.
        """
        self._automl = automl
        self._feature_cols = list(feature_cols)
        classes = getattr(automl, "classes_", None)
        if classes is None and getattr(automl, "model", None) is not None:
            classes = getattr(automl.model, "classes_", None)
        self.classes_ = classes

    def fit(self, X: pd.DataFrame, y: Any = None, **kwargs: Any) -> _FlamlSklearnWrapper:
        """No-op fit for sklearn compatibility; FLAML is already trained.

        FLAML fit happens in :func:`run_flaml_adapter`; this method exists so
        sklearn pipelines and Session wrappers accept the object.

        Parameters
        ----------
        X:
            Ignored; FLAML was fit externally on train data.
        y:
            Ignored.
        **kwargs:
            Ignored.

        Returns
        -------
        _FlamlSklearnWrapper
            ``self`` for sklearn pipeline compatibility.
        """
        del X, y, kwargs
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict labels using FLAML's full predict path.

        Uses ``AutoML.predict`` so internal categorical transforms are applied.

        Parameters
        ----------
        X:
            Feature frame; subset to ``feature_cols`` when all are present.

        Returns
        -------
        array-like
            Predictions from ``AutoML.predict``.
        """
        frame = X[self._feature_cols] if all(c in X.columns for c in self._feature_cols) else X
        return self._automl.predict(frame)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Predict class probabilities when FLAML exposes predict_proba.

        Uses FLAML's full predict path with optional column subsetting so
        string categoricals survive modern XGBoost backends.

        Parameters
        ----------
        X:
            Feature frame; subset to ``feature_cols`` when all are present.

        Returns
        -------
        array-like
            Probability matrix from ``AutoML.predict_proba``.

        Raises
        ------
        AttributeError
            When the FLAML model does not expose ``predict_proba``.
        """
        frame = X[self._feature_cols] if all(c in X.columns for c in self._feature_cols) else X
        if hasattr(self._automl, "predict_proba"):
            return self._automl.predict_proba(frame)
        raise AttributeError("FLAML model does not expose predict_proba")


def _flaml_metric(metric: str) -> str | Callable[..., tuple[float, dict[str, float]]]:
    if metric == "f1_weighted":
        return _weighted_f1_loss
    return "macro_f1" if metric == "f1_macro" else metric


def _weighted_f1_loss(
    x_eval: Any,
    y_eval: Any,
    estimator: Any,
    _labels: Any,
    _x_train: Any,
    _y_train: Any,
    weight_eval: Any = None,
    *_: Any,
) -> tuple[float, dict[str, float]]:
    score = float(f1_score(
        y_eval, estimator.predict(x_eval), average="weighted",
        zero_division=0, sample_weight=weight_eval,
    ))
    return 1.0 - score, {"f1_weighted": score}


def _flaml_trials(
    automl: Any,
    *,
    metric: str,
    higher_is_better: bool,
) -> list[AutoMLTrial]:
    rows: list[AutoMLTrial] = []
    configs = getattr(automl, "best_config_per_estimator", None) or {}
    losses = getattr(automl, "best_loss_per_estimator", None) or {}
    for learner, loss in losses.items():
        config = configs.get(learner)
        if config is None or loss is None or not isfinite(loss):
            continue
        score = _flaml_score(float(loss), higher_is_better=higher_is_better)
        rows.append(
            AutoMLTrial(
                trial=len(rows),
                kind="single",
                family=learner,
                recipe_strategy="flaml_internal",
                params=dict(config),
                mean_score=score,
                std_score=0.0,
                mean_metrics={metric: score},
                std_metrics={metric: 0.0},
            )
        )
    rows.sort(key=lambda t: t.mean_score, reverse=higher_is_better)
    return rows


def _flaml_score(loss: float, *, higher_is_better: bool) -> float:
    return 1.0 - loss if higher_is_better else loss


