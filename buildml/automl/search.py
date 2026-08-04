"""Leakage-safe AutoML: model-family + recipe-strategy search beyond single-estimator HPO."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline as SkPipeline

from buildml.automl.results import AutoMLPlan, AutoMLResult, AutoMLTrial
from buildml.automl.spaces import (
    ModelFamily,
    RecipeStrategy,
    families_for_task,
    family_by_name,
    recipe_strategies,
)
from buildml.automl.types import (
    AutoMLBackend,
    AutoMLBudget,
    AutoMLConfig,
    AutoMLMethod,
    AutoMLSelection,
    EnsembleMode,
)
from buildml.core.errors import LeakageError, ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.model.selection import (
    _LOWER_IS_BETTER,
    _refuse_session_global_cv_leakage,
    cv_score,
)
from buildml.model.supervised import (
    FitResult,
    TaskType,
    _feature_target_frames,
    _infer_task,
    evaluate_estimator,
    fit_estimator,
    fit_kwargs_for_sample_weight,
    weight_column,
)
from buildml.preprocess.fold import (
    PreprocessRecipe,
    build_fold_preprocessor,
    transform_fold_features,
)

CvStrategy = Literal["auto", "kfold", "stratified", "group", "stratified_group", "time"]


@dataclass(slots=True)
class _Candidate:
    kind: Literal["single", "voting", "stacking"]
    family: str
    recipe: RecipeStrategy
    params: dict[str, Any]
    ensemble_bases: tuple[str, ...] = ()


def export_comparison_metrics(result: AutoMLResult, path: str | Path) -> Path:
    """Export ranked trial comparison metrics to JSON for downstream analysis.

    Writes a lightweight JSON file with best candidate summary and per-trial
    scores suitable for dashboards or external comparison tools.

    Parameters
    ----------
    result:
        Completed :class:`~buildml.automl.results.AutoMLResult` from
        :func:`run_automl`.
    path:
        Destination file path; parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        Resolved path to the written JSON file.
    """
    import json
    from pathlib import Path as PathType

    destination = PathType(path)
    board = result.leaderboard()
    payload = {
        "ranking_metric": result.ranking_metric,
        "backend": result.config.get("backend", "native"),
        "method": result.method,
        "selection": result.selection,
        "default_selection_note": (
            "Default selection is 'cv' (train-fold ranking). "
            "Use selection='nested' for outer post-selection estimates."
        ),
        "best_family": result.best_family,
        "best_score": result.best_score,
        "outer_score_mean": result.outer_score_mean,
        "outer_score_std": result.outer_score_std,
        "nested_cv_disclosed": result.selection == "nested",
        "leaderboard": board.to_dict(orient="records"),
        "trials": [
            {
                "trial": t.trial,
                "kind": t.kind,
                "family": t.family,
                "recipe_strategy": t.recipe_strategy,
                "mean_score": t.mean_score,
                "std_score": t.std_score,
                "mean_metrics": dict(t.mean_metrics),
                "params": dict(t.params),
                "ensemble_bases": list(t.ensemble_bases),
            }
            for t in result.trials
        ],
        "disclosures": list(result.disclosures),
        "limitations": list(result.limitations),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def run_automl(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: AutoMLBackend = "native",
    task: TaskType = "auto",
    method: AutoMLMethod = "randomized",
    selection: AutoMLSelection = "cv",
    n_trials: int = 20,
    cv: int | Any = 3,
    outer_cv: int | Any = 3,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    families: tuple[str, ...] | list[str] | None = None,
    include_recipe_search: bool = True,
    include_industry_families: bool = True,
    include_ensembles: bool = False,
    ensemble_mode: EnsembleMode = "voting",
    max_ensemble_bases: int = 3,
    preprocess: PreprocessRecipe | None = None,
    session_preprocess_applied: bool = False,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
    random_state: int | None = 0,
    groups: pd.Series | None = None,
    budget: AutoMLBudget | None = None,
    time_budget: float | None = None,
) -> tuple[AutoMLPlan, AutoMLResult, FitResult | None]:
    """Search model families and fold-local preprocess strategies on train only.

    Goes beyond single-estimator HPO by jointly ranking:

    * estimator **families** (logistic / RF / GB / …)
    * discrete **recipe strategies** (impute/scale/encode/select combinations)
    * modest per-family hyperparameter catalogs
    * optional **voting** ensembles of diverse top families

    Parameters
    ----------
    dataset:
        BuildML dataset with features and a single target column.
    split_plan:
        Train/validation/test split; train partition is used for search and
        refit; test never enters selection scoring.
    backend:
        ``native`` (default), ``optuna`` (deepened Optuna path),
        ``flaml`` or ``autogluon`` (industry adapters; ``buildml[automl-industry]``).
    task:
        ``classification``, ``regression``, or ``auto`` to infer from the target.
    method:
        ``randomized`` (default, no extra), ``grid`` (small exhaustive catalog),
        ``optuna`` (requires ``buildml[automl]``), or ``evolutionary`` (in-tree GA).
    selection:
        ``cv``: rank by train-fold CV;
        ``nested``: outer train folds after inner selection (honest post-selection
        estimate) then refit best globally;
        ``validation``: rank on Session validation (never test). Requires a
        validation partition.
    n_trials:
        Maximum number of candidate trials to evaluate under the trial budget.
    cv:
        Inner cross-validation folds or splitter for ``selection='cv'`` /
        ``'nested'``.
    outer_cv:
        Outer cross-validation folds when ``selection='nested'``.
    cv_strategy:
        CV splitter strategy: ``auto``, ``kfold``, ``stratified``, ``group``,
        ``stratified_group``, or ``time``.
    ranking_metric:
        Metric used to rank candidates; defaults to task-appropriate score.
    families:
        Optional subset of model family names to search; ``None`` uses the full
        catalog for the resolved task.
    include_recipe_search:
        When True, search discrete fold-local recipe strategies. When False,
        use ``preprocess`` (or passthrough) as a fixed recipe.
    include_industry_families:
        When True and ``buildml[automl-industry]`` GBDT libs are installed, extend
        the native catalog with LightGBM / XGBoost / CatBoost families.
    include_ensembles:
        When True, after single-model trials, evaluate a small number of voting
        ensembles built from diverse top families under shared recipes.
    ensemble_mode:
        When ``include_ensembles=True``, score ``voting``, ``stacking``, or ``both``
        ensembles of diverse top families.
    max_ensemble_bases:
        Maximum number of diverse base families combined in one ensemble trial.
    preprocess:
        Fixed fold-local recipe used when ``include_recipe_search=False``.
    session_preprocess_applied:
        When True, Session-global preprocess was already applied to the frame.
    allow_session_global_preprocess:
        Same hard refusal as classical ``cv_score`` / ``grid_search`` when
        Session-global prep already poisoned the frame.
    refit:
        When True, refit the best candidate on all train rows after selection.
    random_state:
        Seed for randomized search and CV splitters.
    groups:
        Optional group labels for grouped CV strategies.
    budget:
        Optional :class:`~buildml.automl.types.AutoMLBudget` caps on trials,
        families, recipes, ensembles, and wall-clock time.
    time_budget:
        Optional wall-clock cap (seconds). Industry backends use this directly;
        native search stops after the current trial when exceeded.

    Returns
    -------
    tuple[AutoMLPlan, AutoMLResult, FitResult | None]
        Train-selected plan, full ranked search result, and optional classical
        :class:`~buildml.model.supervised.FitResult` when ``refit=True``.

    Raises
    ------
    ValidationError
        When backend, method, selection, or partition configuration is invalid.
    LeakageError
        When Session-global preprocess would leak into fold-local CV without
        explicit opt-in.

    Notes
    -----
    **Leakage:** Session test never enters selection. Fold-local recipes refit
    on fold-train only. This is **not** NAS, not causal discovery, and not a
    fully automated AI scientist: the search space is a disclosed finite
    catalog under a trial budget.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    _refuse_session_global_cv_leakage(
        session_preprocess_applied=session_preprocess_applied,
        preprocess=preprocess,
        allow_session_global_preprocess=allow_session_global_preprocess,
    )
    if backend not in {"native", "optuna", "flaml", "autogluon"}:
        raise ValidationError(f"Unsupported AutoML backend: {backend!r}")
    if method not in {"grid", "randomized", "optuna", "evolutionary"}:
        raise ValidationError(f"Unsupported AutoML method: {method!r}")
    if ensemble_mode not in {"voting", "stacking", "both"}:
        raise ValidationError(f"Unsupported ensemble_mode: {ensemble_mode!r}")
    if selection not in {"cv", "nested", "validation"}:
        raise ValidationError(f"Unsupported AutoML selection: {selection!r}")
    if selection == "validation" and not split_plan.validation_indices:
        raise ValidationError(
            "selection='validation' requires a Session validation partition. "
            "Use split(..., validation_size=...) or selection='cv'/'nested'."
        )
    if n_trials < 1:
        raise ValidationError("n_trials must be >= 1")
    if max_ensemble_bases < 2:
        raise ValidationError("max_ensemble_bases must be >= 2")

    budget = budget or AutoMLBudget(max_trials=n_trials)
    if budget.max_trials < 1:
        raise ValidationError("budget.max_trials must be >= 1")
    if time_budget is not None:
        budget.max_time_seconds = float(time_budget)
    n_trials = min(int(n_trials), int(budget.max_trials))

    # Industry adapters: train-only, disclosed internal preprocessing.
    if backend == "flaml":
        from buildml.automl.adapters.flaml import run_flaml_adapter

        return run_flaml_adapter(
            dataset,
            split_plan,
            task=task,
            selection=selection,
            ranking_metric=ranking_metric,
            time_budget=time_budget,
            budget=budget,
            random_state=random_state,
            refit=refit,
        )
    if backend == "autogluon":
        from buildml.automl.adapters.autogluon import run_autogluon_adapter

        return run_autogluon_adapter(
            dataset,
            split_plan,
            task=task,
            selection=selection,
            ranking_metric=ranking_metric,
            time_budget=time_budget,
            budget=budget,
            random_state=random_state,
            refit=refit,
        )

    effective_method: AutoMLMethod = method
    if backend == "optuna":
        effective_method = "optuna"

    x_train, y_train, feature_cols, target, _sw = _feature_target_frames(
        dataset, split_plan, "train"
    )
    # Probe task with a cheap default family.
    probe_task = task
    if probe_task == "auto":
        probe_task = "classification"  # temporary; refined via _infer_task below
    probe_families = families_for_task(
        "classification" if _looks_classification(y_train) else "regression",
        names=None,
        max_families=1,
    )
    resolved_task = _infer_task(y_train, task, probe_families[0].build(random_state))
    metric = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    higher_is_better = metric not in _LOWER_IS_BETTER

    fams = families_for_task(
        resolved_task,
        names=families,
        max_families=budget.max_families,
        include_industry=include_industry_families,
    )
    recipes = recipe_strategies(
        include_recipe_search=include_recipe_search,
        fixed=preprocess,
        max_strategies=budget.max_recipe_strategies,
    )

    config = AutoMLConfig(
        backend=backend,
        method=effective_method,
        selection=selection,
        task=resolved_task,
        n_trials=n_trials,
        cv=int(cv) if isinstance(cv, int) else 3,
        outer_cv=int(outer_cv) if isinstance(outer_cv, int) else 3,
        ranking_metric=metric,
        include_recipe_search=include_recipe_search,
        include_ensembles=include_ensembles,
        max_ensemble_bases=max_ensemble_bases,
        random_state=random_state,
        families=tuple(f.name for f in fams),
        budget=budget,
        extras={
            "ensemble_mode": ensemble_mode,
            "include_industry_families": include_industry_families,
            "time_budget_seconds": budget.max_time_seconds,
        },
    )

    import time as _time

    started = _time.monotonic()

    candidates = _build_candidates(
        fams,
        recipes,
        method=effective_method,
        n_trials=n_trials,
        random_state=random_state,
        include_ensembles=False,  # ensemble trials added after single ranking
    )

    if effective_method == "optuna":
        trials = _run_optuna_trials(
            dataset,
            split_plan,
            fams=fams,
            recipes=recipes,
            task=resolved_task,
            metric=metric,
            n_trials=n_trials,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            random_state=random_state,
            budget=budget,
            started=started,
        )
    elif effective_method == "evolutionary":
        trials = _run_evolutionary_trials(
            dataset,
            split_plan,
            fams=fams,
            recipes=recipes,
            task=resolved_task,
            metric=metric,
            n_trials=n_trials,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            random_state=random_state,
            started=started,
            budget=budget,
        )
    else:
        trials = _score_candidates(
            dataset,
            split_plan,
            candidates,
            task=resolved_task,
            metric=metric,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            random_state=random_state,
            fams=fams,
        )

    if not trials:
        raise ValidationError("AutoML produced no scored trials.")

    trials.sort(key=lambda t: t.mean_score, reverse=higher_is_better)
    for i, trial in enumerate(trials):
        trial.trial = i

    if include_ensembles:
        ens_trials = _score_ensemble_candidates(
            dataset,
            split_plan,
            single_trials=trials,
            fams=fams,
            recipes=recipes,
            task=resolved_task,
            metric=metric,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            random_state=random_state,
            max_bases=max_ensemble_bases,
            max_ensemble_trials=budget.max_ensemble_trials,
            ensemble_mode=ensemble_mode,
        )
        if ens_trials:
            trials.extend(ens_trials)
            trials.sort(key=lambda t: t.mean_score, reverse=higher_is_better)
            for i, trial in enumerate(trials):
                trial.trial = i

    outer_mean: float | None = None
    outer_std: float | None = None
    if selection == "nested":
        outer_mean, outer_std, nested_warnings = _nested_outer_estimate(
            dataset,
            split_plan,
            fams=fams,
            recipes=recipes,
            method=effective_method,
            task=resolved_task,
            metric=metric,
            n_trials=max(4, min(n_trials, 12)),
            outer_cv=outer_cv,
            inner_cv=cv,
            cv_strategy=cv_strategy,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            random_state=random_state,
            include_recipe_search=include_recipe_search,
            preprocess=preprocess,
            budget=budget,
        )
    else:
        nested_warnings = ()

    best = trials[0]
    disclosures = _disclosures(
        backend=backend,
        method=effective_method,
        selection=selection,
        fams=fams,
        recipes=recipes,
        include_ensembles=include_ensembles,
        ensemble_mode=ensemble_mode,
        metric=metric,
        n_trials=len(trials),
        budget=budget,
        session_global_override=bool(
            session_preprocess_applied and allow_session_global_preprocess
        ),
    )
    warnings = list(nested_warnings)
    limitations = _limitations(
        backend=backend,
        selection=selection,
        method=effective_method,
        n_trials=len(trials),
        budget=budget,
    )
    recommendations = _recommendations(
        selection=selection,
        best=best,
        outer_mean=outer_mean,
        held_out=_held_out(split_plan),
    )

    fit_result: FitResult | None = None
    estimator: Any = None
    if refit:
        fit_result = _refit_best(
            dataset,
            split_plan,
            best=best,
            fams=fams,
            task=resolved_task,
            random_state=random_state,
        )
        estimator = fit_result.estimator
    else:
        # Still materialize an unfitted-then-fitted estimator for the plan.
        fit_result = _refit_best(
            dataset,
            split_plan,
            best=best,
            fams=fams,
            task=resolved_task,
            random_state=random_state,
        )
        estimator = fit_result.estimator

    plan = AutoMLPlan(
        task=resolved_task,
        method=effective_method,
        selection=selection,
        ranking_metric=metric,
        best_family=best.family,
        best_recipe_strategy=best.recipe_strategy,
        best_kind=best.kind,
        best_params=dict(best.params),
        best_recipe=dict(best.recipe),
        best_score=float(best.mean_score),
        best_std=float(best.std_score),
        feature_columns=tuple(feature_cols),
        target_column=str(target),
        n_train_rows=int(len(x_train)),
        estimator_=estimator,
        ensemble_bases=tuple(best.ensemble_bases),
        n_trials=len(trials),
        families_searched=tuple(f.name for f in fams),
        recipe_strategies_searched=tuple(r.name for r in recipes),
        outer_score_mean=outer_mean,
        outer_score_std=outer_std,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = AutoMLResult(
        task=resolved_task,
        method=effective_method,
        selection=selection,
        ranking_metric=metric,
        trials=trials,
        best_family=best.family,
        best_recipe_strategy=best.recipe_strategy,
        best_kind=best.kind,
        best_params=dict(best.params),
        best_score=float(best.mean_score),
        best_std=float(best.std_score),
        outer_score_mean=outer_mean,
        outer_score_std=outer_std,
        families_searched=tuple(f.name for f in fams),
        recipe_strategies_searched=tuple(r.name for r in recipes),
        n_train_rows=int(len(x_train)),
        feature_columns=tuple(feature_cols),
        target_column=str(target),
        ensemble_bases=tuple(best.ensemble_bases),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        limitations=tuple(limitations),
        recommendations=tuple(recommendations),
        config=config.to_dict(),
    )
    return plan, result, fit_result


def _looks_classification(y: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(y):
        nunique = int(y.nunique(dropna=True))
        return nunique <= max(20, int(0.05 * max(len(y), 1)))
    return True


def _held_out(split_plan: SplitPlan) -> list[str]:
    held = ["test"]
    if split_plan.validation_indices:
        held.append("validation")
    return held


def _build_candidates(
    fams: list[ModelFamily],
    recipes: list[RecipeStrategy],
    *,
    method: AutoMLMethod,
    n_trials: int,
    random_state: int | None,
    include_ensembles: bool,
) -> list[_Candidate]:
    del include_ensembles  # reserved for future joint generation
    out: list[_Candidate] = []
    if method == "grid":
        for fam, recipe in product(fams, recipes):
            keys = list(fam.param_grid)
            if not keys:
                out.append(_Candidate("single", fam.name, recipe, {}))
                continue
            value_lists = [fam.param_grid[k] for k in keys]
            for values in product(*value_lists):
                params = dict(zip(keys, values, strict=True))
                out.append(_Candidate("single", fam.name, recipe, params))
        if len(out) > n_trials:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(out), size=n_trials, replace=False)
            out = [out[int(i)] for i in sorted(idx)]
        return out

    # randomized: sample family × recipe × params
    rng = np.random.default_rng(random_state)
    for trial_i in range(n_trials):
        fam = fams[int(rng.integers(0, len(fams)))]
        recipe = recipes[int(rng.integers(0, len(recipes)))]
        if fam.param_distributions:
            sampler = ParameterSampler(
                fam.param_distributions,
                n_iter=1,
                random_state=None if random_state is None else int(random_state) + trial_i,
            )
            params = dict(next(iter(sampler)))
        else:
            params = {}
        out.append(_Candidate("single", fam.name, recipe, params))
    return out


def _score_candidates(
    dataset: Dataset,
    split_plan: SplitPlan,
    candidates: list[_Candidate],
    *,
    task: Literal["classification", "regression"],
    metric: str,
    cv: int | Any,
    cv_strategy: CvStrategy,
    selection: AutoMLSelection,
    groups: pd.Series | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
    random_state: int | None,
    fams: list[ModelFamily],
) -> list[AutoMLTrial]:
    fam_map = {f.name: f for f in fams}
    trials: list[AutoMLTrial] = []
    for idx, cand in enumerate(candidates):
        estimator = _build_estimator(cand, fam_map, task=task, random_state=random_state)
        score_pack = _score_one(
            dataset,
            split_plan,
            estimator,
            recipe=cand.recipe.recipe,
            task=task,
            metric=metric,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
        )
        if score_pack is None:
            continue
        mean_score, std_score, mean_metrics, std_metrics = score_pack
        trials.append(
            AutoMLTrial(
                trial=idx,
                kind=cand.kind,
                family=cand.family,
                recipe_strategy=cand.recipe.name,
                params=dict(cand.params),
                recipe=cand.recipe.recipe.to_dict(),
                mean_score=mean_score,
                std_score=std_score,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                ensemble_bases=tuple(cand.ensemble_bases),
            )
        )
    return trials


def _score_one(
    dataset: Dataset,
    split_plan: SplitPlan,
    estimator: Any,
    *,
    recipe: PreprocessRecipe,
    task: Literal["classification", "regression"],
    metric: str,
    cv: int | Any,
    cv_strategy: CvStrategy,
    selection: AutoMLSelection,
    groups: pd.Series | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
) -> tuple[float, float, dict[str, float], dict[str, float]] | None:
    try:
        if selection == "validation":
            return _score_on_validation(
                dataset,
                split_plan,
                estimator,
                recipe=recipe,
                task=task,
                metric=metric,
            )
        # cv and nested both use train-fold CV for candidate ranking;
        # nested adds a separate outer estimate afterward.
        cv_result = cv_score(
            dataset,
            split_plan,
            estimator,
            task=task,
            cv=cv,
            cv_strategy=cv_strategy,
            scoring_metric=metric,
            groups=groups,
            preprocess=recipe,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
        )
        return (
            float(cv_result.mean_metrics[metric]),
            float(cv_result.std_metrics.get(metric, float("nan"))),
            dict(cv_result.mean_metrics),
            dict(cv_result.std_metrics),
        )
    except (ValidationError, LeakageError, ValueError, TypeError):
        return None


def _score_on_validation(
    dataset: Dataset,
    split_plan: SplitPlan,
    estimator: Any,
    *,
    recipe: PreprocessRecipe,
    task: Literal["classification", "regression"],
    metric: str,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Fit on train (fold-local recipe on full train), score on validation only."""
    x_train, y_train, feature_cols, target, sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )

    model = clone(estimator)
    if recipe is not None and not recipe.is_empty():
        prep = build_fold_preprocessor(x_train, recipe, y_train)
        x_fit = transform_fold_features(prep, x_train)
        fitted = clone(model)
        fitted.fit(x_fit, y_train, **fit_kwargs_for_sample_weight(fitted, sample_weight))
        bundled = SkPipeline([("preprocess", prep), ("model", fitted)])
        fit = FitResult(
            estimator=bundled,
            task=task,
            feature_columns=tuple(feature_cols),
            target_column=target,
            n_train_rows=int(len(x_train)),
            weight_column=weight_column(dataset),
        )
    else:
        fit = fit_estimator(dataset, split_plan, model, task=task)

    ev = evaluate_estimator(dataset, split_plan, fit, partition="validation")
    score = float(ev.metrics[metric])
    return score, 0.0, dict(ev.metrics), {k: 0.0 for k in ev.metrics}


def _build_estimator(
    cand: _Candidate,
    fam_map: dict[str, ModelFamily],
    *,
    task: Literal["classification", "regression"],
    random_state: int | None,
) -> Any:
    if cand.kind == "voting":
        named = []
        for base_name in cand.ensemble_bases:
            fam = fam_map[base_name]
            named.append((base_name, fam.build(random_state)))
        if task == "classification":
            if all(hasattr(est, "predict_proba") for _, est in named):
                return VotingClassifier(estimators=named, voting="soft")
            return VotingClassifier(estimators=named, voting="hard")
        return VotingRegressor(estimators=named)
    if cand.kind == "stacking":
        named = []
        for base_name in cand.ensemble_bases:
            fam = fam_map[base_name]
            named.append((base_name, fam.build(random_state)))
        if task == "classification":
            from sklearn.linear_model import LogisticRegression

            return StackingClassifier(
                estimators=named,
                final_estimator=LogisticRegression(max_iter=500),
                cv=3,
            )
        from sklearn.linear_model import Ridge

        return StackingRegressor(estimators=named, final_estimator=Ridge(), cv=3)
    fam = fam_map[cand.family]
    return fam.build(random_state, **cand.params)


def _score_ensemble_candidates(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    single_trials: list[AutoMLTrial],
    fams: list[ModelFamily],
    recipes: list[RecipeStrategy],
    task: Literal["classification", "regression"],
    metric: str,
    cv: int | Any,
    cv_strategy: CvStrategy,
    selection: AutoMLSelection,
    groups: pd.Series | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
    random_state: int | None,
    max_bases: int,
    max_ensemble_trials: int,
    ensemble_mode: EnsembleMode = "voting",
) -> list[AutoMLTrial]:
    """Build voting/stacking candidates from diverse top single-model families."""
    fam_map = {f.name: f for f in fams}
    recipe_map = {r.name: r for r in recipes}
    best_by_family: dict[str, AutoMLTrial] = {}
    for trial in single_trials:
        if trial.kind != "single":
            continue
        cur = best_by_family.get(trial.family)
        if cur is None or trial.mean_score > cur.mean_score:
            best_by_family[trial.family] = trial
    ordered = sorted(best_by_family.values(), key=lambda t: t.mean_score, reverse=True)
    if len(ordered) < 2:
        return []

    ens_trials: list[AutoMLTrial] = []
    top_recipe_name = ordered[0].recipe_strategy
    recipe = recipe_map.get(top_recipe_name, recipes[0])
    kinds: list[Literal["voting", "stacking"]] = []
    if ensemble_mode in {"voting", "both"}:
        kinds.append("voting")
    if ensemble_mode in {"stacking", "both"}:
        kinds.append("stacking")

    for kind in kinds:
        for n_bases in range(2, min(max_bases, len(ordered)) + 1):
            if len(ens_trials) >= max_ensemble_trials:
                break
            bases = tuple(t.family for t in ordered[:n_bases])
            family_label = "+".join(bases)
            cand = _Candidate(
                kind=kind,
                family=family_label,
                recipe=recipe,
                params={},
                ensemble_bases=bases,
            )
            estimator = _build_estimator(cand, fam_map, task=task, random_state=random_state)
            score_pack = _score_one(
                dataset,
                split_plan,
                estimator,
                recipe=recipe.recipe,
                task=task,
                metric=metric,
                cv=cv,
                cv_strategy=cv_strategy,
                selection=selection,
                groups=groups,
                session_preprocess_applied=session_preprocess_applied,
                allow_session_global_preprocess=allow_session_global_preprocess,
            )
            if score_pack is None:
                continue
            mean_score, std_score, mean_metrics, std_metrics = score_pack
            ens_trials.append(
                AutoMLTrial(
                    trial=10_000 + len(ens_trials),
                    kind=kind,
                    family=cand.family,
                    recipe_strategy=recipe.name,
                    params={},
                    recipe=recipe.recipe.to_dict(),
                    mean_score=mean_score,
                    std_score=std_score,
                    mean_metrics=mean_metrics,
                    std_metrics=std_metrics,
                    ensemble_bases=bases,
                )
            )
    return ens_trials


def _run_optuna_trials(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    fams: list[ModelFamily],
    recipes: list[RecipeStrategy],
    task: Literal["classification", "regression"],
    metric: str,
    n_trials: int,
    cv: int | Any,
    cv_strategy: CvStrategy,
    selection: AutoMLSelection,
    groups: pd.Series | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
    random_state: int | None,
    budget: AutoMLBudget,
    started: float,
) -> list[AutoMLTrial]:
    from buildml.automl.extras import require_optuna

    optuna = require_optuna(feature="Optuna AutoML backend")
    import time as _time

    higher_is_better = metric not in _LOWER_IS_BETTER
    secondary = budget.secondary_metric
    multi_obj = bool(budget.multi_objective and secondary)
    if multi_obj:
        directions = [
            "maximize" if higher_is_better else "minimize",
            "maximize" if secondary not in _LOWER_IS_BETTER else "minimize",
        ]
    else:
        directions = ["maximize" if higher_is_better else "minimize"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        if budget.enable_pruning
        else optuna.pruners.NopPruner()
    )
    study_kwargs: dict[str, Any] = {"sampler": sampler, "pruner": pruner}
    if budget.study_storage:
        study_kwargs["storage"] = budget.study_storage
        study_kwargs["study_name"] = "buildml_automl"
        study_kwargs["load_if_exists"] = True

    if multi_obj:
        study = optuna.create_study(directions=directions, **study_kwargs)
    else:
        study = optuna.create_study(
            direction=directions[0],
            **study_kwargs,
        )
    fam_map = {f.name: f for f in fams}
    trial_rows: list[AutoMLTrial] = []

    def _objective(trial: Any) -> float | tuple[float, float]:
        if budget.max_time_seconds is not None:
            elapsed = _time.monotonic() - started
            if elapsed >= budget.max_time_seconds:
                raise optuna.exceptions.OptunaError("AutoML time budget exceeded")

        fam_name = trial.suggest_categorical("family_name", [f.name for f in fams])
        fam = fam_map[fam_name]
        recipe_name = trial.suggest_categorical("recipe_strategy", [r.name for r in recipes])
        recipe = next(r for r in recipes if r.name == recipe_name)
        params: dict[str, Any] = {}
        for key, choices in fam.param_distributions.items():
            if not choices:
                continue
            param_key = f"param__{fam.name}__{key}"
            if any(c is None for c in choices):
                labels = [str(c) for c in choices]
                picked = trial.suggest_categorical(param_key, labels)
                params[key] = next(c for c in choices if str(c) == picked)
            else:
                params[key] = trial.suggest_categorical(param_key, list(choices))

        cand = _Candidate("single", fam.name, recipe, params)
        estimator = _build_estimator(cand, fam_map, task=task, random_state=random_state)
        score_pack = _score_one(
            dataset,
            split_plan,
            estimator,
            recipe=recipe.recipe,
            task=task,
            metric=metric,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
        )
        if score_pack is None:
            if multi_obj:
                bad = float("-inf") if higher_is_better else float("inf")
                return bad, bad
            return float("-inf") if higher_is_better else float("inf")
        mean_score, std_score, mean_metrics, std_metrics = score_pack
        secondary_score = float(mean_metrics.get(secondary, mean_score)) if secondary else mean_score
        trial_rows.append(
            AutoMLTrial(
                trial=int(trial.number),
                kind="single",
                family=fam.name,
                recipe_strategy=recipe.name,
                params=dict(params),
                recipe=recipe.recipe.to_dict(),
                mean_score=mean_score,
                std_score=std_score,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
            )
        )
        if budget.enable_pruning:
            trial.report(mean_score, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        if multi_obj:
            return mean_score, secondary_score
        return mean_score

    study.optimize(_objective, n_trials=n_trials)
    return trial_rows


def _run_evolutionary_trials(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    fams: list[ModelFamily],
    recipes: list[RecipeStrategy],
    task: Literal["classification", "regression"],
    metric: str,
    n_trials: int,
    cv: int | Any,
    cv_strategy: CvStrategy,
    selection: AutoMLSelection,
    groups: pd.Series | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
    random_state: int | None,
    started: float,
    budget: AutoMLBudget,
) -> list[AutoMLTrial]:
    """Evolutionary search over family × recipe × param catalog (native GA)."""
    import time as _time

    from buildml.model.selection import (
        _mutate_individual,
        _parse_evolutionary_genes,
        _sample_evolutionary_individual,
        _tournament_select,
        _uniform_crossover,
    )

    fam_map = {f.name: f for f in fams}
    population_size = max(4, min(12, n_trials // 2))
    n_generations = max(1, n_trials // population_size)
    eval_budget = min(n_trials, population_size * n_generations)
    higher = metric not in _LOWER_IS_BETTER

    param_space: dict[str, Any] = {
        "family_name": [f.name for f in fams],
        "recipe_strategy": [r.name for r in recipes],
    }
    for fam in fams:
        for key, choices in fam.param_distributions.items():
            param_space[f"{fam.name}__{key}"] = list(choices) if choices else [None]

    genes = _parse_evolutionary_genes(param_space=param_space, recipe_space=None)
    if not genes:
        return _score_candidates(
            dataset,
            split_plan,
            _build_candidates(
                fams,
                recipes,
                method="randomized",
                n_trials=n_trials,
                random_state=random_state,
                include_ensembles=False,
            ),
            task=task,
            metric=metric,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
            random_state=random_state,
            fams=fams,
        )

    rng = np.random.default_rng(random_state)
    scored: dict[tuple[tuple[str, Any], ...], AutoMLTrial] = {}

    def _genome_key(individual: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted((str(k), v) for k, v in individual.items()))

    def _evaluate(individual: dict[str, Any]) -> AutoMLTrial:
        key = _genome_key(individual)
        cached = scored.get(key)
        if cached is not None:
            return cached
        if len(scored) >= eval_budget:
            return AutoMLTrial(
                trial=-1,
                kind="single",
                family="budget_exhausted",
                recipe_strategy="",
                mean_score=float("-inf") if higher else float("inf"),
            )
        if budget.max_time_seconds is not None:
            if _time.monotonic() - started >= budget.max_time_seconds:
                return AutoMLTrial(
                    trial=-1,
                    kind="single",
                    family="time_budget_exhausted",
                    recipe_strategy="",
                    mean_score=float("-inf") if higher else float("inf"),
                )

        fam_name = str(individual.get("family_name", fams[0].name))
        recipe_name = str(individual.get("recipe_strategy", recipes[0].name))
        fam = fam_map.get(fam_name, fams[0])
        recipe = next((r for r in recipes if r.name == recipe_name), recipes[0])
        params = {
            key.split("__", 1)[1]: individual[gene]
            for gene in individual
            if gene.startswith(f"{fam.name}__")
            for key in [gene]
        }
        cand = _Candidate("single", fam.name, recipe, params)
        estimator = _build_estimator(cand, fam_map, task=task, random_state=random_state)
        score_pack = _score_one(
            dataset,
            split_plan,
            estimator,
            recipe=recipe.recipe,
            task=task,
            metric=metric,
            cv=cv,
            cv_strategy=cv_strategy,
            selection=selection,
            groups=groups,
            session_preprocess_applied=session_preprocess_applied,
            allow_session_global_preprocess=allow_session_global_preprocess,
        )
        if score_pack is None:
            row = AutoMLTrial(
                trial=len(scored),
                kind="single",
                family=fam.name,
                recipe_strategy=recipe.name,
                params=params,
                mean_score=float("-inf") if higher else float("inf"),
            )
        else:
            mean_score, std_score, mean_metrics, std_metrics = score_pack
            row = AutoMLTrial(
                trial=len(scored),
                kind="single",
                family=fam.name,
                recipe_strategy=recipe.name,
                params=dict(params),
                recipe=recipe.recipe.to_dict(),
                mean_score=mean_score,
                std_score=std_score,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
            )
        scored[key] = row
        return row

    population = [_sample_evolutionary_individual(genes, rng) for _ in range(population_size)]
    fitness = [_evaluate(ind) for ind in population]

    for _generation in range(n_generations):
        ranked_idx = sorted(
            range(len(population)),
            key=lambda i: fitness[i].mean_score,
            reverse=higher,
        )
        if _generation + 1 >= n_generations or len(scored) >= eval_budget:
            break
        next_pop: list[dict[str, Any]] = []
        next_fit: list[AutoMLTrial] = []
        for elite_rank in ranked_idx[:2]:
            next_pop.append(dict(population[elite_rank]))
            next_fit.append(fitness[elite_rank])
        while len(next_pop) < population_size:
            if len(scored) >= eval_budget:
                filler = ranked_idx[len(next_pop) % len(ranked_idx)]
                next_pop.append(dict(population[filler]))
                next_fit.append(fitness[filler])
                continue
            p1 = _tournament_select(population, fitness, 3, higher, rng)
            p2 = _tournament_select(population, fitness, 3, higher, rng)
            if rng.random() < 0.7:
                child, _sib = _uniform_crossover(p1, p2, genes, rng)
            else:
                child = dict(p1)
            child = _mutate_individual(child, genes, 0.2, rng)
            next_pop.append(child)
            next_fit.append(_evaluate(child))
        population = next_pop[:population_size]
        fitness = next_fit[:population_size]

    trials = [t for t in scored.values() if t.family not in {"budget_exhausted", "time_budget_exhausted"}]
    trials.sort(key=lambda t: t.mean_score, reverse=higher)
    for i, trial in enumerate(trials):
        trial.trial = i
    return trials


def _nested_outer_estimate(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    fams: list[ModelFamily],
    recipes: list[RecipeStrategy],
    method: AutoMLMethod,
    task: Literal["classification", "regression"],
    metric: str,
    n_trials: int,
    outer_cv: int | Any,
    inner_cv: int | Any,
    cv_strategy: CvStrategy,
    groups: pd.Series | None,
    session_preprocess_applied: bool,
    allow_session_global_preprocess: bool,
    random_state: int | None,
    include_recipe_search: bool,
    preprocess: PreprocessRecipe | None,
    budget: AutoMLBudget,
) -> tuple[float | None, float | None, tuple[str, ...]]:
    """Honest outer estimate: for each outer fold, run a smaller inner AutoML."""
    del session_preprocess_applied, allow_session_global_preprocess
    del include_recipe_search, preprocess, budget
    from buildml.model.selection import _resolve_splitter

    x_train, y_train, _fc, _tg, sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    group_values, _strategy_name, splitter, row_order = _resolve_splitter(
        dataset=dataset,
        split_plan=split_plan,
        y_train=y_train,
        cv=outer_cv,
        cv_strategy=cv_strategy,
        groups=groups,
        task=task,
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

    outer_scores: list[float] = []
    warnings: list[str] = []
    split_iter = (
        splitter.split(x_reset, y_reset, group_reset)
        if group_reset is not None
        else splitter.split(x_reset, y_reset)
    )
    fam_map = {f.name: f for f in fams}

    for fold_id, (train_pos, eval_pos) in enumerate(split_iter):
        if set(train_pos) & set(eval_pos):
            raise LeakageError("AutoML nested outer fold train/eval indices overlap")
        # Build a temporary Dataset/SplitPlan-like view via index remapping:
        # score candidates with cv_score-style logic on outer-train only, then
        # evaluate the winner on outer-eval.
        inner_candidates = _build_candidates(
            fams,
            recipes,
            method="randomized" if method == "optuna" else method,
            n_trials=n_trials,
            random_state=None if random_state is None else int(random_state) + fold_id,
            include_ensembles=False,
        )
        best_score = None
        best_cand: _Candidate | None = None
        higher = metric not in _LOWER_IS_BETTER
        for cand in inner_candidates:
            estimator = _build_estimator(cand, fam_map, task=task, random_state=random_state)
            # Manual fold CV on outer-train positions only.
            try:
                score = _manual_inner_cv_score(
                    x_reset.iloc[list(train_pos)],
                    y_reset.iloc[list(train_pos)],
                    None if w_reset is None else w_reset.iloc[list(train_pos)],
                    estimator,
                    recipe=cand.recipe.recipe,
                    task=task,
                    metric=metric,
                    n_splits=int(inner_cv) if isinstance(inner_cv, int) else 3,
                    random_state=random_state,
                )
            except (ValidationError, ValueError, TypeError):
                continue
            if best_score is None or (score > best_score if higher else score < best_score):
                best_score = score
                best_cand = cand
        if best_cand is None:
            warnings.append(f"Outer fold {fold_id}: no viable inner candidate.")
            continue
        # Fit winner on outer-train, score outer-eval.
        estimator = _build_estimator(best_cand, fam_map, task=task, random_state=random_state)
        x_otr = x_reset.iloc[list(train_pos)]
        y_otr = y_reset.iloc[list(train_pos)]
        w_otr = None if w_reset is None else w_reset.iloc[list(train_pos)]
        x_oe = x_reset.iloc[list(eval_pos)]
        y_oe = y_reset.iloc[list(eval_pos)]
        recipe = best_cand.recipe.recipe
        if recipe is not None and not recipe.is_empty():
            prep = build_fold_preprocessor(x_otr, recipe, y_otr)
            x_fit = transform_fold_features(prep, x_otr)
            x_score = transform_fold_features(prep, x_oe)
        else:
            x_fit, x_score = x_otr, x_oe
        model = clone(estimator)
        model.fit(x_fit, y_otr, **fit_kwargs_for_sample_weight(model, w_otr))
        y_pred = model.predict(x_score)
        from buildml.model.selection import _score_predictions

        fold_metrics = _score_predictions(task, y_oe, y_pred, sample_weight=None)
        outer_scores.append(float(fold_metrics[metric]))

    if not outer_scores:
        return None, None, tuple(warnings + ["Nested outer estimate produced no folds."])
    mean = float(np.mean(outer_scores))
    std = float(np.std(outer_scores, ddof=1)) if len(outer_scores) > 1 else 0.0
    return mean, std, tuple(warnings)


def _manual_inner_cv_score(
    x: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None,
    estimator: Any,
    *,
    recipe: PreprocessRecipe,
    task: Literal["classification", "regression"],
    metric: str,
    n_splits: int,
    random_state: int | None,
) -> float:
    from sklearn.model_selection import KFold, StratifiedKFold

    from buildml.model.selection import _score_predictions

    n_splits = max(2, min(n_splits, len(x)))
    if task == "classification" and y.nunique() > 1:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = splitter.split(x, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(x)

    scores: list[float] = []
    x_reset = x.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    w_reset = None if sample_weight is None else sample_weight.reset_index(drop=True)
    for train_pos, eval_pos in splits:
        x_tr = x_reset.iloc[list(train_pos)]
        y_tr = y_reset.iloc[list(train_pos)]
        x_ev = x_reset.iloc[list(eval_pos)]
        y_ev = y_reset.iloc[list(eval_pos)]
        w_tr = None if w_reset is None else w_reset.iloc[list(train_pos)]
        if recipe is not None and not recipe.is_empty():
            prep = build_fold_preprocessor(x_tr, recipe, y_tr)
            x_fit = transform_fold_features(prep, x_tr)
            x_score = transform_fold_features(prep, x_ev)
        else:
            x_fit, x_score = x_tr, x_ev
        model = clone(estimator)
        model.fit(x_fit, y_tr, **fit_kwargs_for_sample_weight(model, w_tr))
        y_pred = model.predict(x_score)
        metrics = _score_predictions(task, y_ev, y_pred, sample_weight=None)
        scores.append(float(metrics[metric]))
    if not scores:
        raise ValidationError("Inner CV produced no scores")
    return float(np.mean(scores))


def _refit_best(
    dataset: Dataset,
    split_plan: SplitPlan,
    *,
    best: AutoMLTrial,
    fams: list[ModelFamily],
    task: Literal["classification", "regression"],
    random_state: int | None,
) -> FitResult:
    fam_map = {f.name: f for f in fams}
    if best.kind == "voting":
        # Rebuild family map entries for bases (best.family is joined names).
        for base in best.ensemble_bases:
            if base not in fam_map:
                fam_map[base] = family_by_name(task, base)
    if best.kind == "single":
        family_name = best.family
    elif best.ensemble_bases:
        family_name = best.ensemble_bases[0]
    else:
        family_name = best.family
    cand = _Candidate(
        kind=best.kind,
        family=family_name,
        recipe=RecipeStrategy(
            name=best.recipe_strategy,
            recipe=PreprocessRecipe(**_recipe_kwargs_from_dict(best.recipe)),
            description="best",
        ),
        params=dict(best.params),
        ensemble_bases=tuple(best.ensemble_bases),
    )
    # Fix family for voting: _build_estimator uses ensemble_bases.
    if best.kind in {"voting", "stacking"}:
        cand = _Candidate(
            kind=best.kind,
            family=best.family,
            recipe=cand.recipe,
            params={},
            ensemble_bases=tuple(best.ensemble_bases),
        )
    estimator = _build_estimator(cand, fam_map, task=task, random_state=random_state)
    recipe = cand.recipe.recipe
    x_train, y_train, feature_cols, target, sample_weight = _feature_target_frames(
        dataset, split_plan, "train"
    )
    if recipe is not None and not recipe.is_empty():
        prep = build_fold_preprocessor(x_train, recipe, y_train)
        x_fit = transform_fold_features(prep, x_train)
        fitted = clone(estimator)
        fitted.fit(x_fit, y_train, **fit_kwargs_for_sample_weight(fitted, sample_weight))
        bundled = SkPipeline([("preprocess", prep), ("model", fitted)])
        return FitResult(
            estimator=bundled,
            task=task,
            feature_columns=tuple(feature_cols),
            target_column=target,
            n_train_rows=int(len(x_train)),
            weight_column=weight_column(dataset),
        )
    return fit_estimator(dataset, split_plan, estimator, task=task)


def _recipe_kwargs_from_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild PreprocessRecipe kwargs from a to_dict() payload (safe subset)."""
    allowed = {
        "impute",
        "scale",
        "encode",
        "select",
        "outliers",
        "binning",
        "dates",
        "text",
        "reduce",
        "fill_value",
        "min_frequency",
        "select_threshold",
        "select_k",
        "select_score_func",
        "outlier_action",
        "iqr_multiplier",
        "zscore_threshold",
        "n_bins",
        "binning_encode_as",
        "date_include_time",
        "date_drop_original",
        "text_max_features",
        "reduce_n_components",
        "reduce_prefix",
        "reduce_drop_input",
        "target_smoothing",
        "text_drop_input",
    }
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if key.endswith("_columns") or key == "text_ngram_range":
            continue
        if key == "select_estimator":
            continue
        out[key] = value
    # Column lists
    for key in (
        "impute_columns",
        "scale_columns",
        "encode_columns",
        "select_columns",
        "outlier_columns",
        "binning_columns",
        "date_columns",
        "text_columns",
        "reduce_columns",
    ):
        if key in payload and payload[key] is not None:
            out[key] = tuple(payload[key])
    if "text_ngram_range" in payload and payload["text_ngram_range"] is not None:
        out["text_ngram_range"] = tuple(payload["text_ngram_range"])
    return out


def _disclosures(
    *,
    backend: AutoMLBackend,
    method: AutoMLMethod,
    selection: AutoMLSelection,
    fams: list[ModelFamily],
    recipes: list[RecipeStrategy],
    include_ensembles: bool,
    ensemble_mode: EnsembleMode,
    metric: str,
    n_trials: int,
    budget: AutoMLBudget,
    session_global_override: bool,
) -> list[str]:
    tips = [
        (
            "AutoML searches a finite disclosed catalog of model families and "
            "fold-local preprocess strategies: not neural architecture search (NAS), "
            "not causal discovery, and not a fully automated AI scientist."
        ),
        (
            f"Backend={backend}, method={method}, selection={selection}, "
            f"ranking_metric={metric}, n_trials_scored={n_trials}."
        ),
        (
            f"Families searched: {[f.name for f in fams]}; "
            f"recipe strategies: {[r.name for r in recipes]}."
        ),
        "Session test never enters selection scoring or fold membership.",
        (
            "Fold-local PreprocessRecipe steps refit on fold-train (or full train for "
            "validation selection / final refit) only."
        ),
    ]
    if budget.max_time_seconds is not None:
        tips.append(f"time_budget={budget.max_time_seconds}s cap disclosed.")
    if budget.max_trials:
        tips.append(f"trial_budget={budget.max_trials} cap disclosed.")
    if include_ensembles:
        tips.append(
            f"Optional {ensemble_mode} ensembles of diverse top families were scored "
            "under the same leakage contract."
        )
    if selection == "cv":
        tips.append(
            "Default selection='cv': rankings use train-fold CV means (optimistic vs "
            "outer holdout). For post-selection claims use selection='nested' "
            "(outer mean±std) or selection='validation', then confirm on Session test."
        )
    if selection == "nested":
        tips.append(
            "selection='nested' (prominent honesty path): outer nested scores are the "
            "post-selection estimate; inner means are selection evidence only."
        )
    if selection == "validation":
        tips.append(
            "Candidates were ranked on Session validation; confirm once on test after "
            "freezing the winner."
        )
    if session_global_override:
        tips.append(
            "allow_session_global_preprocess=True was set; Session-global prep may have "
            "poisoned fold honesty."
        )
    if method == "optuna" or backend == "optuna":
        tips.append(
            "Optuna TPE with optional pruning/study persistence (buildml[automl])."
        )
        if budget.multi_objective and budget.secondary_metric:
            tips.append(
                f"Multi-objective Optuna: primary={metric}, secondary={budget.secondary_metric}."
            )
    if method == "evolutionary":
        tips.append("Native evolutionary (GA) search over family/recipe/param genes.")
    if backend in {"flaml", "autogluon"}:
        tips.append(
            f"Industry backend={backend} bypasses fold-local recipe search; "
            "internal preprocessing applies on train only."
        )
    return tips


def _limitations(
    *,
    backend: AutoMLBackend,
    selection: AutoMLSelection,
    method: AutoMLMethod,
    n_trials: int,
    budget: AutoMLBudget,
) -> list[str]:
    out = [
        (
            f"Trial budget and catalog size bound exploration "
            f"({n_trials} scored trials, backend={backend}, method={method})."
        ),
        "Default catalogs omit deep nets and arbitrary sklearn Pipeline DAGs.",
        (
            "Recipe strategy search covers impute/scale/encode/select combinations: "
            "not arbitrary preprocess graphs."
        ),
        (
            "selection='cv' rankings are optimistic relative to a true outer holdout; "
            "prefer nested or validation+final test confirm."
            if selection == "cv"
            else (
                f"selection={selection!r} still leaves residual optimism "
                "from finite search."
            )
        ),
    ]
    if backend in {"flaml", "autogluon"}:
        out.append(
            "Industry adapters do not support nested CV or fold-local recipe strategy search."
        )
    if budget.max_time_seconds is not None:
        out.append("Time budget may stop search before all trial slots are explored.")
    return out


def _recommendations(
    *,
    selection: AutoMLSelection,
    best: AutoMLTrial,
    outer_mean: float | None,
    held_out: list[str],
) -> list[str]:
    tips = [
        (
            f"Winner: kind={best.kind}, family={best.family}, "
            f"recipe={best.recipe_strategy}, score={best.mean_score:.6f}."
        ),
        (
            f"Confirm once on held-out partition(s): {', '.join(held_out)} "
            "via evaluate_automl / evaluate."
        ),
        (
            "Persist with save_automl_bundle (disclosures) and/or "
            "save_pipeline (preprocess + estimator)."
        ),
    ]
    if outer_mean is not None:
        tips.append(
            f"Nested outer mean={outer_mean:.6f}; treat this as the post-selection estimate."
        )
    if selection == "cv":
        tips.append(
            "Default selection='cv' ranks by train-fold CV only. "
            "Prefer selection='nested' (prominent outer estimate) or "
            "selection='validation' before strong post-selection claims; "
            "then confirm once on Session test."
        )
    return tips
