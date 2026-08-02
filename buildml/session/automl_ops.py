"""Thin Session facades over buildml.automl."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.automl.checkpoint import load_automl_bundle, save_automl_bundle
from buildml.automl.explain_hooks import fit_result_summary
from buildml.automl.search import run_automl
from buildml.automl.types import AutoMLBudget, AutoMLMethod, AutoMLSelection
from buildml.core.errors import ValidationError
from buildml.model.supervised import EvaluateResult, evaluate_estimator
from buildml.preprocess.fold import PreprocessRecipe

TaskType = Literal["classification", "regression", "auto"]
CvStrategy = Literal["auto", "kfold", "stratified", "group", "stratified_group", "time"]


def run_automl_op(
    session,
    *,
    task: TaskType = "auto",
    method: AutoMLMethod = "randomized",
    selection: AutoMLSelection = "cv",
    n_trials: int = 20,
    cv: int | Any = 3,
    outer_cv: int | Any = 3,
    cv_strategy: CvStrategy = "auto",
    ranking_metric: str | None = None,
    families: Sequence[str] | None = None,
    include_recipe_search: bool = True,
    include_ensembles: bool = False,
    max_ensemble_bases: int = 3,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
    random_state: int | None = 0,
    groups: pd.Series | None = None,
    budget: AutoMLBudget | None = None,
) -> Any:
    """Run AutoML model-family + recipe-strategy search on the train partition.

    Notes
    -----
    **Leakage:** Same refusal as classical CV/search when Session-global
    preprocess already poisoned the frame. Session test never enters selection.
    """
    session.assert_can_fit("train")
    plan, result, fit_result = run_automl(
        session.dataset,
        session._split_plan,
        task=task,
        method=method,
        selection=selection,
        n_trials=n_trials,
        cv=cv,
        outer_cv=outer_cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        families=None if families is None else tuple(families),
        include_recipe_search=include_recipe_search,
        include_ensembles=include_ensembles,
        max_ensemble_bases=max_ensemble_bases,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
        random_state=random_state,
        groups=groups,
        budget=budget,
    )
    session._automl_plan = plan
    session._automl_result = result
    if fit_result is not None:
        session._fit_result = fit_result
    session._record(
        "run_automl",
        {
            "method": method,
            "selection": selection,
            "task": task,
            "n_trials": n_trials,
            "cv": cv if isinstance(cv, int) else type(cv).__name__,
            "outer_cv": outer_cv if isinstance(outer_cv, int) else type(outer_cv).__name__,
            "ranking_metric": ranking_metric,
            "families": None if families is None else list(families),
            "include_recipe_search": include_recipe_search,
            "include_ensembles": include_ensembles,
            "max_ensemble_bases": max_ensemble_bases,
            "allow_session_global_preprocess": allow_session_global_preprocess,
            "refit": refit,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def evaluate_automl(
    session,
    *,
    partition: Literal["train", "validation", "test"] = "test",
) -> EvaluateResult:
    """Evaluate the last AutoML winner with classical supervised metrics."""
    if session._fit_result is None:
        raise ValidationError("No fitted AutoML winner. Call run_automl first.")
    plan = getattr(session, "_automl_plan", None)
    result = evaluate_estimator(
        session.dataset, session._split_plan, session._fit_result, partition=partition
    )
    if plan is not None:
        tips = list(result.recommendations)
        tips.insert(
            0,
            (
                f"AutoML winner family={plan.best_family}, "
                f"recipe={plan.best_recipe_strategy}, kind={plan.best_kind}."
            ),
        )
        for note in plan.disclosures[:3]:
            tips.append(note)
        result.recommendations = tips
        result.diagnostics["automl"] = plan.to_dict()
    session._record(
        "evaluate_automl",
        {
            "partition": partition,
            "best_family": None if plan is None else plan.best_family,
            "selection": None if plan is None else plan.selection,
        },
        result_summary=result.to_dict(),
    )
    return result


def save_automl_bundle_op(session, path: str | Path) -> Path:
    """Persist the active AutoMLPlan as ``buildml.automl_bundle.v1``."""
    plan = getattr(session, "_automl_plan", None)
    if plan is None:
        raise ValidationError("No AutoML plan. Call run_automl first.")
    out = save_automl_bundle(
        path,
        plan,
        fit_result=getattr(session, "_fit_result", None),
        automl_result=getattr(session, "_automl_result", None),
    )
    session._record(
        "save_automl_bundle",
        {"path": str(out)},
        result_summary={
            "path": str(out),
            "best_family": plan.best_family,
            "best_recipe_strategy": plan.best_recipe_strategy,
            "method": plan.method,
        },
    )
    return out


def load_automl_bundle_op(session, path: str | Path) -> Any:
    """Load an AutoML bundle into this Session."""
    plan, fit_result = load_automl_bundle(path)
    session._automl_plan = plan
    session._automl_result = None
    session._fit_result = fit_result
    session._record(
        "load_automl_bundle",
        {
            "path": str(path),
            "best_family": plan.best_family,
            "method": plan.method,
        },
        result_summary=plan.to_dict(),
    )
    return session
