"""Thin Session facades over buildml.automl."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

import pandas as pd

from buildml.automl.checkpoint import load_automl_bundle, save_automl_bundle
from buildml.automl.explain_hooks import fit_result_summary
from buildml.automl.search import run_automl
from buildml.automl.types import (
    AutoMLBackend,
    AutoMLBudget,
    AutoMLMethod,
    AutoMLSelection,
    EnsembleMode,
)
from buildml.core.errors import ValidationError
from buildml.model.supervised import EvaluateResult, evaluate_estimator
from buildml.preprocess.fold import PreprocessRecipe

TaskType = Literal["classification", "regression", "auto"]
CvStrategy = Literal["auto", "kfold", "stratified", "group", "stratified_group", "time"]


def run_automl_op(
    session,
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
    families: Sequence[str] | None = None,
    include_recipe_search: bool = True,
    include_industry_families: bool = True,
    include_ensembles: bool = False,
    ensemble_mode: EnsembleMode = "voting",
    max_ensemble_bases: int = 3,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
    random_state: int | None = 0,
    groups: pd.Series | None = None,
    budget: AutoMLBudget | None = None,
    time_budget: float | None = None,
) -> Any:
    """Run AutoML model-family and recipe-strategy search on the train partition.

    Delegates to :func:`buildml.automl.search.run_automl`, stores the
    :class:`~buildml.automl.results.AutoMLPlan` and winner on Session, and
    optionally refits the best candidate. Follow with :func:`evaluate_automl`
    for classical supervised metrics on a holdout partition.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    backend:
        ``native``, ``optuna``, ``flaml``, or ``autogluon`` search backend.
    task:
        ``classification``, ``regression``, or ``auto`` to infer from target.
    method:
        Search method (``randomized``, ``grid``, ``optuna``, ``evolutionary``).
    selection:
        How to rank trials: ``cv``, ``nested``, or ``validation``.
    n_trials:
        Maximum candidate trials under the trial budget.
    cv:
        Inner CV folds or splitter for ``selection='cv'`` / ``'nested'``.
    outer_cv:
        Outer CV folds when ``selection='nested'``.
    cv_strategy:
        CV splitter strategy (``auto``, ``kfold``, ``stratified``, etc.).
    ranking_metric:
        Metric to rank candidates; defaults to task-appropriate score.
    families:
        Optional subset of model family names to search.
    include_recipe_search:
        When True, search discrete fold-local recipe strategies.
    include_industry_families:
        When True, extend catalog with GBDT families when extras installed.
    include_ensembles:
        When True, evaluate voting ensembles from diverse top families.
    ensemble_mode:
        Ensemble types to score when ``include_ensembles=True``.
    max_ensemble_bases:
        Maximum base families combined in one ensemble trial.
    preprocess:
        Fixed fold-local recipe when ``include_recipe_search=False``.
    allow_session_global_preprocess:
        Allow search when Session-global preprocess was already applied.
    refit:
        When True, refit the best candidate on all train rows after selection.
    random_state:
        Seed for randomized search and CV splitters.
    groups:
        Optional group labels for grouped CV strategies.
    budget:
        Structured trial/time budget caps for the search loop.
    time_budget:
        Optional wall-clock seconds cap for the search.

    Returns
    -------
    AutoMLResult
        Ranked trial table, winner metadata, and search disclosures.
        Session ``_fit_result`` is set when ``refit=True``.

    Notes
    -----
    **Leakage:** Same refusal as classical CV/search when Session-global
    preprocess already poisoned the frame. Session test never enters selection.
    """
    session.assert_can_fit("train")
    plan, result, fit_result = run_automl(
        session.dataset,
        session._split_plan,
        backend=backend,
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
        include_industry_families=include_industry_families,
        include_ensembles=include_ensembles,
        ensemble_mode=ensemble_mode,
        max_ensemble_bases=max_ensemble_bases,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
        random_state=random_state,
        groups=groups,
        budget=budget,
        time_budget=time_budget,
    )
    session._automl_plan = plan
    session._automl_result = result
    if fit_result is not None:
        session._fit_result = fit_result
    session._record(
        "run_automl",
        {
            "backend": backend,
            "method": method,
            "selection": selection,
            "task": task,
            "n_trials": n_trials,
            "cv": cv if isinstance(cv, int) else type(cv).__name__,
            "outer_cv": outer_cv if isinstance(outer_cv, int) else type(outer_cv).__name__,
            "ranking_metric": ranking_metric,
            "families": None if families is None else list(families),
            "include_recipe_search": include_recipe_search,
            "include_industry_families": include_industry_families,
            "include_ensembles": include_ensembles,
            "ensemble_mode": ensemble_mode,
            "max_ensemble_bases": max_ensemble_bases,
            "allow_session_global_preprocess": allow_session_global_preprocess,
            "refit": refit,
            "random_state": random_state,
            "time_budget": time_budget,
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
    """Evaluate the last AutoML winner with classical supervised metrics.

    Delegates to :func:`buildml.model.supervised.evaluate_estimator` on the
    refitted winner stored in Session ``_fit_result``. Annotates diagnostics
    with AutoML plan metadata when available.

    Parameters
    ----------
    session:
        Active Session with a refitted AutoML winner in ``_fit_result``.
    partition:
        Split partition to score (``train``, ``validation``, or ``test``).

    Returns
    -------
    EvaluateResult
        Metrics, diagnostics, and recommendations for the winning estimator.

    Raises
    ------
    ValidationError
        When no refitted AutoML winner exists on the Session.
    """
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
    """Persist the active AutoML plan as ``buildml.automl_bundle.v1``.

    Delegates to :func:`buildml.automl.checkpoint.save_automl_bundle`.
    Reload with :func:`load_automl_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with an AutoML plan from :func:`run_automl_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no AutoML plan exists on the Session.
    """
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


def load_automl_bundle_op(session, path: str | Path, *, trusted: bool = False) -> Any:
    """Load an AutoML bundle into this Session.

    Delegates to :func:`buildml.automl.checkpoint.load_automl_bundle`,
    restores plan and refitted winner, and clears search result cache.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded AutoML plan.
    path:
        Path to a ``buildml.automl_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with AutoML plan and fit result attached for chaining.
    """
    plan, fit_result = load_automl_bundle(path, trusted=trusted)
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
    return cast("Session", session)