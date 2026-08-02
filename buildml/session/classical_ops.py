"""Classical fit / evaluate / CV / search / pipeline orchestration."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


def fit(
    session, estimator: Any, *, task: Literal['classification', 'regression', 'auto'] = "auto"
) -> Session:
    """Fit a sklearn-compatible estimator on the train partition.

    Parameters
    ----------
    estimator:
        Unfitted estimator instance.
    task:
        Task type or ``auto``.

    Notes
    -----
    **Leakage:** Fits on train only. Call after split and preparation.
    """
    session.assert_can_fit("train")
    session._fit_result = fit_estimator(session.dataset, session._split_plan, estimator, task=task)
    session._record(
        "fit",
        {"estimator": type(estimator).__name__, "task": task},
        result_summary=session._fit_result.to_dict(),
    )
    return session


def predict(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "test",
    return_proba: bool = False,
) -> pd.Series | pd.DataFrame:
    """Predict labels or probabilities on a partition.

    Parameters
    ----------
    partition:
        Split partition to score.
    return_proba:
        If True and supported, return class probabilities.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    preds = predict_estimator(
        session.dataset,
        session._split_plan,
        session._fit_result,
        partition=partition,
        return_proba=return_proba,
    )
    session._record(
        "predict", {"partition": partition, "n_rows": int(len(preds)), "proba": return_proba}
    )
    return preds


def evaluate(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "test",
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
    include_plots: bool = False,
) -> EvaluateResult:
    """Evaluate the last fitted estimator on a partition.

    Returns metrics, diagnostics (confusion matrix / residuals), and
    recommendations — not a single score.

    Parameters
    ----------
    partition:
        Split partition to score.
    include_plots / export_figures / export_html:
        Optionally build the eval plot board (requires ``buildml[viz]``)
        and persist figures/HTML. Plot board is also stored on
        :attr:`last_plot_board`.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    result = evaluate_estimator(
        session.dataset, session._split_plan, session._fit_result, partition=partition
    )
    if include_plots or export_figures is not None or export_html is not None:
        board = build_eval_plot_board(
            session.dataset,
            session._split_plan,
            session._fit_result,
            partition=partition,
            export_figures=export_figures,
            export_html=export_html,
        )
        session._last_plot_board = board
        result.diagnostics["plot_board"] = {
            "figure_dir": board.figure_dir,
            "html_path": board.html_path,
            "figure_paths": dict(board.figure_paths),
            "skipped": list(board.skipped),
            "interpretation": list(board.interpretation),
        }
    session._record(
        "evaluate",
        {
            "partition": partition,
            "include_plots": include_plots,
            "export_figures": export_figures,
            "export_html": export_html,
        },
        result_summary=result.to_dict(),
    )
    return result


def eval_plots(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "test",
    include_learning_curve: bool = True,
    include_importance: bool = True,
    n_importance_repeats: int = 6,
    learning_curve_cv: int = 3,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
    show: bool = False,
) -> PlotBoardReport:
    """Build an evaluation plot board for the fitted estimator.

    Adaptive panels include confusion/residuals, ROC/PR, calibration,
    threshold tradeoffs, learning curves, and permutation importance.
    Panels degrade gracefully when ``predict_proba`` or binary targets
    are unavailable.

    Notes
    -----
    Requires ``pip install 'buildml[viz]'``. Delegates to
    :func:`buildml.model.plot_boards.build_eval_plot_board`.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    board = build_eval_plot_board(
        session.dataset,
        session._split_plan,
        session._fit_result,
        partition=partition,
        include_learning_curve=include_learning_curve,
        include_importance=include_importance,
        n_importance_repeats=n_importance_repeats,
        learning_curve_cv=learning_curve_cv,
        export_figures=export_figures,
        export_html=export_html,
        show=show,
    )
    session._last_plot_board = board
    session._record(
        "eval_plots",
        {
            "partition": partition,
            "include_learning_curve": include_learning_curve,
            "include_importance": include_importance,
            "n_importance_repeats": n_importance_repeats,
            "learning_curve_cv": learning_curve_cv,
            "n_figures": len(board.figure_paths) or len(board.figures),
            "n_skipped": len(board.skipped),
            "figure_dir": board.figure_dir,
            "html_path": board.html_path,
        },
    )
    return board


def compare_models(
    session,
    estimators: dict[str, Any],
    *,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    partition: Literal['train', 'validation', 'test'] = "test",
    ranking_metric: str | None = None,
) -> ModelComparison:
    """Fit/evaluate multiple estimators and return a ranked comparison card."""
    session.assert_can_fit("train")
    comparison = compare_estimators(
        session.dataset,
        session._split_plan,
        estimators,
        task=task,
        partition=partition,
        ranking_metric=ranking_metric,
    )
    session._last_comparison = comparison
    winner = comparison.rows[0]["model"]
    session._fit_result = comparison.fits[winner]
    session._record(
        "compare_models",
        {
            "estimators": list(estimators),
            "task": task,
            "partition": partition,
            "ranking_metric": ranking_metric,
        },
        result_summary={"winner": winner, "ranking_metric": comparison.ranking_metric},
    )
    return comparison


def cv_score(
    session,
    estimator: Any,
    *,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    cv: int | Any = 5,
    cv_strategy: Literal['auto', 'kfold', 'stratified', 'group', 'stratified_group', 'time'] = "auto",
    scoring_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
) -> CVScoreResult:
    """Cross-validate an estimator on the train partition only.

    Returns mean±std fold metrics, interpretation, limitations, and
    recommendations. The test partition is never used for fold membership
    or scoring.

    Parameters
    ----------
    estimator:
        Unfitted sklearn-compatible estimator.
    cv / cv_strategy:
        Fold count or splitter; strategy selects k-fold, stratified,
        group, or time-aware folds when ``cv`` is an integer.
    scoring_metric:
        Primary metric for summaries (defaults by task).
    groups:
        Optional group labels aligned to train rows.
    preprocess:
        Optional fold-local :class:`PreprocessRecipe` refit each fold.
    allow_session_global_preprocess:
        Explicit opt-in when Session-global preprocess already ran.
        Default ``False`` refuses that path even if a fold-local recipe is
        passed (recipes do not rebuild from raw/unpoisoned rows).

    Notes
    -----
    **Leakage:** If Session impute/encode/scale/text/reduce already ran, CV
    refuses unless ``allow_session_global_preprocess=True``. Prefer
    re-ingesting unpoisoned data, then fold-local recipes (including
    ``text`` and ``reduce``) for selection claims that include
    preprocessing. Custom transforms and resample stay Session-global.
    """
    session.assert_can_fit("train")
    result = run_cv_score(
        session.dataset,
        session._split_plan,
        estimator,
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        scoring_metric=scoring_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
    )
    session._last_cv = result
    session._record(
        "cv_score",
        {
            "estimator": type(estimator).__name__,
            "task": task,
            "cv": cv if isinstance(cv, int) else type(cv).__name__,
            "cv_strategy": cv_strategy,
            "scoring_metric": scoring_metric,
            "fold_preprocess": None if preprocess is None else preprocess.to_dict(),
        },
        result_summary={
            "scoring_metric": result.scoring_metric,
            "mean": result.mean_metrics.get(result.scoring_metric),
            "std": result.std_metrics.get(result.scoring_metric),
            "n_splits": result.n_splits,
            "cv_strategy": result.cv_strategy,
        },
    )
    return result


def nested_cv_score(
    session,
    estimator: Any,
    *,
    param_grid: dict[str, list[Any]] | None = None,
    param_distributions: dict[str, Any] | None = None,
    recipe_grid: dict[str, list[Any]] | None = None,
    recipe_distributions: dict[str, Any] | None = None,
    param_space: Any | None = None,
    recipe_space: Any | None = None,
    inner_search: Literal['auto', 'grid', 'randomized', 'optuna', 'evolutionary'] = "auto",
    n_iter: int = 10,
    n_trials: int = 20,
    population_size: int = 8,
    n_generations: int = 3,
    random_state: int | None = 42,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    outer_cv: int | Any = 5,
    inner_cv: int | Any = 3,
    cv_strategy: Literal['auto', 'kfold', 'stratified', 'group', 'stratified_group', 'time'] = "auto",
    scoring_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
    warm_start_studies: bool = False,
) -> NestedCVResult:
    """Outer-loop estimate after inner hyperparameter / recipe-knob search.

    Each outer fold chooses estimator params and/or fold-local recipe knobs
    (``select_k``, ``n_bins``, …) with inner CV on that fold's training rows
    only, then scores the winner on the outer-eval rows. Session test and
    validation partitions never enter either loop.

    Parameters
    ----------
    param_grid / param_distributions:
        Estimator search space (at most one). Optional when a recipe space
        is provided.
    recipe_grid / recipe_distributions:
        Fold-local recipe knob space (at most one). Requires ``preprocess``.
    param_space / recipe_space:
        Optuna spaces when ``inner_search='optuna'`` (or ``auto`` with these
        args). Declare-style dicts for ``inner_search='evolutionary'``.
        Optuna requires ``pip install 'buildml[optuna]'``.
    inner_search:
        ``auto``, ``grid``, ``randomized``, ``optuna``, or ``evolutionary``.
    n_trials:
        Optuna inner trials per outer fold; evolutionary ``max_evaluations``.
    population_size / n_generations:
        Evolutionary GA knobs when ``inner_search='evolutionary'``.
    outer_cv / inner_cv:
        Outer and inner fold counts or sklearn splitters.
    preprocess:
        Fold-local :class:`PreprocessRecipe` refit in both loops.
    warm_start_studies:
        Opt-in Optuna study sharing across outer folds (default False).
        Safe for Session test/validation (never scored); see nested CV notes.

    Notes
    -----
    Prefer this over reporting :meth:`grid_search` mean CV as a
    post-selection generalization claim. Read ``mean_metrics`` /
    ``std_metrics`` for the outer estimate and
    ``outer_folds[*].best_params`` / ``best_recipe_knobs`` for chosen
    configs (including Optuna / evolutionary winners).
    """
    session.assert_can_fit("train")
    result = run_nested_cv_score(
        session.dataset,
        session._split_plan,
        estimator,
        param_grid=param_grid,
        param_distributions=param_distributions,
        recipe_grid=recipe_grid,
        recipe_distributions=recipe_distributions,
        param_space=param_space,
        recipe_space=recipe_space,
        inner_search=inner_search,
        n_iter=n_iter,
        n_trials=n_trials,
        population_size=population_size,
        n_generations=n_generations,
        random_state=random_state,
        task=task,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        cv_strategy=cv_strategy,
        scoring_metric=scoring_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
        warm_start_studies=warm_start_studies,
    )
    session._last_nested_cv = result
    session._record(
        "nested_cv_score",
        {
            "estimator": type(estimator).__name__,
            "task": task,
            "outer_cv": outer_cv if isinstance(outer_cv, int) else type(outer_cv).__name__,
            "inner_cv": inner_cv if isinstance(inner_cv, int) else type(inner_cv).__name__,
            "cv_strategy": cv_strategy,
            "scoring_metric": scoring_metric,
            "search_method": result.search_method,
            "inner_search": inner_search,
            "n_trials": n_trials if result.search_method in {"optuna", "evolutionary"} else None,
            "population_size": population_size
            if result.search_method == "evolutionary"
            else None,
            "n_generations": n_generations if result.search_method == "evolutionary" else None,
            "warm_start_studies": bool(warm_start_studies),
            "recipe_grid": None
            if recipe_grid is None
            else {k: list(v) for k, v in recipe_grid.items()},
            "fold_preprocess": None if preprocess is None else preprocess.to_dict(),
        },
        result_summary={
            "scoring_metric": result.scoring_metric,
            "mean": result.mean_metrics.get(result.scoring_metric),
            "std": result.std_metrics.get(result.scoring_metric),
            "n_outer_splits": result.n_outer_splits,
            "n_inner_splits": result.n_inner_splits,
            "param_stability": result.inner_selection_summary.get("param_stability"),
            "search_method": result.search_method,
            "warm_start_studies": result.warm_start_studies,
        },
    )
    return result


def grid_search(
    session,
    estimator: Any,
    param_grid: dict[str, list[Any]] | None = None,
    *,
    recipe_grid: dict[str, list[Any]] | None = None,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    cv: int | Any = 5,
    cv_strategy: Literal['auto', 'kfold', 'stratified', 'group', 'stratified_group', 'time'] = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Grid-search estimator params and/or fold-local recipe knobs.

    Ranks configurations by mean CV score, never peeking at test. When
    ``refit=True`` (default), the winning params/knobs are refit on full
    train and become the active :attr:`fit_result`.
    """
    session.assert_can_fit("train")
    result = run_grid_search(
        session.dataset,
        session._split_plan,
        estimator,
        param_grid,
        recipe_grid=recipe_grid,
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )
    session._last_search = result
    if refit and result.refit_result is not None:
        session._fit_result = result.refit_result
    session._record(
        "grid_search",
        {
            "estimator": type(estimator).__name__,
            "param_grid": None
            if param_grid is None
            else {k: list(v) for k, v in param_grid.items()},
            "recipe_grid": None
            if recipe_grid is None
            else {k: list(v) for k, v in recipe_grid.items()},
            "task": task,
            "cv": cv if isinstance(cv, int) else type(cv).__name__,
            "cv_strategy": cv_strategy,
            "ranking_metric": ranking_metric,
            "refit": refit,
        },
        result_summary={
            "best_params": result.best_params,
            "best_recipe_knobs": result.best_recipe_knobs,
            "best_score": result.best_score,
            "best_std": result.best_std,
            "ranking_metric": result.ranking_metric,
            "n_trials": len(result.trials),
        },
    )
    return result


def randomized_search(
    session,
    estimator: Any,
    param_distributions: dict[str, Any] | None = None,
    *,
    recipe_distributions: dict[str, Any] | None = None,
    n_iter: int = 10,
    random_state: int | None = 42,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    cv: int | Any = 5,
    cv_strategy: Literal['auto', 'kfold', 'stratified', 'group', 'stratified_group', 'time'] = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Randomized search over estimator params and/or recipe knobs.

    Same leakage contract as :meth:`grid_search`: folds stay inside train;
    the winner may be refit onto the full training partition.
    """
    session.assert_can_fit("train")
    result = run_randomized_search(
        session.dataset,
        session._split_plan,
        estimator,
        param_distributions,
        recipe_distributions=recipe_distributions,
        n_iter=n_iter,
        random_state=random_state,
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )
    session._last_search = result
    if refit and result.refit_result is not None:
        session._fit_result = result.refit_result
    session._record(
        "randomized_search",
        {
            "estimator": type(estimator).__name__,
            "param_distributions": None
            if param_distributions is None
            else {k: str(v) for k, v in param_distributions.items()},
            "recipe_distributions": None
            if recipe_distributions is None
            else {k: str(v) for k, v in recipe_distributions.items()},
            "n_iter": n_iter,
            "random_state": random_state,
            "task": task,
            "cv": cv if isinstance(cv, int) else type(cv).__name__,
            "cv_strategy": cv_strategy,
            "ranking_metric": ranking_metric,
            "refit": refit,
        },
        result_summary={
            "best_params": result.best_params,
            "best_recipe_knobs": result.best_recipe_knobs,
            "best_score": result.best_score,
            "best_std": result.best_std,
            "ranking_metric": result.ranking_metric,
            "n_trials": len(result.trials),
        },
    )
    return result


def optuna_search(
    session,
    estimator: Any,
    *,
    param_space: Any | None = None,
    recipe_space: Any | None = None,
    n_trials: int = 20,
    random_state: int | None = 42,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    cv: int | Any = 5,
    cv_strategy: Literal['auto', 'kfold', 'stratified', 'group', 'stratified_group', 'time'] = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Optuna TPE search with leakage-safe train-fold CV.

    Requires ``pip install 'buildml[optuna]'``. ``param_space`` may be a
    ``trial -> dict`` callable or a declare-style mapping
    (``float`` / ``int`` / ``categorical``). ``recipe_space`` sweeps
    fold-local recipe knobs and requires ``preprocess``.
    """
    session.assert_can_fit("train")
    result = run_optuna_search(
        session.dataset,
        session._split_plan,
        estimator,
        param_space=param_space,
        recipe_space=recipe_space,
        n_trials=n_trials,
        random_state=random_state,
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )
    session._last_search = result
    if refit and result.refit_result is not None:
        session._fit_result = result.refit_result
    session._record(
        "optuna_search",
        {
            "estimator": type(estimator).__name__,
            "n_trials": n_trials,
            "random_state": random_state,
            "task": task,
            "cv": cv if isinstance(cv, int) else type(cv).__name__,
            "cv_strategy": cv_strategy,
            "ranking_metric": ranking_metric,
            "refit": refit,
            "has_param_space": param_space is not None,
            "has_recipe_space": recipe_space is not None,
        },
        result_summary={
            "best_params": result.best_params,
            "best_recipe_knobs": result.best_recipe_knobs,
            "best_score": result.best_score,
            "best_std": result.best_std,
            "ranking_metric": result.ranking_metric,
            "n_trials": len(result.trials),
        },
    )
    return result


def evolutionary_search(
    session,
    estimator: Any,
    *,
    param_space: dict[str, Any] | None = None,
    recipe_space: dict[str, Any] | None = None,
    population_size: int = 12,
    n_generations: int = 5,
    elite_size: int = 2,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    max_evaluations: int | None = None,
    random_state: int | None = 42,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    cv: int | Any = 5,
    cv_strategy: Literal['auto', 'kfold', 'stratified', 'group', 'stratified_group', 'time'] = "auto",
    ranking_metric: str | None = None,
    groups: pd.Series | None = None,
    preprocess: PreprocessRecipe | None = None,
    allow_session_global_preprocess: bool = False,
    refit: bool = True,
) -> SearchResult:
    """Genetic-algorithm HPO with leakage-safe train-fold CV.

    In-tree NumPy GA (population, tournament selection, crossover/mutation,
    elitism) — not random search renamed, not NAS, not a swarm zoo.
    ``param_space`` / ``recipe_space`` use the same declare-style float/int/
    categorical forms as Optuna declare spaces (dicts only; no trial callables).
    """
    session.assert_can_fit("train")
    result = run_evolutionary_search(
        session.dataset,
        session._split_plan,
        estimator,
        param_space=param_space,
        recipe_space=recipe_space,
        population_size=population_size,
        n_generations=n_generations,
        elite_size=elite_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        max_evaluations=max_evaluations,
        random_state=random_state,
        task=task,
        cv=cv,
        cv_strategy=cv_strategy,
        ranking_metric=ranking_metric,
        groups=groups,
        preprocess=preprocess,
        session_preprocess_applied=session._session_preprocess_applied(),
        allow_session_global_preprocess=allow_session_global_preprocess,
        refit=refit,
    )
    session._last_search = result
    if refit and result.refit_result is not None:
        session._fit_result = result.refit_result
    session._record(
        "evolutionary_search",
        {
            "estimator": type(estimator).__name__,
            "population_size": population_size,
            "n_generations": n_generations,
            "elite_size": elite_size,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "max_evaluations": max_evaluations,
            "random_state": random_state,
            "task": task,
            "cv": cv if isinstance(cv, int) else type(cv).__name__,
            "cv_strategy": cv_strategy,
            "ranking_metric": ranking_metric,
            "refit": refit,
            "has_param_space": param_space is not None,
            "has_recipe_space": recipe_space is not None,
        },
        result_summary={
            "best_params": result.best_params,
            "best_recipe_knobs": result.best_recipe_knobs,
            "best_score": result.best_score,
            "best_std": result.best_std,
            "ranking_metric": result.ranking_metric,
            "n_trials": len(result.trials),
            "n_evaluations": None
            if not isinstance(result.study, dict)
            else result.study.get("n_evaluations"),
        },
    )
    return result


def save_model(session, path: str | Path) -> Path:
    """Persist the last fitted estimator bundle.

    This stores the estimator and feature contract only. Prefer
    :meth:`save_pipeline` when impute/encode/scale plans must travel with
    the model.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    destination = save_fit_result(path, session._fit_result)
    session._record("save_model", {"path": str(destination)})
    return destination


def load_model(session, path: str | Path) -> Session:
    """Load a previously saved fitted estimator bundle into this session."""
    session._fit_result = load_fit_result(path)
    session._record("load_model", {"path": str(path)})
    return session


def save_pipeline(
    session,
    path: str | Path,
    *,
    evaluate_partition: Literal['train', 'validation', 'test'] | None = "test",
    title: str | None = None,
) -> Path:
    """Persist fitted preprocess plans, estimator, and a model card.

    Layout includes ``model.joblib``, ``plans.joblib``, ``meta.json``, and
    ``model_card`` JSON/Markdown. Persists impute, encode, scale, dates,
    outliers, binning, feature selection, and resample (lineage) plans when
    present. This is not a Session checkpoint: data, splits, and full
    history remain checkpoint concerns.

    Parameters
    ----------
    path:
        Destination directory.
    evaluate_partition:
        If set and a split exists, attach metrics from that partition to
        the model card. Use ``None`` to skip evaluation at save time.
    title:
        Optional model-card title.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    metrics: dict[str, dict[str, float]] = {}
    notes = [
        "Pipeline bundle stores fitted preprocess plans and the estimator feature contract.",
        "It does not embed a Session checkpoint or the raw training frame.",
        "Resample plans are lineage metadata only and are not reapplied at inference.",
    ]
    if evaluate_partition is not None and session._split_plan is not None:
        try:
            evaluation = evaluate_estimator(
                session.dataset,
                session._split_plan,
                session._fit_result,
                partition=evaluate_partition,
            )
            metrics[evaluate_partition] = dict(evaluation.metrics)
        except (ValidationError, ValueError, TypeError) as exc:
            notes.append(f"Evaluation at save time was skipped: {exc}")
    from buildml.pipeline.bundle import CHECKPOINT_COMPATIBILITY
    from buildml.pipeline.card import build_model_card, load_model_card

    preprocess_summary = session._preprocess_summary()
    card = build_model_card(
        fit_result=session._fit_result,
        dataset_schema=session.dataset.schema.to_dict(),
        preprocess_summary=preprocess_summary,
        history=session._history,
        metrics=metrics,
        title=title,
        notes=notes,
        lineage={
            "artifact": "pipeline_bundle",
            "contains_checkpoint": False,
            "contains_raw_dataset": False,
            "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
            "plans_present": sorted(
                (key for key, value in preprocess_summary.items() if value is not None)
            ),
        },
    )
    destination = save_pipeline_bundle(
        path,
        fit_result=session._fit_result,
        impute_plan=session._impute_plan,
        encode_plan=session._encode_plan,
        scale_plan=session._scale_plan,
        date_plan=session._date_plan,
        outlier_plan=session._outlier_plan,
        binning_plan=session._binning_plan,
        feature_select_plan=session._feature_select_plan,
        text_plan=session._text_plan,
        reduce_plan=session._reduce_plan,
        custom_plan=session._custom_plan,
        resample_plan=session._resample_plan,
        model_card=card,
        dataset_schema=session.dataset.schema.to_dict(),
        roles={k: v.value for k, v in session.dataset.roles.items()},
        history=session._history,
        metrics=metrics,
        title=title,
    )
    session._model_card = load_model_card(destination)
    session._last_pipeline_path = Path(destination)
    session._record(
        "save_pipeline",
        {
            "path": str(destination),
            "evaluate_partition": evaluate_partition,
            "has_impute": session._impute_plan is not None,
            "has_encode": session._encode_plan is not None,
            "has_scale": session._scale_plan is not None,
            "has_dates": session._date_plan is not None,
            "has_outliers": session._outlier_plan is not None,
            "has_binning": session._binning_plan is not None,
            "has_feature_select": session._feature_select_plan is not None,
            "has_text": session._text_plan is not None,
            "has_reduce": session._reduce_plan is not None,
            "has_custom": session._custom_plan is not None,
            "has_resample": session._resample_plan is not None,
        },
        result_summary={"path": str(destination), "metrics_partitions": list(metrics)},
    )
    return destination


def load_pipeline(session, path: str | Path) -> Session:
    """Load a pipeline bundle (estimator + preprocess plans + model card).

    Restores :attr:`fit_result`, preprocess plan attributes, and
    :attr:`model_card`. Does not replace the dataset or split; attach
    compatible data separately (or via :meth:`checkpoint_load`).
    """
    bundle = load_pipeline_bundle(path)
    session._fit_result = bundle.fit_result
    session._impute_plan = bundle.impute_plan
    session._encode_plan = bundle.encode_plan
    session._scale_plan = bundle.scale_plan
    session._date_plan = bundle.date_plan
    session._outlier_plan = bundle.outlier_plan
    session._binning_plan = bundle.binning_plan
    session._feature_select_plan = bundle.feature_select_plan
    session._text_plan = bundle.text_plan
    session._reduce_plan = bundle.reduce_plan
    session._custom_plan = bundle.custom_plan
    session._resample_plan = bundle.resample_plan
    session._model_card = bundle.model_card
    session._record(
        "load_pipeline",
        {
            "path": str(path),
            "estimator": bundle.fit_result.to_dict().get("estimator"),
            "has_model_card": bundle.model_card is not None,
            "bundle_format": bundle.bundle_format,
            "plans_format": bundle.plans_format,
            "plans_present": [
                name
                for name, plan in (
                    ("impute", bundle.impute_plan),
                    ("encode", bundle.encode_plan),
                    ("scale", bundle.scale_plan),
                    ("dates", bundle.date_plan),
                    ("outliers", bundle.outlier_plan),
                    ("binning", bundle.binning_plan),
                    ("feature_select", bundle.feature_select_plan),
                    ("text", bundle.text_plan),
                    ("reduce", bundle.reduce_plan),
                    ("custom", bundle.custom_plan),
                    ("resample", bundle.resample_plan),
                )
                if plan is not None
            ],
        },
    )
    return session


def predict_from_pipeline(
    session,
    path: str | Path,
    data: Dataset | pd.DataFrame | None = None,
    *,
    roles: dict[str, ColumnRole | str] | None = None,
    return_proba: bool = False,
    apply_plans: bool = True,
) -> PipelinePredictResult:
    """Score a frame through a saved pipeline bundle in one call.

    Parameters
    ----------
    path:
        Pipeline bundle directory.
    data:
        Score frame. Defaults to this session's dataset when omitted.
    roles:
        Optional roles when ``data`` is a bare DataFrame.
    return_proba:
        Request class probabilities when the estimator supports them.
    apply_plans:
        Replay fitted preprocess plans from the bundle before predict
        (default True).

    Notes
    -----
    Does not mutate this session's dataset or fit_result. Prefer this for
    inference-only scoring of new frames.
    """
    if data is None:
        if session._dataset is None:
            raise ValidationError("No dataset attached. Ingest data or pass data=...")
        score_data: Dataset | pd.DataFrame = session.dataset
    else:
        score_data = data
    result = run_predict_from_pipeline(
        path, score_data, roles=roles, return_proba=return_proba, apply_plans=apply_plans
    )
    session._record(
        "predict_from_pipeline",
        {
            "path": str(path),
            "return_proba": return_proba,
            "apply_plans": apply_plans,
            "n_rows": result.n_rows,
        },
        warnings=result.warnings,
        result_summary=result.to_dict(),
    )
    return result


def prepare_design_matrix(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "train",
    columns: list[str] | tuple[str, ...] | None = None,
    sample_rows: int | None = None,
    random_state: int | None = 0,
) -> MaterializePrepResult:
    """Project/sample columns via the active engine before sklearn materialize.

    When ``columns`` is omitted and a split exists, prepares the partition
    feature+target design matrix. Disclosures record projection and any
    sampling; sklearn still requires an in-memory matrix.
    """
    if columns is not None:
        result = prepare_design_frame(
            session.dataset,
            columns,
            sample_rows=sample_rows,
            random_state=random_state,
            context=f"session prepare_design_matrix ({partition})",
        )
    else:
        session.assert_can_fit("train")
        assert session._split_plan is not None
        result = materialize_partition_design(
            session.dataset,
            session._split_plan,
            partition,
            sample_rows=sample_rows,
            random_state=random_state,
        )
    session._record(
        "prepare_design_matrix",
        {
            "partition": partition,
            "sample_rows": sample_rows,
            "engine": result.engine,
            "sampled": result.sampled,
        },
        result_summary=result.to_dict(),
    )
    return result


def calibration(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "test",
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Probability calibration diagnostics for the fitted classifier.

    Returns Brier/ECE, reliability curve points, and interpretation tips.
    Optional figure/HTML export uses the viz extra.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    report = calibration_report(
        session.dataset,
        session._split_plan,
        session._fit_result,
        partition=partition,
        export_figures=export_figures,
        export_html=export_html,
    )
    session._last_diagnostic = report
    session._record(
        "calibration",
        {"partition": partition, "export_figures": export_figures, "export_html": export_html},
        result_summary=report.to_dict(),
    )
    return report


def tune_threshold(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "test",
    fp_cost: float | None = None,
    fn_cost: float | None = None,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Sweep binary decision thresholds with precision/recall/F1 and optional costs.

    Parameters
    ----------
    partition:
        Rows used for the sweep. Prefer ``validation`` when selecting a
        policy; use ``test`` only to confirm a fixed threshold.
    fp_cost, fn_cost:
        Non-negative false-positive / false-negative costs. Provide both to
        minimize expected cost on the scored partition.
    tp_benefit, tn_benefit:
        Optional benefits subtracted from cost for true positives / negatives.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    report = threshold_report(
        session.dataset,
        session._split_plan,
        session._fit_result,
        partition=partition,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        export_figures=export_figures,
        export_html=export_html,
    )
    session._last_diagnostic = report
    session._record(
        "tune_threshold",
        {
            "partition": partition,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "tp_benefit": tp_benefit,
            "tn_benefit": tn_benefit,
            "export_figures": export_figures,
            "export_html": export_html,
        },
        result_summary=report.to_dict(),
    )
    return report


def learning_curve(
    session,
    estimator: Any,
    *,
    task: Literal['classification', 'regression', 'auto'] = "auto",
    cv: int = 5,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Compute learning curves on the training partition."""
    report = learning_curve_report(
        session.dataset,
        session._split_plan,
        estimator,
        task=task,
        cv=cv,
        export_figures=export_figures,
        export_html=export_html,
    )
    session._last_diagnostic = report
    session._record(
        "learning_curve",
        {
            "estimator": type(estimator).__name__,
            "task": task,
            "cv": cv,
            "export_figures": export_figures,
            "export_html": export_html,
        },
        result_summary=report.to_dict(),
    )
    return report


def feature_importance(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "test",
    n_repeats: int = 8,
    export_figures: str | Path | None = None,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Permutation feature importance on a holdout partition."""
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    report = permutation_importance_report(
        session.dataset,
        session._split_plan,
        session._fit_result,
        partition=partition,
        n_repeats=n_repeats,
        export_figures=export_figures,
        export_html=export_html,
    )
    session._last_diagnostic = report
    session._record(
        "feature_importance",
        {
            "partition": partition,
            "n_repeats": n_repeats,
            "export_figures": export_figures,
            "export_html": export_html,
        },
        result_summary=report.to_dict(),
    )
    return report


def error_slices(
    session,
    *,
    by: str | Sequence[str],
    partition: Literal['train', 'validation', 'test'] = "test",
    max_segments: int = 20,
    min_segment_n: int = 5,
    export_html: str | Path | None = None,
) -> DiagnosticReport:
    """Slice prediction errors by one or more columns on a partition.

    Notes
    -----
    Observational only: segment gaps are not fairness proof. Prefer
    validation for exploration and keep test for a final estimate.
    Segments with ``n < min_segment_n`` are listed under ``small_segments``.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    report = segment_error_report(
        session.dataset,
        session._split_plan,
        session._fit_result,
        by=by,
        partition=partition,
        max_segments=max_segments,
        min_segment_n=min_segment_n,
        export_html=export_html,
    )
    session._last_diagnostic = report
    session._record(
        "error_slices",
        {
            "by": by if isinstance(by, str) else list(by),
            "partition": partition,
            "max_segments": max_segments,
            "min_segment_n": min_segment_n,
            "export_html": export_html,
        },
        result_summary=report.to_dict(),
    )
    return report
