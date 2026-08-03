"""Classical fit / evaluate / CV / search / pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

import pandas as pd

from buildml.session._imports import (
    ColumnRole,
    CVScoreResult,
    Dataset,
    DiagnosticReport,
    EvaluateResult,
    MaterializePrepResult,
    ModelComparison,
    NestedCVResult,
    PipelinePredictResult,
    PlotBoardReport,
    PreprocessRecipe,
    SearchResult,
    ValidationError,
    build_eval_plot_board,
    calibration_report,
    compare_estimators,
    evaluate_estimator,
    fit_estimator,
    learning_curve_report,
    load_fit_result,
    load_pipeline_bundle,
    materialize_partition_design,
    permutation_importance_report,
    predict_estimator,
    prepare_design_frame,
    run_cv_score,
    run_evolutionary_search,
    run_grid_search,
    run_nested_cv_score,
    run_optuna_search,
    run_predict_from_pipeline,
    run_randomized_search,
    save_fit_result,
    save_pipeline_bundle,
    segment_error_report,
    threshold_report,
)


def fit(
    session, estimator: Any, *, task: Literal['classification', 'regression', 'auto'] = "auto"
) -> "Session":
    """Train a model on the training rows.

    This is the step everything before it was preparing for. BuildML reads
    the column roles to work out what the inputs and the target are, hands
    the training rows to your estimator, and stores the fitted model on the
    session so :meth:`predict`, :meth:`evaluate`, and :meth:`save_pipeline`
    can find it.

    You supply the estimator yourself — any object with scikit-learn's
    ``fit`` and ``predict`` methods works, including XGBoost, LightGBM, and
    CatBoost models. BuildML does not maintain a private registry of model
    names, so anything installed in your environment is available and you
    configure it in the usual way.

    Before fitting, the training scope is checked: if there is no split, or
    an earlier step tried to widen the fit beyond the train rows, this
    raises rather than quietly producing an inflated score.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator instance, already configured with whatever
        hyperparameters you want — ``RandomForestClassifier(max_depth=6)``,
        not the class itself. BuildML fits a reference to this object, so
        it is the one that ends up in the pipeline.
    task:
        Whether this is ``'classification'`` or ``'regression'``. The
        default ``'auto'`` infers it from the target column, which is
        correct nearly always; state it explicitly when the target is
        numeric but really represents classes, or when integer class labels
        would otherwise be read as a quantity to predict.

    Returns
    -------
    Session
        ``self``, so the fit chains into :meth:`evaluate`. The fitted model
        and its metadata are on :attr:`fit_result`.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split exists. Fitting on everything leaves nothing to honestly
        measure against, so BuildML refuses rather than allowing it.
    ~buildml.core.errors.ValidationError
        No target column is assigned, features are still non-numeric or
        contain missing values, or the target does not fit the requested
        task.

    Notes
    -----
    **Leakage:** Fits on train only. Call after split and preparation.

    Only the training rows reach the estimator. Validation and test rows
    stay untouched until you ask for them by name, which is what makes the
    eventual test score meaningful.

    If a ``weight`` role column is assigned and the estimator supports
    sample weights, it is passed through, so rare-but-important rows can be
    given more influence without resampling.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {"x": [0.1, 0.9, 0.2, 0.8], "y": [0, 1, 0, 1]}
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5, stratify=True)
    >>> _ = session.fit(LogisticRegression())
    >>> session.fit_result.task
    'classification'

    Any scikit-learn-compatible estimator works the same way:

    >>> from xgboost import XGBClassifier  # doctest: +SKIP
    >>> _ = session.fit(XGBClassifier(n_estimators=200))  # doctest: +SKIP

    See Also
    --------
    Session.evaluate : Measure what the fitted model actually achieves.
    Session.cv_score : A more stable estimate than a single holdout.
    Session.grid_search : Choose hyperparameters instead of guessing.
    Session.run_automl : Let a search pick the estimator for you.
    """
    session.assert_can_fit("train")
    session._fit_result = fit_estimator(session.dataset, session._split_plan, estimator, task=task)
    session._record(
        "fit",
        {"estimator": type(estimator).__name__, "task": task},
        result_summary=session._fit_result.to_dict(),
    )
    return cast("Session", session)
def predict(
    session,
    *,
    partition: Literal['train', 'validation', 'test'] = "test",
    return_proba: bool = False,
) -> pd.Series | pd.DataFrame:
    """Run the fitted model over one partition and return its predictions.

    Use this when you want the predictions themselves — to inspect them,
    join them back to identifiers, or compute something BuildML does not
    provide. If what you want is a score, :meth:`evaluate` computes metrics
    and diagnostics in one call instead.

    The features are rebuilt exactly as they were at fit time, using the
    column order recorded on :attr:`fit_result`, so the model receives what
    it expects.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        Which rows to score: ``'test'`` (the default) for the honest
        estimate, ``'validation'`` while tuning, or ``'train'`` to compare
        against the others. A model that scores far better on train than on
        test is overfitting, and comparing the two is how you see it.
    return_proba:
        When True, return each class's predicted probability rather than a
        single chosen label. Probabilities are what you need to move a
        decision threshold (:meth:`tune_threshold`), to rank cases by risk,
        or to check calibration. Ignored by estimators that do not expose
        ``predict_proba``.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        A Series of predicted labels or values, indexed to match the
        partition's rows. With ``return_proba=True`` on a classifier, a
        DataFrame with one column per class instead.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted yet, no split exists, or the named
        partition is not part of the current split.

    Notes
    -----
    Predicting on ``'train'`` tells you how well the model memorised, not
    how well it generalises. It is a useful diagnostic and a misleading
    headline number.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5, stratify=True)
    >>> _ = session.fit(LogisticRegression())
    >>> len(session.predict(partition="test"))
    2

    Get probabilities when you intend to choose your own cut-off:

    >>> proba = session.predict(partition="test", return_proba=True)
    >>> proba.shape[1]
    2

    See Also
    --------
    Session.evaluate : Metrics and diagnostics rather than raw output.
    Session.predict_from_pipeline : Score new data outside this session.
    Session.tune_threshold : Pick the cut-off these probabilities feed.
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
    """Measure the fitted model, and explain what the measurement means.

    A single accuracy figure hides more than it reveals. 95% accuracy is
    excellent when the classes are balanced and worthless when 95% of rows
    belong to one class — the same number, opposite conclusions. So this
    returns a card rather than a score: several complementary metrics, the
    diagnostics behind them, and written recommendations about what to look
    at next.

    For classification you get accuracy and balanced accuracy, weighted
    precision and recall, macro and weighted F1, and — where the estimator
    exposes probabilities — ROC-AUC, average precision, and log loss, plus
    the confusion matrix showing which classes are being mistaken for
    which. Precision and recall matter most when errors are asymmetric:
    precision is how often a positive prediction is right, recall is how
    many of the real positives you caught, and improving one generally
    costs the other. Balanced accuracy is the one to read on imbalanced
    data, because plain accuracy is dominated by the majority class.

    For regression you get error magnitudes (MAE, RMSE) alongside R², plus
    residual diagnostics. MAE is the average miss in the target's own
    units. RMSE punishes large misses disproportionately, so a gap between
    the two means a few predictions are badly wrong. R² is the share of
    variance explained, and it can be negative — that simply means the
    model does worse than always predicting the mean.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        Which rows to score. ``'test'`` is the honest estimate and should
        be used once, at the end; ``'validation'`` is for the comparisons
        you make while deciding. Evaluating on ``'train'`` alongside test
        is the standard way to detect overfitting.
    export_figures:
        Directory to write diagnostic figures into. Implies plotting, and
        requires ``pip install 'buildml[viz]'``.
    export_html:
        Path for a self-contained HTML report of the same figures — handy
        to attach to a review or send to someone without a Python
        environment. Also implies plotting.
    include_plots:
        Build the diagnostic plot board without writing it anywhere. The
        board is stored on :attr:`last_plot_board` either way.

    Returns
    -------
    ~buildml.model.supervised.EvaluateResult
        The evaluation card: ``metrics``, ``diagnostics`` (confusion
        matrix, residual summaries, plot paths), the ``n_rows`` scored, and
        ``recommendations``. Call its ``show()`` method for a readable
        digest instead of reading the dictionaries by hand.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted, no split exists, or the named partition
        is not part of the current split.
    ~buildml.core.errors.MissingExtraError
        Plots were requested without ``buildml[viz]`` installed.

    Notes
    -----
    Every glance at the test set spends a little of its independence. If
    you evaluate on test, adjust something, and evaluate again, the test
    score has quietly become a tuning signal and is no longer the unbiased
    estimate you think it is. Tune against validation or
    :meth:`cv_score`, and keep test for the end.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8], "y": [0, 1, 0, 1]})
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5, stratify=True)
    >>> result = session.fit(LogisticRegression()).evaluate()
    >>> result.task
    'classification'
    >>> "accuracy" in result.metrics and "balanced_accuracy" in result.metrics
    True

    Compare train against test to see whether the model is overfitting:

    >>> train_score = session.evaluate(partition="train").metrics["accuracy"]
    >>> test_score = session.evaluate(partition="test").metrics["accuracy"]

    Produce a shareable report with figures:

    >>> _ = session.evaluate(export_html="reports/eval.html")  # doctest: +SKIP

    See Also
    --------
    Session.eval_plots : Build the diagnostic board on its own.
    Session.calibration : Check whether probabilities mean what they say.
    Session.error_slices : Find the subgroups where the model fails.
    Session.compare_models : Put several candidates side by side.
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
    """Draw the standard diagnostic charts for a fitted model, in one call.

    Numbers tell you how well a model does; pictures tell you how it fails.
    This assembles the panels worth looking at for the model you actually
    fitted, skipping the ones that do not apply rather than erroring.

    Depending on the task and what the estimator supports, the board can
    include the confusion matrix or residual plots, the ROC and
    precision-recall curves, a calibration curve showing whether predicted
    probabilities match observed frequencies, the precision-recall
    trade-off across thresholds, a learning curve indicating whether more
    data would help, and permutation importance ranking the features the
    model relied on.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        Which rows the diagnostics describe. ``'test'`` shows the
        behaviour you will get in deployment; ``'train'`` next to it
        reveals overfitting.
    include_learning_curve:
        Add the learning curve. It refits the model on increasing
        subsamples, so it is the slowest panel — turn it off for a quick
        look. Read it as: converged curves mean more data will not help,
        a persistent gap means it will.
    include_importance:
        Add permutation importance, which measures how much the score drops
        when a feature's values are shuffled. Slower than reading the
        model's built-in importances, but model-agnostic and harder to
        mislead.
    n_importance_repeats:
        How many times each feature is shuffled. More repeats give a
        steadier ranking at proportional cost; the default trades a little
        noise for speed.
    learning_curve_cv:
        Fold count used at each learning-curve sample size. Kept low by
        default because the curve refits at every point.
    export_figures:
        Directory to write the individual figures into. ``None`` keeps them
        in memory only.
    export_html:
        Path for a single self-contained HTML page holding every panel —
        the artefact to attach to a review.
    show:
        Display the figures interactively, for notebook use.

    Returns
    -------
    ~buildml.model.plot_boards.PlotBoardReport
        The board: paths to any figures written, which panels were
        ``skipped`` and why, and an ``interpretation`` explaining what each
        panel shows. Also stored on :attr:`last_plot_board`.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted, or no split exists.
    ~buildml.core.errors.MissingExtraError
        Plotting requires ``pip install 'buildml[viz]'``.

    Notes
    -----
    Delegates to :func:`buildml.model.plot_boards.build_eval_plot_board`.

    Panels degrade gracefully. A model without ``predict_proba`` has no
    ROC or calibration curve, and a multi-class target has no single
    precision-recall trade-off; those panels are listed in ``skipped``
    with a reason rather than raising.

    Examples
    --------
    >>> board = session.eval_plots(export_html="reports/board.html")  # doctest: +SKIP
    >>> board.skipped  # doctest: +SKIP
    ['roc_curve: estimator has no predict_proba']

    See Also
    --------
    Session.evaluate : Metrics, with these plots available inline.
    Session.calibration : The calibration panel on its own.
    Session.learning_curve : The learning-curve panel on its own.
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
    """Try several models on the same data and rank what you get.

    "Which algorithm should I use?" has no answer in the abstract — it
    depends on your data, and the reliable way to find out is to try a few.
    This fits each estimator on the training rows, evaluates them all on
    the same partition, and returns them ranked, so the comparison is
    genuinely like-for-like.

    A sensible starting set is one linear model, one tree ensemble, and one
    gradient-boosting model. That covers very different assumptions about
    the data, and the spread between them tells you a lot: if the linear
    model keeps up, your relationships are mostly additive and you should
    probably prefer it for the interpretability.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimators:
        Label to unfitted estimator instance. The labels are yours and
        appear in the ranking, so name them for what distinguishes them
        (``"rf_depth6"``) rather than by class.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'`` to infer it
        from the target. Every estimator is treated as the same task.
    partition:
        Which rows to score on. Use ``'validation'`` while choosing —
        ranking candidates on ``'test'`` and then reporting the winner's
        test score overstates it, because the winner was selected using
        that very number.
    ranking_metric:
        Which metric orders the table. ``None`` uses the task default.
        Choose deliberately when errors are asymmetric: ranking by accuracy
        on imbalanced data will happily crown a model that never predicts
        the rare class.

    Returns
    -------
    ~buildml.model.compare.ModelComparison
        The ranked comparison, holding each model's metrics, the ordering,
        and the metric used to produce it.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split exists, so there is nothing to compare on.
    ~buildml.core.errors.ValidationError
        ``estimators`` is empty, no target is assigned, or the features are
        not yet numeric and complete.

    Notes
    -----
    Each model is scored on a single fixed partition, so small differences
    between them are within noise. When two candidates finish close
    together, confirm with :meth:`cv_score` before declaring a winner —
    a one-point gap on a few hundred rows frequently reverses on a
    different split.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> comparison = session.compare_models(
    ...     {
    ...         "logistic": LogisticRegression(max_iter=500),
    ...         "forest": RandomForestClassifier(random_state=0),
    ...     },
    ...     partition="validation",
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.cv_score : Confirm a close result across several folds.
    Session.run_automl : Search a space of models rather than a shortlist.
    Session.fit_voting : Combine candidates instead of picking one.
    """
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
    """Score a model across several rotating holdouts, not just one.

    A single train/test split gives you one number, and that number depends
    on which rows happened to land in test. On a few thousand rows the
    swing between two random splits is easily a couple of percentage
    points — enough to pick the wrong model.

    Cross-validation removes that luck. The training rows are divided into
    ``cv`` folds; the model is fitted ``cv`` times, each time holding out a
    different fold and scoring on it. You end up with ``cv`` scores instead
    of one, and their spread is as informative as their average: a high
    mean with a wide spread means the result is fragile, not good.

    The session's test partition takes no part in any of this. Folds are
    cut from the training rows only, so test stays untouched for the final
    measurement.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator instance. It is cloned for each fold, so the
        object you pass is never itself fitted and can be reused.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'`` to infer it
        from the target.
    cv:
        How many folds, or a scikit-learn splitter object for full control.
        Five is the usual compromise: more folds train on more data per
        fold and give a less biased estimate, at proportionally more time.
    cv_strategy:
        How rows are assigned to folds when ``cv`` is a number. ``'auto'``
        reads the column roles and picks for you. ``'stratified'``
        preserves class balance in every fold, which matters for imbalanced
        classification. ``'group'`` keeps an entity's rows in the same fold
        — the cross-validation equivalent of :meth:`group_split`.
        ``'stratified_group'`` does both. ``'time'`` only ever trains on
        folds earlier than the one being scored. Choosing wrongly here
        recreates the leakage the split was designed to prevent.
    scoring_metric:
        Which metric the summary reports. ``None`` uses the task default.
    groups:
        Group labels aligned to the training rows, for the group-aware
        strategies. ``None`` uses the ``group``-role column.
    preprocess:
        A :class:`~buildml.preprocess.fold.PreprocessRecipe` refitted
        inside every fold. This is the leakage-correct way to include
        preprocessing in a cross-validated estimate: the scaler and encoder
        are learned from that fold's training rows and applied to its
        held-out rows, exactly as they would be in production.
    allow_session_global_preprocess:
        Permit cross-validation to proceed even though session-wide
        preprocessing already ran. Off by default, and the refusal is the
        point — see the note below.

    Returns
    -------
    ~buildml.model.selection.CVScoreResult
        Per-fold scores with their mean and standard deviation, plus an
        ``interpretation``, the ``limitations`` of the estimate, and
        ``recommendations``. Also stored on :attr:`last_cv`.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        Session-wide preprocessing already ran and
        ``allow_session_global_preprocess`` was not set.
    ~buildml.core.errors.ValidationError
        No split exists, the requested strategy needs a role column that is
        not assigned, or a fold would be empty.

    Notes
    -----
    **Leakage:** If Session impute/encode/scale/text/reduce already ran, CV
    refuses unless ``allow_session_global_preprocess=True``. Prefer
    re-ingesting unpoisoned data, then fold-local recipes (including
    ``text`` and ``reduce``) for selection claims that include
    preprocessing. Custom transforms and resample stay Session-global.

    The reason for that refusal is worth understanding. Calling
    :meth:`scale` fits one scaler across all the training rows. If you then
    cross-validate, each fold's held-out rows were already involved in
    computing that scaler's mean, so every fold score is slightly
    optimistic. The recipe mechanism exists to avoid this: it defers
    preprocessing until the fold boundary is known. Overriding the refusal
    does not fix the estimate, it only silences the warning about it.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> result = session.cv_score(
    ...     RandomForestClassifier(random_state=0), cv=5, cv_strategy="stratified"
    ... )  # doctest: +SKIP
    >>> result.mean_metrics["accuracy"], result.std_metrics["accuracy"]  # doctest: +SKIP
    (0.884, 0.021)

    With preprocessing done correctly, inside each fold:

    >>> from buildml.preprocess.fold import PreprocessRecipe
    >>> recipe = PreprocessRecipe(impute="median", scale="standard")
    >>> result = session.cv_score(
    ...     RandomForestClassifier(), preprocess=recipe
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.nested_cv_score : When you are also tuning hyperparameters.
    Session.grid_search : Search a space, using this scoring underneath.
    Session.evaluate : The single-holdout estimate this replaces.
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
    """Estimate how well your *tuning procedure* generalises, not just one model.

    There is a subtle trap in the usual workflow. You run :meth:`grid_search`,
    it reports the cross-validated score of the winning configuration, and
    you quote that number as your expected performance. But the winner was
    chosen *because* it scored well on those folds, so its score is
    optimistically biased — you picked the luckiest configuration and then
    reported its luck as skill. On a large search space the inflation can be
    several points.

    Nested cross-validation removes the bias by giving the search its own
    private data. The rows are split into outer folds. Within each outer
    fold's training portion, an independent inner search picks the best
    configuration; that winner is then scored once on the outer fold's
    held-out rows, which the search never saw. Averaging those outer scores
    gives an honest estimate of what "run my tuning procedure on data like
    this" is worth.

    Note what is being estimated: the procedure, not a single model. Each
    outer fold may crown a different winner, and that is fine — the spread
    across folds tells you how stable your tuning is. To get a model to
    deploy, run :meth:`grid_search` (or a sibling) once on the full training
    set afterwards, and quote the nested score as its expected performance.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator, cloned fresh for every candidate in both
        loops.
    param_grid:
        Exhaustive estimator search space, ``{"max_depth": [3, 5, 8]}``.
        Mutually exclusive with ``param_distributions``. Optional when you
        are only searching recipe knobs.
    param_distributions:
        Sampled estimator search space for a randomized inner search.
    recipe_grid:
        Search space over preprocessing knobs — ``select_k``, ``n_bins``,
        and friends — refit inside each fold. Requires ``preprocess``.
    recipe_distributions:
        Sampled counterpart to ``recipe_grid``.
    param_space:
        Optuna search space for the estimator, used when ``inner_search``
        is ``'optuna'``. Declare-style dicts also drive the evolutionary
        search. Optuna needs ``pip install 'buildml[optuna]'``.
    recipe_space:
        Optuna or evolutionary search space for the recipe knobs.
    inner_search:
        Which search runs inside each outer fold. ``'auto'`` infers it from
        which spaces you supplied. Note the cost: the inner search runs once
        per outer fold, so an exhaustive grid multiplies quickly.
    n_iter:
        Candidates sampled per outer fold when the inner search is
        randomized.
    n_trials:
        Optuna trials per outer fold; doubles as ``max_evaluations`` for the
        evolutionary search.
    population_size:
        Candidates per generation for the evolutionary inner search.
    n_generations:
        Generations the evolutionary inner search runs for.
    random_state:
        Seed for fold assignment and candidate sampling, so the estimate
        reproduces.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'`` to infer from
        the target.
    outer_cv:
        Number of outer folds, or a scikit-learn splitter. These folds
        produce the reported estimate.
    inner_cv:
        Number of inner folds, or a splitter. Kept smaller than ``outer_cv``
        by default because it runs many more times.
    cv_strategy:
        How rows are assigned to folds. ``'stratified'`` preserves class
        balance, ``'group'`` keeps related rows together, ``'time'``
        respects chronology. ``'auto'`` picks from the data and roles.
    scoring_metric:
        Metric the inner search optimises and the outer loop reports.
        Defaults to a sensible choice for the task.
    groups:
        Group labels for the group-aware strategies.
    preprocess:
        A :class:`~buildml.preprocess.fold.PreprocessRecipe` refit inside
        every fold of both loops. This is what keeps preprocessing honest:
        imputation values and scalers are learned from fold-training rows
        only.
    allow_session_global_preprocess:
        Permit running against session-wide preprocessing that was fit
        before splitting. Off by default because it leaks; the guard exists
        for deliberate exceptions.
    warm_start_studies:
        Share one Optuna study across outer folds so later folds benefit
        from earlier trials. Faster, but the folds are no longer fully
        independent searches — the outer estimate stays valid because the
        outer-eval rows are still never scored during search.

    Returns
    -------
    ~buildml.model.selection.NestedCVResult
        ``mean_metrics`` and ``std_metrics`` hold the honest estimate and
        its fold-to-fold spread. ``outer_folds`` records each fold's chosen
        ``best_params`` and ``best_recipe_knobs``, which is where you look
        to judge whether tuning is stable or thrashing.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, no search space was supplied, mutually exclusive
        spaces were combined, or recipe knobs were given without a recipe.
    ~buildml.core.errors.MissingExtraError
        Optuna was requested without ``buildml[optuna]`` installed.

    Notes
    -----
    **Cost:** total fits are roughly ``outer_cv × inner_cv × candidates``.
    With five outer folds, three inner folds, and fifty candidates that is
    750 fits. Start with a randomized inner search and small fold counts.

    **Leakage:** the session's test and validation partitions never enter
    either loop. Both loops draw only from training rows.

    **Reading the spread:** if ``std_metrics`` is large relative to
    ``mean_metrics``, the procedure is unstable — usually too small a
    dataset for the size of the search space.

    Examples
    --------
    >>> result = session.nested_cv_score(  # doctest: +SKIP
    ...     RandomForestClassifier(),
    ...     param_distributions={"max_depth": [3, 5, 8, None]},
    ...     inner_search="randomized",
    ...     n_iter=8,
    ... )
    >>> result.mean_metrics["accuracy"]  # doctest: +SKIP

    See Also
    --------
    Session.cv_score : Honest estimate for a single fixed configuration.
    Session.grid_search : The inner search, run once, to get a deployable model.
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
    """Try every combination of the settings you list, and keep the best.

    Hyperparameters are the knobs you set before training — tree depth,
    regularisation strength, learning rate — and the right values depend on
    your data. Grid search takes the values you consider plausible, builds
    every combination of them, cross-validates each one on the training
    rows, and ranks the results.

    It is exhaustive, which is both its strength and its weakness. You are
    guaranteed to find the best point *in the grid you specified*, and you
    pay for the guarantee combinatorially: three parameters with four
    values each is 64 fits, times ``cv`` folds. Use it when the space is
    small or you already know roughly where to look. Use
    :meth:`randomized_search` or :meth:`optuna_search` when it is not.

    Recipe knobs can be searched alongside estimator parameters. Whether
    five features or fifty work better is a modelling decision like any
    other, and searching it inside the folds keeps the choice honest.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator instance supplying the defaults that the grid
        overrides.
    param_grid:
        Parameter name to the list of values to try, for example
        ``{"max_depth": [3, 6, 12]}``. Names match what the estimator's
        ``set_params`` accepts, including nested ``step__param`` forms.
    recipe_grid:
        Preprocessing knobs to search the same way, such as
        ``{"select_k": [5, 10, 20]}``. Requires ``preprocess``.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count, or a scikit-learn splitter, used to score each
        configuration.
    cv_strategy:
        How folds are formed — see :meth:`cv_score`, which describes the
        same options and the same hazards.
    ranking_metric:
        Which metric decides the winner. ``None`` uses the task default.
        This choice *is* the objective you are optimising, so pick the one
        that reflects the cost of being wrong.
    groups:
        Group labels for the group-aware strategies. ``None`` uses the
        ``group``-role column.
    preprocess:
        Fold-local recipe refit inside each fold, so the tuning estimate is
        not inflated by preprocessing that saw the held-out rows.
    allow_session_global_preprocess:
        Proceed despite session-wide preprocessing having already run. See
        the leakage note on :meth:`cv_score`.
    refit:
        When True (the default), retrain the winning configuration on the
        whole training partition and install it as :attr:`fit_result`, so
        :meth:`predict` and :meth:`evaluate` immediately use the tuned
        model. Set False to inspect the ranking before committing.

    Returns
    -------
    ~buildml.model.selection.SearchResult
        The ranked search: every trial with its score, the
        ``best_params``, ``best_score`` and ``best_std``, the winner's full
        ``best_cv`` breakdown, and the ``refit_result`` when refitting was
        requested. ``to_frame()`` renders the trials as a DataFrame. Also
        stored on :attr:`last_search`.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        Session-wide preprocessing already ran and was not explicitly
        allowed.
    ~buildml.core.errors.ValidationError
        Neither a parameter grid nor a recipe grid was supplied, a
        parameter name is not one the estimator accepts, or ``recipe_grid``
        was given without ``preprocess``.

    Notes
    -----
    Folds are cut from the training partition only; test never influences
    the ranking.

    The best cross-validation score is an optimistic estimate of the
    winner's true performance. Searching many configurations and reporting
    the maximum selects for luck as well as quality. Treat the tuned
    model's honest number as the one from :meth:`evaluate` on test, or from
    :meth:`nested_cv_score` if you want that without spending the test set.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> search = session.grid_search(
    ...     RandomForestClassifier(random_state=0),
    ...     {"max_depth": [3, 6, None], "min_samples_leaf": [1, 5]},
    ...     cv=5,
    ... )  # doctest: +SKIP
    >>> search.best_params  # doctest: +SKIP
    {'max_depth': 6, 'min_samples_leaf': 5}
    >>> search.to_frame().head()  # doctest: +SKIP

    See Also
    --------
    Session.randomized_search : Sample the space instead of enumerating it.
    Session.optuna_search : Let earlier trials guide later ones.
    Session.nested_cv_score : An unbiased estimate of a tuned model.
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
    """Sample settings at random, which usually beats an exhaustive grid.

    The result is counter-intuitive but well established: given the same
    computing budget, randomly sampling a hyperparameter space typically
    finds a better configuration than exhaustively searching a grid.

    The reason is that parameters differ enormously in how much they
    matter. A grid of four values across three parameters spends 64 fits,
    but only ever tries four distinct values of the parameter that
    actually drives performance — the other two dimensions multiply the
    cost without adding resolution where it counts. Random sampling tries
    64 *different* values of every parameter, so the important one gets
    explored properly.

    You also gain control of the budget. ``n_iter`` sets the number of
    fits directly, so adding a parameter to explore costs nothing extra.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator instance supplying the defaults being varied.
    param_distributions:
        Parameter name to either a list to choose uniformly from, or a
        ``scipy.stats`` distribution to draw from. Distributions are the
        better choice for continuous parameters, and log-uniform is the
        right shape for learning rates and regularisation strengths, where
        the interesting variation is in orders of magnitude.
    recipe_distributions:
        Preprocessing knobs sampled the same way. Requires ``preprocess``.
    n_iter:
        How many configurations to sample — your entire budget, in fits per
        fold. Start small to gauge the cost of one fit, then raise it.
    random_state:
        Seed for the sampling, so a search can be reproduced exactly.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count or splitter used to score each sampled configuration.
    cv_strategy:
        How folds are formed — see :meth:`cv_score`.
    ranking_metric:
        Which metric decides the winner. ``None`` uses the task default.
    groups:
        Group labels for the group-aware strategies.
    preprocess:
        Fold-local recipe refit inside each fold.
    allow_session_global_preprocess:
        Proceed despite session-wide preprocessing. See :meth:`cv_score`.
    refit:
        Retrain the winner on the full training partition and install it as
        :attr:`fit_result`. On by default.

    Returns
    -------
    ~buildml.model.selection.SearchResult
        The ranked trials, ``best_params``, ``best_score``, the winner's
        ``best_cv`` breakdown, and the refit model when requested. Also
        stored on :attr:`last_search`.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        Session-wide preprocessing already ran and was not explicitly
        allowed.
    ~buildml.core.errors.ValidationError
        No search space was supplied, a parameter name is not one the
        estimator accepts, or recipe distributions were given without
        ``preprocess``.

    Notes
    -----
    Same leakage contract as :meth:`grid_search`: folds stay inside train;
    the winner may be refit onto the full training partition.

    Examples
    --------
    >>> from scipy.stats import loguniform, randint
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> search = session.randomized_search(
    ...     RandomForestClassifier(random_state=0),
    ...     {"max_depth": randint(2, 20), "min_samples_leaf": randint(1, 30)},
    ...     n_iter=40,
    ... )  # doctest: +SKIP

    A learning rate should be sampled across magnitudes, not linearly:

    >>> space = {"learning_rate": loguniform(1e-3, 1e-1)}  # doctest: +SKIP

    See Also
    --------
    Session.grid_search : Exhaustive, for small well-understood spaces.
    Session.optuna_search : Adaptive, for larger budgets.
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
    """Search adaptively, letting each trial learn from the ones before it.

    Grid and random search are memoryless: the hundredth configuration is
    chosen with no knowledge of the ninety-nine already scored. Optuna
    instead builds a model of which regions of the space produce good
    results and concentrates its sampling there, while still exploring
    enough to avoid getting stuck.

    That adaptivity pays off as the budget grows. For a handful of trials
    it behaves much like random search; for fifty or more it usually
    reaches a better configuration, because it stops re-testing regions it
    has already established are poor. It also handles conditional and
    mixed discrete/continuous spaces naturally, which grids do badly.

    Requires ``pip install 'buildml[optuna]'``.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator instance supplying the defaults being tuned.
    param_space:
        The space to search, given either as a callable taking an Optuna
        ``trial`` and returning a parameter dict — full control, including
        parameters that only exist depending on others — or as a
        declarative mapping using ``float``, ``int``, and ``categorical``
        entries.
    recipe_space:
        Preprocessing knobs described the same way. Requires
        ``preprocess``.
    n_trials:
        How many configurations to evaluate. This is where Optuna earns its
        keep; below roughly twenty trials it has little history to learn
        from.
    random_state:
        Seed for the sampler, making a search reproducible.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count or splitter used to score each trial.
    cv_strategy:
        How folds are formed — see :meth:`cv_score`.
    ranking_metric:
        The metric Optuna optimises. ``None`` uses the task default.
    groups:
        Group labels for the group-aware strategies.
    preprocess:
        Fold-local recipe refit inside each fold.
    allow_session_global_preprocess:
        Proceed despite session-wide preprocessing. See :meth:`cv_score`.
    refit:
        Retrain the winner on the full training partition and install it as
        :attr:`fit_result`. On by default.

    Returns
    -------
    ~buildml.model.selection.SearchResult
        The ranked trials, ``best_params``, ``best_score``, the winner's
        ``best_cv`` breakdown, and the underlying Optuna ``study`` for
        further analysis. Also stored on :attr:`last_search`.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        Optuna is not installed.
    ~buildml.core.errors.LeakageError
        Session-wide preprocessing already ran and was not explicitly
        allowed.
    ~buildml.core.errors.ValidationError
        No search space was supplied, or a recipe space was given without
        ``preprocess``.

    Notes
    -----
    Folds are cut from the training partition only; test never influences
    the search.

    The returned ``study`` supports Optuna's own analysis tools, including
    parameter-importance ranking — often more useful than the winning
    configuration itself, since it tells you which knobs are worth tuning
    at all next time.

    Examples
    --------
    Declarative form, which covers most cases:

    >>> space = {
    ...     "max_depth": {"type": "int", "low": 2, "high": 20},
    ...     "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
    ... }
    >>> search = session.optuna_search(
    ...     estimator, param_space=space, n_trials=60
    ... )  # doctest: +SKIP

    Callable form, when one parameter depends on another:

    >>> def space(trial):  # doctest: +SKIP
    ...     kind = trial.suggest_categorical("kernel", ["linear", "rbf"])
    ...     params = {"kernel": kind}
    ...     if kind == "rbf":
    ...         params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    ...     return params

    See Also
    --------
    Session.randomized_search : Simpler, and adequate for small budgets.
    Session.evolutionary_search : Population-based, no extra dependency.
    Session.run_automl : Search over model families as well as settings.
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
    """Evolve a population of configurations across generations.

    This borrows from natural selection. A population of random
    configurations is scored; the better ones are more likely to be chosen
    as parents; parents are combined to produce offspring; offspring are
    randomly perturbed; the best few survive untouched. Repeat for
    ``n_generations``.

    The advantage over random sampling is recombination. If one
    configuration happens to have a good tree depth and another a good
    learning rate, crossover can produce a child with both — something
    independent sampling can only stumble on. That makes evolutionary
    search well suited to spaces where parameters interact.

    Compared with :meth:`optuna_search` it needs no extra dependency (the
    algorithm is implemented here in NumPy) and it explores more broadly,
    since a whole population advances at once rather than a single
    adaptive sampler. It typically needs more total evaluations to reach
    the same quality.

    The total number of fits is roughly ``population_size *
    n_generations``, each multiplied by ``cv`` folds — worth computing
    before you start.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator instance supplying the defaults being tuned.
    param_space:
        Declarative mapping of parameter name to a ``float``, ``int``, or
        ``categorical`` specification. Callables are not accepted here;
        the genetic operators need a described space they can recombine.
    recipe_space:
        Preprocessing knobs described the same way. Requires
        ``preprocess``.
    population_size:
        How many configurations exist in each generation. Larger
        populations explore more of the space per generation and cost
        proportionally more.
    n_generations:
        How many rounds of selection and recombination to run. More
        generations refine further, with diminishing returns once the
        population converges.
    elite_size:
        How many top performers pass into the next generation unchanged.
        This guarantees the best result never gets worse; setting it too
        high causes the population to converge prematurely on one region.
    crossover_rate:
        The probability that two parents are recombined rather than copied.
        High values mix aggressively, which is the mechanism that combines
        good traits from different configurations.
    mutation_rate:
        The probability that a parameter is randomly perturbed after
        crossover. This is the only source of genuinely new values once the
        population has converged; too low and the search stalls, too high
        and it degenerates into random sampling.
    tournament_size:
        How many random candidates compete to become a parent. Larger
        tournaments favour the strongest more strongly, converging faster
        but exploring less.
    max_evaluations:
        A hard cap on total configurations evaluated, stopping the run
        early once reached. ``None`` runs all generations.
    random_state:
        Seed for the stochastic operators, making the run reproducible.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count or splitter used to score each configuration.
    cv_strategy:
        How folds are formed — see :meth:`cv_score`.
    ranking_metric:
        The metric acting as the fitness function. ``None`` uses the task
        default.
    groups:
        Group labels for the group-aware strategies.
    preprocess:
        Fold-local recipe refit inside each fold.
    allow_session_global_preprocess:
        Proceed despite session-wide preprocessing. See :meth:`cv_score`.
    refit:
        Retrain the winner on the full training partition and install it as
        :attr:`fit_result`. On by default.

    Returns
    -------
    ~buildml.model.selection.SearchResult
        Every evaluated configuration with its score, the ``best_params``,
        ``best_score``, and the winner's ``best_cv`` breakdown. Also stored
        on :attr:`last_search`.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        Session-wide preprocessing already ran and was not explicitly
        allowed.
    ~buildml.core.errors.ValidationError
        No search space was supplied, a space was given as a callable
        rather than a mapping, or ``elite_size`` is not smaller than
        ``population_size``.

    Notes
    -----
    Folds are cut from the training partition only; test never influences
    the search.

    This is a plain genetic algorithm — population, tournament selection,
    crossover, mutation, elitism. It is not neural architecture search and
    not a particle swarm, and it is not random search under another name:
    the recombination step is what makes it different.

    Examples
    --------
    >>> space = {
    ...     "max_depth": {"type": "int", "low": 2, "high": 24},
    ...     "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
    ...     "booster": {"type": "categorical", "choices": ["gbtree", "dart"]},
    ... }
    >>> search = session.evolutionary_search(
    ...     estimator, param_space=space, population_size=16, n_generations=8
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.optuna_search : Adaptive single-sampler alternative.
    Session.randomized_search : Cheaper when parameters do not interact.
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
    """Save the fitted estimator and the feature contract it expects.

    This writes the model itself plus the list of feature columns and their
    order — enough to load the model back and call it, provided your data
    is already in the right shape.

    It is almost never what you want. A model trained on scaled, encoded,
    imputed data will produce nonsense if handed raw data, and this bundle
    does not carry the plans needed to prepare it. Use
    :meth:`save_pipeline` instead unless you are deliberately keeping the
    preprocessing under separate control.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        Destination path for the bundle.

    Returns
    -------
    pathlib.Path
        Where the bundle was written.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted yet.

    See Also
    --------
    Session.save_pipeline : Save the preprocessing with the model.
    Session.checkpoint_save : Save the whole session, data included.
    """
    if session._fit_result is None:
        raise ValidationError("No fitted estimator. Call fit(...) first.")
    destination = save_fit_result(path, session._fit_result)
    session._record("save_model", {"path": str(destination)})
    return destination


def load_model(session, path: str | Path, *, trusted: bool = False) -> "Session":
    """Load an estimator bundle written by :meth:`save_model`.

    Restores :attr:`fit_result` — the estimator and its feature contract —
    into this session. The dataset and split are left alone, so you can
    attach a fitted model to data you loaded separately.

    Because the bundle carries no preprocessing plans, whatever data you
    attach must already be in the exact form the model was trained on.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        Path to the bundle written by :meth:`save_model`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``self``, so the load chains into a predict.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The path holds no readable bundle.

    See Also
    --------
    Session.load_pipeline : Load a model together with its preprocessing.
    Session.predict_from_pipeline : Score new data in a single call.
    """
    session._fit_result = load_fit_result(path, trusted=trusted)
    session._record("load_model", {"path": str(path)})
    return cast("Session", session)
def save_pipeline(
    session,
    path: str | Path,
    *,
    evaluate_partition: Literal['train', 'validation', 'test'] | None = "test",
    title: str | None = None,
) -> Path:
    """Save everything needed to score new data: model, prep, and card.

    This is the artefact you deploy. A model on its own is not enough,
    because raw incoming data does not look like the matrix the model was
    trained on — the categories need the same encoding, the numbers the
    same scaling, the gaps the same fill values. Saving the fitted plans
    alongside the estimator means score-time transformation reproduces
    training exactly, months later and on a different machine.

    The bundle is a directory containing ``model.joblib``, ``plans.joblib``
    (imputation, encoding, scaling, date expansion, outlier fences,
    binning, feature selection, and resampling lineage where present),
    ``meta.json``, and a model card in both JSON and Markdown.

    This is not a session checkpoint. It carries what is needed for
    inference — no data, no split membership, no operation history. To
    resume interrupted work rather than deploy a result, use
    :meth:`checkpoint_save`.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        Destination directory, created if it does not exist.
    evaluate_partition:
        Which partition to score so the card records how the model
        performed. ``'test'`` by default. Pass ``None`` to skip, which is
        what you want when the session has no split attached.
    title:
        A human-readable name for the model card. Worth setting — this is
        what the person reading the card in six months sees first.

    Returns
    -------
    pathlib.Path
        The bundle directory that was written.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted, or ``evaluate_partition`` names a
        partition that does not exist in the current split.

    Notes
    -----
    The model card records what the model was trained on, what it scored,
    and which preprocessing travelled with it. It is generated from the
    session's own history rather than written by hand, so it cannot drift
    from what actually happened.

    Examples
    --------
    >>> path = session.save_pipeline(
    ...     "artifacts/churn_v3", title="Churn model, Q1 refresh"
    ... )  # doctest: +SKIP

    Later, in a scoring job that has never seen this session:

    >>> from buildml import Session
    >>> scorer = Session.ingest(new_rows)  # doctest: +SKIP
    >>> result = scorer.predict_from_pipeline("artifacts/churn_v3")  # doctest: +SKIP

    See Also
    --------
    Session.load_pipeline : Restore this bundle into a session.
    Session.predict_from_pipeline : Score without restoring first.
    Session.checkpoint_save : Save work in progress rather than a result.
    Session.serve_bundle : Put a saved bundle behind an HTTP endpoint.
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


def load_pipeline(session, path: str | Path, *, trusted: bool = False) -> "Session":
    """Restore a saved model together with its preprocessing.

    Reads a bundle written by :meth:`save_pipeline` and installs its
    contents on this session: the fitted estimator lands on
    :attr:`fit_result`, the preprocessing plans on their respective
    properties (:attr:`scale_plan`, :attr:`encode_plan`, and so on), and
    the model card on :attr:`model_card`.

    Your data and split are untouched. That is deliberate — it lets you
    attach a trained model to a fresh batch of rows and score them, which
    is the usual reason to load a pipeline at all. Once loaded, run
    :meth:`apply_preprocess_plans` to transform the attached data, then
    :meth:`predict`.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        The bundle directory written by :meth:`save_pipeline`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``self``, so the load chains into scoring.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The directory is not a readable pipeline bundle, or its contents
        are incomplete.

    Notes
    -----
    **Security:** pipeline bundles deserialise fitted estimators and plans
    via joblib/pickle. Only load bundles you created or fully trust —
    untrusted pickles can execute code on load.

    For one-shot scoring, :meth:`predict_from_pipeline` does the load,
    transform, and predict in a single call and leaves the session
    unchanged. Prefer that in inference jobs; prefer this when you want the
    restored plans on the session for further work.

    See Also
    --------
    Session.save_pipeline : Create the bundle this reads.
    Session.predict_from_pipeline : Load, transform, and score in one step.
    Session.checkpoint_load : Restore data and split as well.
    """
    bundle = load_pipeline_bundle(path, trusted=trusted)
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
    return cast("Session", session)
def predict_from_pipeline(
    session,
    path: str | Path,
    data: Dataset | pd.DataFrame | None = None,
    *,
    roles: dict[str, ColumnRole | str] | None = None,
    return_proba: bool = False,
    apply_plans: bool = True,
    trusted: bool = False,
) -> PipelinePredictResult:
    """Score new rows through a saved bundle, in one call.

    This is the inference path. Point it at a directory written by
    :meth:`save_pipeline` and give it rows to score; it loads the model and
    its preprocessing plans, transforms the rows exactly as training did,
    and returns the predictions.

    Nothing on the session changes — not the dataset, not
    :attr:`fit_result`, not the plans. That makes it safe to call inside a
    batch job or a service handler, repeatedly, against different bundles,
    without one call contaminating the next.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    path:
        The pipeline bundle directory to score through.
    data:
        The rows to score, as a Dataset or a plain DataFrame. ``None`` uses
        this session's dataset.
    roles:
        Column roles to apply when ``data`` is a bare DataFrame with no
        role information of its own. Needed when the bundle's
        preprocessing distinguishes features from identifiers.
    return_proba:
        Return class probabilities instead of chosen labels, where the
        estimator supports it. Use this when a downstream decision applies
        its own threshold rather than accepting the default cut-off.
    apply_plans:
        Replay the bundle's preprocessing before predicting. Leave on
        unless your incoming rows are already fully transformed — turning
        it off on raw data feeds the model inputs it cannot interpret and
        produces confident nonsense rather than an error.
    trusted:
        Must be ``True`` to deserialize the bundle's pickle/joblib payloads.
        Pass only for artifacts you created or fully trust.

    Returns
    -------
    ~buildml.pipeline.score.PipelinePredictResult
        The predictions plus the context needed to trust them: which
        preprocessing steps ran, how many rows were scored, and any
        warnings about the incoming data.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The bundle cannot be read, or the incoming rows are missing a
        column the model or its plans require.

    Notes
    -----
    A column present at training and absent now is caught here rather than
    producing a silently wrong answer. Schema drift between training and
    serving is among the most common production failures, and this is where
    it surfaces.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> incoming = pd.DataFrame({"tenure": [4], "plan": ["basic"]})  # doctest: +SKIP
    >>> result = Session.ingest(incoming).predict_from_pipeline(
    ...     "artifacts/churn_v3", return_proba=True, trusted=True
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.save_pipeline : Create the bundle this reads.
    Session.apply_preprocess_plans : The transform half, on its own.
    Session.serve_bundle : Expose a bundle over HTTP instead.
    """
    if data is None:
        if session._dataset is None:
            raise ValidationError("No dataset attached. Ingest data or pass data=...")
        score_data: Dataset | pd.DataFrame = session.dataset
    else:
        score_data = data
    result = run_predict_from_pipeline(
        path,
        score_data,
        roles=roles,
        return_proba=return_proba,
        apply_plans=apply_plans,
        trusted=trusted,
    )
    session._record(
        "predict_from_pipeline",
        {
            "path": str(path),
            "return_proba": return_proba,
            "apply_plans": apply_plans,
            "trusted": trusted,
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
    """Narrow the data in the engine before pulling it into memory.

    scikit-learn needs an in-memory matrix, which is a problem when the
    table is larger than memory. The way through is to do the reduction
    where the data already lives: let Polars or DuckDB select just the
    columns you need and, if necessary, sample the rows, so that only the
    reduced result crosses into Pandas.

    This matters only when :meth:`with_engine` has attached a native
    engine. On plain Pandas the data is already in memory and this is a
    no-op with bookkeeping.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        Which partition to prepare. Defaults to ``'train'``, the one that
        usually needs to fit in memory for a fit call.
    columns:
        Which columns to project. ``None`` selects the feature and target
        columns for the partition, which is what a fit needs and nothing
        more.
    sample_rows:
        Cap the result at this many rows, drawn at random. ``None`` keeps
        all of them. Sampling makes an oversized partition trainable, at
        the cost of learning from less of it — the sample is recorded in
        the disclosures so the compromise stays visible.
    random_state:
        Seed for the sampling, so the same subset is drawn each run.

    Returns
    -------
    ~buildml.data.engines.prep.MaterializePrepResult
        The prepared matrix together with disclosures recording which
        columns were projected and whether rows were sampled.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, the partition is not part of it, or a named column
        is absent.

    Notes
    -----
    Projection and sampling reduce what must be materialised; they do not
    make scikit-learn out-of-core. The estimator still receives an
    in-memory matrix. If the reduced partition still does not fit, reach
    for :meth:`fit_online`, which learns incrementally from batches.

    See Also
    --------
    Session.with_engine : Attach the engine that makes this worthwhile.
    Session.fit_online : Train without holding all the data at once.
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
    """Check whether predicted probabilities mean what they claim.

    A classifier that outputs ``0.8`` is asserting that eight out of ten
    such cases are positive. Often that is simply false — the model ranks
    cases correctly while its probabilities are systematically too
    confident or too timid. Ranking quality (ROC-AUC) cannot detect this,
    because rescaling every probability leaves the ranking untouched.

    Calibration matters whenever the number itself is used rather than just
    the ordering: expected-value calculations, risk thresholds, or anything
    shown to a person who will read "80%" as eighty percent. This groups
    predictions into probability bands and compares each band's claimed
    rate against the rate actually observed.

    You get the Brier score (mean squared error of the probabilities),
    expected calibration error (the average gap between claimed and
    observed), and the reliability curve points behind both. A perfectly
    calibrated model traces the diagonal; sagging below it means
    overconfidence.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        Which rows to assess. Calibration must be measured on data the
        model did not learn from — on training rows almost any model looks
        well calibrated.
    export_figures:
        Directory to write the reliability diagram into. Requires
        ``pip install 'buildml[viz]'``.
    export_html:
        Path for a self-contained HTML version of the same.

    Returns
    -------
    ~buildml.model.diagnostics.DiagnosticReport
        The calibration findings: Brier score, expected calibration error,
        reliability curve points, and an interpretation of what the shape
        implies.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted, the fitted model is not a classifier, or
        it does not expose ``predict_proba``.
    ~buildml.core.errors.MissingExtraError
        Figures were requested without ``buildml[viz]`` installed.

    Notes
    -----
    Poor calibration is usually fixable without retraining, by fitting a
    small correction from probability to observed rate on held-out data
    (Platt scaling or isotonic regression). Note which models tend to need
    it: naive Bayes is famously overconfident, boosted trees push
    probabilities toward the extremes, and a random forest averaging many
    votes is typically already close.

    Examples
    --------
    >>> report = session.calibration(partition="validation")  # doctest: +SKIP
    >>> report.metrics["brier_score"]  # doctest: +SKIP
    0.084

    See Also
    --------
    Session.tune_threshold : Choose the cut-off these probabilities feed.
    Session.eval_plots : The reliability curve alongside other panels.
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
    """Choose the cut-off that turns a probability into a decision.

    A classifier outputs a probability, but acting on it requires a line:
    above this, treat as positive. The default line is 0.5, and 0.5 is
    almost never the right answer — it silently assumes that a false alarm
    and a miss cost the same amount.

    They rarely do. Missing a fraudulent transaction costs the value of the
    fraud; flagging a legitimate one costs an annoyed customer. Missing a
    disease costs far more than an unnecessary follow-up test. The correct
    threshold follows from those costs, not from the midpoint of the
    probability range.

    This sweeps every candidate threshold and reports how precision,
    recall, and F1 move as the line shifts. Supply ``fp_cost`` and
    ``fn_cost`` and it goes further, computing expected cost at each
    threshold and identifying the one that minimises it — turning a
    modelling choice into an arithmetic one.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        Which rows to sweep over. Use ``'validation'`` while choosing;
        selecting a threshold on ``'test'`` and then reporting that
        partition's score means the score was tuned on the data it claims
        to be independent of.
    fp_cost:
        What one false positive costs — flagging something that was fine.
        Any consistent unit works; only the ratio to ``fn_cost`` affects
        the chosen threshold.
    fn_cost:
        What one false negative costs — missing something real. Must be
        given together with ``fp_cost``.
    tp_benefit:
        What correctly catching a positive is worth, subtracted from
        expected cost. Useful when a true positive earns something concrete
        rather than merely avoiding a loss.
    tn_benefit:
        What correctly leaving a negative alone is worth.
    export_figures:
        Directory to write the threshold sweep chart into. Requires
        ``pip install 'buildml[viz]'``.
    export_html:
        Path for a self-contained HTML version of the same.

    Returns
    -------
    ~buildml.model.diagnostics.DiagnosticReport
        The sweep: metrics at every candidate threshold, the recommended
        cut-off, and — when costs were supplied — the expected cost curve
        and its minimum.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted, the target is not binary, the model
        exposes no ``predict_proba``, or exactly one of ``fp_cost`` and
        ``fn_cost`` was supplied.
    ~buildml.core.errors.MissingExtraError
        Figures were requested without ``buildml[viz]`` installed.

    Notes
    -----
    The threshold you pick here is part of the model as deployed. Record it
    with the pipeline, because a bundle scored at 0.5 when it was tuned for
    0.18 will behave nothing like the version you evaluated.

    A cost-optimal threshold is only as good as the costs. If they are
    guesses, look at how flat the cost curve is around its minimum: a flat
    region means the exact number hardly matters, and a sharp one means
    your guess needs to be right.

    Examples
    --------
    >>> report = session.tune_threshold(
    ...     partition="validation", fp_cost=1.0, fn_cost=12.0
    ... )  # doctest: +SKIP
    >>> report.metrics["best_threshold"]  # doctest: +SKIP
    0.18

    See Also
    --------
    Session.calibration : Confirm the probabilities are trustworthy first.
    Session.predict : Obtain probabilities to apply the chosen cut-off to.
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
    """Find out whether more data would help, before you go and get it.

    When a model underperforms there are two very different remedies, and
    pursuing the wrong one wastes weeks. Either the model is too simple to
    capture the pattern, in which case more rows change nothing and you
    need a richer model or better features; or it is complex enough but
    starved of examples, in which case more rows are exactly what is
    needed.

    A learning curve distinguishes them. The model is refitted on
    increasing fractions of the training rows and scored each time, giving
    two lines: performance on the data it trained on, and performance on
    held-out data.

    Read them by their gap and their slope. A wide gap that is still
    closing as the curves extend rightward means more data will help. Two
    curves that have converged and flattened, both mediocre, mean the model
    has learned everything it can from these features — more rows will not
    move it. Converged and both excellent means you are done.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    estimator:
        An unfitted estimator to trace. Usually the same one you fitted, so
        the curve describes the model you are actually considering.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``.
    cv:
        Fold count used at each sample size, so every point on the curve is
        itself averaged rather than a single noisy measurement.
    export_figures:
        Directory to write the curve into. Requires
        ``pip install 'buildml[viz]'``.
    export_html:
        Path for a self-contained HTML version of the same.

    Returns
    -------
    ~buildml.model.diagnostics.DiagnosticReport
        The curve points at each training size, the train and validation
        scores at each, and an interpretation of what the shape implies.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, or the training partition is too small to
        subdivide.
    ~buildml.core.errors.MissingExtraError
        Figures were requested without ``buildml[viz]`` installed.

    Notes
    -----
    The model is refitted once per sample size per fold, so this is among
    the slower diagnostics. Lower ``cv`` for a quick read.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> report = session.learning_curve(
    ...     RandomForestClassifier(random_state=0), cv=5
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.eval_plots : The learning curve alongside other diagnostics.
    Session.cv_score : Score at full training size only.
    """
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
    """Measure which features the model genuinely depends on.

    The method is simple and that is its strength: take one feature, shuffle
    its values across rows so it keeps its distribution but loses its
    relationship to the target, and re-score. However far the score falls
    is how much the model was relying on that feature. Repeat for each
    feature.

    This works for any model — the internals are never inspected, only the
    predictions — so a neural network, a boosted ensemble, and a linear
    model can be compared on the same footing. It also avoids the known
    distortions of tree-based built-in importances, which systematically
    favour high-cardinality and continuous features regardless of whether
    they carry signal.

    Run it on held-out rows. Importance measured on training data tells you
    what the model memorised; importance on a holdout tells you what
    actually generalises, which is the question worth asking.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    partition:
        Which rows to measure on. Default ``'test'``; ``'validation'`` is
        the better choice if you intend to act on the result and want to
        keep test clean.
    n_repeats:
        How many times each feature is shuffled. Shuffling is random, so a
        single pass is noisy; more repeats give a steadier ranking at
        proportionally more time.
    export_figures:
        Directory to write the importance chart into. Requires
        ``pip install 'buildml[viz]'``.
    export_html:
        Path for a self-contained HTML version of the same.

    Returns
    -------
    ~buildml.model.diagnostics.DiagnosticReport
        Per-feature importance with the spread across repeats, ranked, plus
        an interpretation. The spread matters: a feature whose importance
        varies wildly between repeats has not been shown to matter.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted, or no split exists.
    ~buildml.core.errors.MissingExtraError
        Figures were requested without ``buildml[viz]`` installed.

    Notes
    -----
    Correlated features mislead this method, and it is worth knowing how.
    If two columns carry the same information, shuffling either one leaves
    the model able to recover the signal from the other, so both look
    unimportant — even though together they are essential. When you see a
    feature you expect to matter scoring near zero, check what it is
    correlated with before concluding it is useless.

    Importance is not causation. A feature the model leans on may be a
    symptom of the outcome rather than a cause of it, and intervening on it
    will change nothing. Use :meth:`fit_causal` when the question is what
    to act on.

    Examples
    --------
    >>> report = session.feature_importance(
    ...     partition="validation", n_repeats=20
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.select_features : Act on importance by dropping columns.
    Session.error_slices : Where the model fails, rather than on what.
    Session.fit_causal : What to change, rather than what predicts.
    """
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
    """Break performance down by subgroup, to find where the model fails.

    An overall score is an average, and averages conceal. A model at 92%
    accuracy might be at 97% for the large customer segment and 61% for the
    small one — a difference invisible in the headline number and highly
    visible to the people in that second group.

    This splits the scored rows by the columns you name and reports metrics
    for each segment alongside the overall figure, so the gaps become
    explicit. Slice by region, product line, customer tier, or any
    categorical column whose subgroups you would be unhappy to serve badly.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    by:
        One column name, or several. Passing several slices by their
        combination, which finds interaction failures — a model can be fine
        on each of two dimensions separately and poor on a particular
        intersection.
    partition:
        Which rows to slice. Use ``'validation'`` while exploring so test
        stays reserved.
    max_segments:
        Cap on how many segments to report, keeping a high-cardinality
        column from producing an unreadable table. The largest segments are
        kept.
    min_segment_n:
        Minimum rows for a segment to be reported as a finding. Below this,
        a metric is mostly noise — three rows and two errors is not a 67%
        error rate, it is three rows. Smaller segments are listed
        separately rather than discarded.
    export_html:
        Optional path to write an HTML report of segment findings.

    Returns
    -------
    ~buildml.model.diagnostics.DiagnosticReport
        Per-segment metrics and sizes, the segments that fell below
        ``min_segment_n`` under ``small_segments``, and an interpretation
        highlighting the largest gaps.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No model has been fitted, no split exists, or a named column is
        absent from the partition.

    Notes
    -----
    Observational only: segment gaps are not fairness proof. Prefer
    validation for exploration and keep test for a final estimate.
    Segments with ``n < min_segment_n`` are listed under ``small_segments``.

    A gap tells you where the model is worse, not why. Small segments have
    fewer training examples, may genuinely be harder to predict, and may
    have been measured differently. Any of those explains a gap without
    implicating the model. Treat the output as a list of places to
    investigate, not a verdict.

    Examples
    --------
    >>> report = session.error_slices(
    ...     by="region", partition="validation"
    ... )  # doctest: +SKIP

    Look for an interaction the single-column view would hide:

    >>> report = session.error_slices(by=["region", "product_tier"])  # doctest: +SKIP

    See Also
    --------
    Session.feature_importance : What the model uses, rather than where it
        fails.
    Session.evaluate : The aggregate this decomposes.
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
